"""Fitting one notch resonance in ``S21``.

The transmission past a resonator that is capacitively coupled to the
feedline is the asymmetric notch of Khalil et al. 2012 and Probst et al.
2015, Eq. (27) of the paper, here with the cable delay and the baseline of a
real measurement:

    S21(f) = a e^{i phi_bg} e^{-2 pi i tau f}
             [ 1 - e^{i alpha} (Qr / |Qc|) / (1 + 2 i Qr (f - fr) / fr) ],

with the loaded quality factor ``Qr``, the magnitude of the complex coupling
quality factor ``|Qc|`` (written ``Qc_abs`` throughout), the asymmetry angle
``alpha`` of the coupling, the cable delay ``tau`` and a complex baseline
``a e^{i phi_bg}``. The diameter-correction-method coupling Q that the CMT
model uses is ``Qc = |Qc| / cos(alpha)``, and ``1/Qi = 1/Qr - 1/Qc``. This is
the circle fit of Sec. IV B of the paper.

The fit is the three-stage circle fit of Probst et al.:

1. :func:`initial_fit` -- remove the cable delay and fit a circle (centre
   ``zc``, radius ``r``) to the data. Translate the data so that the circle
   centre is at the origin and rotate them by ``-arg(zc)``,
   ``z_t = (zc - s21') e^{-i arg zc}``. On ``z_t`` the resonance point
   ``f = fr`` lies near the positive real axis and the off-resonance end of
   the circle, which the points at the window edges approach, near the
   negative real axis (exactly on the axis for ``alpha = 0``).
2. :func:`fit_phase` -- fit the arctangent phase response of ``z_t`` for
   ``fr``, ``Qr`` and the phase offset ``theta0``; starting values of
   ``|Qc|`` and ``alpha`` follow from the circle geometry.
3. :func:`refined_fit` -- a bounded, weighted least-squares fit of the full
   complex model to the delay-corrected data, started from the results of
   the first two stages. Points where the phase turns fastest (the resonance
   itself) are weighted up so the baseline cannot pull the fit.

The cable delay is the awkward parameter: fitting it together with the
resonance is degenerate with the baseline phase. :func:`fit_resonance_trace`
removes it the way the paper's parameter table was produced -- a line fitted
to the phase in a band at each edge of the sweep, after dividing out the
fitted resonance, iterated to convergence. Fitting the raw edge phase would
pick up the resonance tail and bias ``Qr`` and ``|Qc|`` high by a few
percent.

Failures
--------
:func:`refined_fit`, :func:`fit_resonance` and :func:`fit_resonance_trace`
return NaNs, with a ``RuntimeWarning``, exactly where the original code
returned NaNs: when the phase fit of stage 2 or the full fit of stage 3 does
not converge within its evaluation budget, or the fitted ``Qr`` ends on a
bound (internally a :class:`NotchFitError`). Every other error propagates, as
it did in the original. In particular a ``ValueError`` is raised for data
the fit cannot take:

* a baseline above ``|S21| = 2``: the full fit bounds the baseline amplitude
  to ``0 <= a <= 2`` and starts it at the median ``|S21|`` of the outer 10%
  of the window at each edge, so data with more than about 6 dB of net gain
  must be normalised first;
* a window whose span is outside ``fr/5e5 ... fr/100``: the phase fit starts
  ``Qr`` at ``fr / span`` and bounds it to ``100 ... 5e5``;
* NaN or infinite samples;
* degenerate data on which no circle can be fitted (constant or collinear
  points: no positive eigenvalue in :func:`~umux_cmt.fitting.circle.fit_circle`).

All numerics in this module are those of the code the paper was produced
with; only plotting and printing were removed.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter

from .circle import fit_circle


class NotchFitError(RuntimeError):
    """A notch fit failed in a way :func:`refined_fit` answers with NaNs.

    The phase fit of stage 2 or the full fit of stage 3 did not converge, or
    the fitted ``Qr`` ended on a bound: the ``RuntimeError`` cases the
    original code caught. :func:`fit_resonance_stages` raises it;
    :func:`refined_fit`, :func:`fit_resonance` and :func:`fit_resonance_trace`
    catch it and return NaNs.
    """


###############################################################################
#                                 THE MODEL                                   #
###############################################################################

def notch_model(f, fr, Qr, Qc, alpha):
    """Normalised notch ``1 - e^{i alpha} (Qr/Qc) / (1 + 2i Qr (f - fr)/fr)``.

    The ``notch()`` of the paper's parameter notebook.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    fr, Qr, Qc, alpha : float
        Resonance frequency, loaded Q, ``|Qc|`` and asymmetry angle.

    Returns
    -------
    ndarray of complex
        ``S21`` without baseline and delay.
    """
    return 1 - np.exp(1j * alpha) * (Qr / Qc) / (1 + 2j * Qr * (f - fr) / fr)


def phase_model(f, fr, Qr, theta0):
    """Phase of the translated circle, ``-theta0 + 2 arctan(2 Qr (1 - f/fr))``.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    fr, Qr, theta0 : float
        Resonance frequency, loaded Q and phase offset in rad.

    Returns
    -------
    ndarray
        Phase in rad.
    """
    return -theta0 + 2 * np.arctan(2 * Qr * (1 - f / fr))


def _t21_p_func(f, a, bg_phase, fr, Qr, Qc, alpha):
    """Delay-corrected model: baseline ``a e^{i bg_phase}`` times the notch."""
    baseline = a * np.exp(1.0j * bg_phase)
    resonator_term = (np.exp(1.0j * alpha) * Qr / Qc) / (1 + 2.0j * Qr * (f - fr) / fr)
    return baseline * (1 - resonator_term)


def _t21_wrapper_func(f_stacked, a, bg_phase, fr, Qr, Qc, alpha):
    """The model as a real vector ``[Re, Im]`` for ``curve_fit``."""
    N = len(f_stacked) // 2
    f_unique = f_stacked[:N]
    t21_complex = _t21_p_func(f_unique, a, bg_phase, fr, Qr, Qc, alpha)
    return np.concatenate((np.real(t21_complex), np.imag(t21_complex)))


###############################################################################
#                              CABLE DELAY                                    #
###############################################################################

def calculate_tau(f, s21):
    """Cable delay from the mean phase slope of a wide sweep.

    A line is fitted to the unwrapped phase over the whole trace; the slope is
    ``-2 pi tau``. Meant for a sweep that is wide compared with the resonances
    in it (the full comb), where the resonances average out. For a single
    resonance use :func:`fit_resonance_trace`.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    s21 : ndarray of complex
        Transmission.

    Returns
    -------
    float
        Delay in s.
    """
    phase = np.unwrap(np.angle(s21))
    slope, intercept = np.polyfit(f, phase, 1)
    tau = -slope / (2 * np.pi)
    return tau


###############################################################################
#                        STAGE 1 AND 2: CIRCLE AND PHASE                      #
###############################################################################

def initial_fit(f_region, s21_region, tau, full_output=False):
    """Circle fit and canonical rotation of a delay-corrected resonance.

    The delay ``tau`` is removed and a circle (centre ``zc``, radius ``r``)
    is fitted algebraically to the delay-corrected data ``s21'``. The data
    are then translated so that the circle centre is at the origin and
    rotated by ``-arg(zc)``, ``z_t = (zc - s21') e^{-i arg zc}``. ``zc``,
    ``z_t`` and ``r`` are all the later stages use; ``r`` sets ``|Qc|`` in
    :func:`fit_phase`.

    As in the original code, a second circle is fitted to ``z_t``
    (algebraically; replaced by the centre of mass and half the largest
    ``|z_t|`` if its radius is below 5% or above 5 times the largest
    ``|z_t|``; then refined geometrically by a bounded ``soft_l1``
    least-squares fit). It does not enter any result and is returned only
    with ``full_output=True``, for diagnostic plots.

    Parameters
    ----------
    f_region : ndarray
        Frequencies in Hz.
    s21_region : ndarray of complex
        Raw transmission in the window.
    tau : float
        Cable delay to remove, in s.
    full_output : bool, optional
        Also return the refined circle of ``z_t``.

    Returns
    -------
    zc : complex
        Centre of the circle fitted to the delay-corrected data.
    z_t : ndarray of complex
        Translated and rotated data.
    r : float
        Radius of the circle fitted to the delay-corrected data.
    circle_t : tuple of float
        ``(xc_t, yc_t, r_t)``, the refined circle of ``z_t``; only with
        ``full_output=True``.

    Raises
    ------
    ValueError
        From :func:`~umux_cmt.fitting.circle.fit_circle`, for degenerate or
        non-finite data.
    """
    arg = 2 * np.pi * tau * f_region
    s21_p = s21_region * (1.0 * np.cos(arg) + 1.0j * np.sin(arg))
    s21_p_real = np.real(s21_p)
    s21_p_imag = np.imag(s21_p)

    # Algebraic guess
    xc, yc, r = fit_circle(s21_p_real, s21_p_imag)
    zc = 1.0 * xc + 1.0j * yc
    zc_angle = np.angle(zc)
    z_t = (zc - s21_p) * np.exp(-1j * zc_angle)

    xc_t_alg, yc_t_alg, r_t_alg = fit_circle(np.real(z_t), np.imag(z_t))

    # Sanity check: a radius far outside the spread of the data means noise
    # broke the eigenvalue problem; fall back to the centre of mass.
    data_spread = np.max(np.abs(z_t))
    if r_t_alg < data_spread * 0.05 or r_t_alg > data_spread * 5:
        xc_t_alg = np.mean(np.real(z_t))
        yc_t_alg = np.mean(np.imag(z_t))
        r_t_alg = data_spread / 2.0

    # Bounded geometric refinement, kept close to the algebraic guess
    def calc_distance(c, x, y):
        return np.sqrt((x - c[0])**2 + (y - c[1])**2) - c[2]

    bounds_lower = [xc_t_alg - 2*r_t_alg, yc_t_alg - 2*r_t_alg, 0.2*r_t_alg]
    bounds_upper = [xc_t_alg + 2*r_t_alg, yc_t_alg + 2*r_t_alg, 5.0*r_t_alg]

    try:
        res = least_squares(
            calc_distance,
            x0=[xc_t_alg, yc_t_alg, r_t_alg],
            args=(np.real(z_t), np.imag(z_t)),
            bounds=(bounds_lower, bounds_upper),
            loss='soft_l1'
        )
        xc_t, yc_t, r_t = res.x
    except Exception:
        xc_t, yc_t, r_t = xc_t_alg, yc_t_alg, r_t_alg

    if full_output:
        return zc, z_t, r, (xc_t, yc_t, r_t)
    return zc, z_t, r


def fit_phase(f_region, z_t, zc, r):
    """Fit the arctangent phase of the canonically rotated circle.

    The phase of ``z_t`` is fitted with :func:`phase_model`,
    ``-theta0 + 2 arctan(2 Qr (1 - f/fr))``. The fit is bounded (``fr``
    inside the window, ``100 <= Qr <= 5e5``, ``|theta0| <= pi``) and started
    from the point of smallest ``|phase|``, with ``Qr = fr / span``. From the
    circle geometry ``|Qc| = Qr (|zc| + r) / (2 r)`` and
    ``alpha = theta0 - arg zc``.

    The phase is taken from ``np.angle`` without unwrapping, as in the code
    the paper was produced with. For asymmetry angles beyond about 0.2 rad the
    phase can cross +-pi inside the window, and this stage then starts the
    full fit from a wrong minimum (large chi-squared, ``Qr`` far off); narrow
    the window or rotate the data if that happens on your device.

    The ``|Qc|`` and ``alpha`` of this stage only start the full fit of
    :func:`refined_fit`, which reports the final values. With these sign
    conventions this stage's ``alpha`` comes out as ``-(alpha + phi_bg)``
    for a circle whose phase does not wrap inside the window, so it can
    differ from the final ``alpha`` even in sign.

    Parameters
    ----------
    f_region : ndarray
        Frequencies in Hz.
    z_t : ndarray of complex
        Translated, rotated data from :func:`initial_fit`.
    zc : complex
        Circle centre from :func:`initial_fit`.
    r : float
        Circle radius from :func:`initial_fit`.

    Returns
    -------
    params : list
        ``[fr, Qr, theta0, Qc_abs, alpha]``; all NaN if the fit did not
        converge.
    params_unc : list
        One-sigma uncertainties of ``fr``, ``Qr`` and ``theta0`` from the
        covariance (``Qc_abs`` and ``alpha`` are NaN).

    Raises
    ------
    ValueError
        If the starting ``Qr = fr / span`` is outside ``100 ... 5e5``, or the
        data are not finite.
    """
    phase = np.angle(z_t)
    phase_func = phase_model

    fr_idx = np.argmin(np.abs(phase))
    fr_guess = f_region[fr_idx]

    bw_guess = f_region[-1] - f_region[0]
    Qr_guess = fr_guess / bw_guess if bw_guess > 0 else 10000.0

    theta0_guess = phase[fr_idx]

    p0 = [fr_guess, Qr_guess, theta0_guess]

    bounds = (
        [f_region[0], 100.0, -np.pi],
        [f_region[-1], 500000.0, np.pi]
    )

    try:
        popt, pcov = curve_fit(
            phase_func,
            f_region,
            phase,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
    except RuntimeError:
        return [np.nan] * 5, [np.nan] * 5

    fr_fit, Qr_fit, theta0_fit = popt
    perr = np.sqrt(np.diag(pcov))
    fr_err, Qr_err, theta0_err = perr

    Qc_fit = Qr_fit * (np.abs(zc) + r) / (2 * r)
    alpha_fit = theta0_fit - np.angle(zc)
    Qc_err = np.nan
    alpha_err = np.nan

    params = [fr_fit, Qr_fit, theta0_fit, Qc_fit, alpha_fit]
    params_unc = [fr_err, Qr_err, theta0_err, Qc_err, alpha_err]
    return params, params_unc


###############################################################################
#                         STAGE 3: FULL COMPLEX FIT                           #
###############################################################################

def _refined_fit(f_region, s21_region, tau):
    """Body of :func:`refined_fit`; raises :class:`NotchFitError` instead of returning NaNs.

    The control flow is that of the original ``refined_fit``: errors of the
    circle and phase stages propagate, and a ``RuntimeError`` in the full
    fit -- the case the original caught and answered with NaNs -- is raised
    as :class:`NotchFitError`.

    Returns
    -------
    popt : ndarray
        ``[a, bg_phase, fr, Qr, Qc_abs, alpha]``.
    chi2 : float
        Reduced chi-squared of the fit, from :func:`residues`.
    extras : dict
        ``z_t`` and the phase-fit parameters, for the diagnostic plots.
    """
    # 1. Delay correction: strip the electrical delay so the circle stands still
    arg = 2 * np.pi * tau * f_region
    s21_p = s21_region * np.exp(1.0j * arg)

    # Stack the delay-corrected data for the optimiser
    s21_stacked_real = np.concatenate((np.real(s21_p), np.imag(s21_p)))
    f_stacked = np.concatenate((f_region, f_region))

    # 2. Algebraic guesses and the phase fit (outside the try, as in the
    # original: their errors propagate)
    zc, z_t, r, circle_t = initial_fit(f_region, s21_region, tau, full_output=True)
    params, _ = fit_phase(f_region, z_t, zc, r)

    fr_fit_phase, Qr_fit_phase, theta0_fit_phase, Qc_fit_phase, alpha_fit_phase = params

    try:
        if np.isnan(fr_fit_phase):
            raise RuntimeError("Phase fitting failed.")

        # 3. Guess sanitisation and bounds. fr is kept inside the window: the
        # caller placed the window on a detected dip, so the resonance is in here
        # by construction, and wider bounds let the optimiser fit the baseline
        # with the far tail of a resonance that is not in the data at all.
        alpha_guess = (alpha_fit_phase + np.pi) % (2 * np.pi) - np.pi

        fr_min = f_region[0]
        fr_max = f_region[-1]
        fr_guess = np.clip(fr_fit_phase, fr_min + 1, fr_max - 1)

        Qr_guess = np.clip(Qr_fit_phase, 11.0, 0.99e7)
        Qc_guess = np.clip(Qc_fit_phase, 11.0, 0.99e7)

        num_edge_points = max(1, int(len(s21_region) * 0.1))

        # Baseline amplitude and phase from the outer 10% of points at each edge
        edge_complex = np.concatenate([s21_p[:num_edge_points], s21_p[-num_edge_points:]])
        a_guess = np.median(np.abs(edge_complex))
        bg_phase_guess = np.angle(np.mean(edge_complex))

        params_guess = [a_guess, bg_phase_guess, fr_guess, Qr_guess, Qc_guess, alpha_guess]

        # Order: [a, bg_phase, fr, Qr, Qc, alpha]
        bounds_lower = [0.0, -np.pi, fr_min, 10.0, 10.0, -2 * np.pi]
        bounds_upper = [2.0, np.pi, fr_max, 1e7, 1e7, 2 * np.pi]

        # 4. Phase-derivative weighting: points where the (smoothed) phase turns
        # fastest get sigma = 0.2, the flat baseline sigma = 1.
        phase_unwrapped = np.unwrap(np.angle(z_t))
        window_size = min(21, len(phase_unwrapped) - (1 if len(phase_unwrapped) % 2 == 0 else 0))

        if window_size > 3:
            phase_smoothed = savgol_filter(phase_unwrapped, window_length=window_size, polyorder=3)
        else:
            phase_smoothed = phase_unwrapped

        dphase_df = np.abs(np.gradient(phase_smoothed, f_region))

        max_dphase = np.max(dphase_df)
        if max_dphase > 0:
            weights = dphase_df / max_dphase
        else:
            weights = np.zeros_like(dphase_df)

        sigma_1d = 1.0 - (0.8 * weights)
        sigma_stacked = np.concatenate((sigma_1d, sigma_1d))

        # 5. The fit
        popt, pcov = curve_fit(_t21_wrapper_func, f_stacked, s21_stacked_real,
                               p0=params_guess,
                               sigma=sigma_stacked,
                               absolute_sigma=False,
                               bounds=(bounds_lower, bounds_upper),
                               maxfev=5000)

        a_fine, bg_phase_fine, fr_fine, Qr_fine, Qc_fine, alpha_fine = popt

        if Qr_fine <= 10 or Qr_fine >= 1e7:
            raise RuntimeError("Fitted Qr hit physical boundaries or is invalid.")

        # Map the fit back to the raw, delay-carrying data space for the residuals
        t21_p_fit = _t21_p_func(f_region, *popt)
        t21_fit = t21_p_fit * np.exp(-1.0j * arg)

        chi2 = residues(f_region, s21_region, t21_fit)
    except RuntimeError as e:
        raise NotchFitError(str(e)) from e

    extras = dict(z_t=z_t, zc=zc, r=r, circle_t=circle_t, s21_corrected=s21_p,
                  phase_params=(fr_fit_phase, Qr_fit_phase, theta0_fit_phase))
    return popt, chi2, extras


def refined_fit(f_region, s21_region, tau):
    """Full complex notch fit of one resonance window.

    Removes the delay ``tau``, seeds the fit with :func:`initial_fit` and
    :func:`fit_phase`, and fits the six-parameter model
    ``a e^{i bg_phase} [1 - e^{i alpha} (Qr/Qc)/(1 + 2i Qr (f-fr)/fr)]`` to the
    real and imaginary parts with the bounds ``0 <= a <= 2``,
    ``|bg_phase| <= pi``, ``fr`` inside the window, ``10 <= Qr, Qc <= 1e7``,
    ``|alpha| <= 2 pi`` and the phase-derivative weighting described in the
    module docstring.

    Parameters
    ----------
    f_region : ndarray
        Frequencies in Hz, covering one resonance.
    s21_region : ndarray of complex
        Raw transmission in the window.
    tau : float
        Cable delay in s (``0`` if the data are already delay-corrected).

    Returns
    -------
    (a, tau, fr, Qr, Qc_abs, alpha, chi2) : tuple
        Baseline amplitude, the input delay, resonance frequency, loaded Q,
        ``|Qc|``, asymmetry angle and reduced chi-squared. All NaN (with a
        ``RuntimeWarning``) if the phase fit or the full fit does not
        converge or ``Qr`` ends on a bound.

    Raises
    ------
    ValueError
        For data the fit cannot take (baseline ``|S21| > 2``, a window span
        outside ``fr/5e5 ... fr/100``, non-finite or degenerate data); see
        the module docstring. The original code raised in the same cases.

    See Also
    --------
    fit_resonance : the same fit returned as a :class:`NotchFit`.
    """
    try:
        popt, chi2, _ = _refined_fit(f_region, s21_region, tau)
    except NotchFitError as e:
        warnings.warn(f"notch fit failed for this window, returning NaNs ({e})", RuntimeWarning, stacklevel=2)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    a_fine, bg_phase_fine, fr_fine, Qr_fine, Qc_fine, alpha_fine = popt
    return a_fine, tau, fr_fine, Qr_fine, Qc_fine, alpha_fine, chi2


def residues(f_region, s21_region, t21_fit):
    """Reduced chi-squared of a complex fit.

    The noise variance is estimated from the first 2% of the trace as half the
    mean squared difference of neighbouring points (so that the resonance does
    not enter it), and the residuals are counted in both quadratures with
    ``k = 6`` fitted parameters.

    Parameters
    ----------
    f_region : ndarray
        Frequencies in Hz.
    s21_region, t21_fit : ndarray of complex
        Data and model on the same grid.

    Returns
    -------
    float
        ``chi2 / (2 n - 6)``.
    """
    res = s21_region - t21_fit
    n_complex_points = len(res)

    m = int(len(f_region) * 0.02)
    if m >= len(s21_region):
        m = len(s21_region) - 1

    sigmaz2 = np.sum(np.abs(np.diff(s21_region[:m + 1]))**2) / (2 * m)
    sigma_real_sq = sigmaz2 / 2.0
    k = 6
    dof = 2 * n_complex_points - k
    chi2_val = np.sum(np.abs(res)**2) / sigma_real_sq
    reduced_chi2 = chi2_val / dof

    return reduced_chi2


###############################################################################
#                              RESULT CLASSES                                 #
###############################################################################

@dataclass
class NotchFit:
    """Result of :func:`fit_resonance`: one notch fitted in one window.

    The fields hold what :func:`refined_fit` returns,
    ``(a, tau, fr, Qr, Qc_abs, alpha, chi2)``, plus the baseline phase; all
    are NaN if the fit failed. ``Qc_abs`` is the fitted magnitude ``|Qc|``;
    the DCM coupling Q the simulation uses is :attr:`Qc_dcm`.
    """

    a: float
    """Baseline amplitude."""
    tau: float
    """Cable delay that was removed before the fit, in s."""
    fr: float
    """Resonance frequency in Hz."""
    Qr: float
    """Loaded quality factor."""
    Qc_abs: float
    """Magnitude of the complex coupling quality factor, ``|Qc|``."""
    alpha: float
    """Asymmetry angle of the coupling in rad."""
    chi2: float
    """Reduced chi-squared, see :func:`residues`."""
    bg_phase: float = np.nan
    """Baseline phase in rad."""

    def __repr__(self):
        return (f"NotchFit(fr={self.fr / 1e6:.4f} MHz, Qr={self.Qr:.0f}, Qc_abs={self.Qc_abs:.0f}, "
                f"alpha={self.alpha:.4f}, chi2={self.chi2:.2f})")

    @property
    def ok(self):
        """``True`` unless the fit failed (NaNs)."""
        return not np.isnan(self.fr)

    @property
    def Qc_dcm(self):
        """Diameter-correction-method coupling Q, ``|Qc| / cos(alpha)``."""
        return self.Qc_abs / np.cos(self.alpha)

    @property
    def Qi(self):
        """Internal quality factor (DCM), ``1 / (1/Qr - 1/Qc_dcm)``."""
        return 1 / (1 / self.Qr - 1 / self.Qc_dcm)

    @property
    def linewidth(self):
        """Resonator bandwidth ``fr / Qr`` in Hz."""
        return self.fr / self.Qr

    def as_tuple(self):
        """``(fr, Qr, Qc_abs, alpha)``, the resonator parameters.

        The same order as :meth:`TraceFit.as_tuple`. The full tuple of
        :func:`refined_fit` is ``(a, tau, fr, Qr, Qc_abs, alpha, chi2)``.
        """
        return self.fr, self.Qr, self.Qc_abs, self.alpha

    def model(self, f, with_delay=True):
        """The fitted ``S21`` at frequencies ``f``.

        Parameters
        ----------
        f : ndarray
            Frequencies in Hz.
        with_delay : bool, optional
            Include the delay factor ``e^{-2 pi i tau f}`` (the model of the
            raw data). ``False`` gives the model of the delay-corrected data.

        Returns
        -------
        ndarray of complex
        """
        f = np.asarray(f, float)
        s = _t21_p_func(f, self.a, self.bg_phase, self.fr, self.Qr, self.Qc_abs, self.alpha)
        if with_delay:
            s = s * np.exp(-2.0j * np.pi * self.tau * f)
        return s


def fit_resonance(f, s21, tau=0.0):
    """Fit one notch resonance; :func:`refined_fit` returned as a :class:`NotchFit`.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz, covering one resonance.
    s21 : ndarray of complex
        Raw transmission in the window.
    tau : float, optional
        Cable delay to remove, in s.

    Returns
    -------
    NotchFit
        All fields NaN (and a ``RuntimeWarning``) where :func:`refined_fit`
        returns NaNs.

    Raises
    ------
    ValueError
        Where :func:`refined_fit` raises.
    """
    try:
        popt, chi2, _ = _refined_fit(f, s21, tau)
    except NotchFitError as e:
        warnings.warn(f"notch fit failed for this window, returning NaNs ({e})", RuntimeWarning, stacklevel=2)
        return NotchFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    a, bg_phase, fr, Qr, Qc_abs, alpha = popt
    return NotchFit(a, tau, fr, Qr, Qc_abs, alpha, chi2, bg_phase)


@dataclass
class NotchStages:
    """The intermediate results of the circle fit, for illustrating the method.

    Returned by :func:`fit_resonance_stages`; drawn by
    :func:`umux_cmt.fitting.plots.circle_stages`.
    """

    fit: NotchFit
    """The final fit."""
    f: np.ndarray = field(repr=False)
    """Frequencies in Hz."""
    s21_corrected: np.ndarray = field(repr=False)
    """Delay-corrected data ``s21 e^{2 pi i tau f}``."""
    zc: complex
    """Centre of the circle fitted to ``s21_corrected``."""
    r: float
    """Its radius."""
    z_t: np.ndarray = field(repr=False)
    """The data translated to the circle centre and rotated, ``(zc - s21') e^{-i arg zc}``."""
    circle_t: tuple
    """``(xc, yc, r)`` of the geometrically refined circle of ``z_t``."""
    phase_params: tuple
    """``(fr, Qr, theta0)`` of the phase fit."""

    def phase_model(self, f):
        """The fitted phase of ``z_t`` at frequencies ``f``.

        Parameters
        ----------
        f : ndarray
            Frequencies in Hz.

        Returns
        -------
        ndarray
            :func:`phase_model` with the parameters of the phase fit, in rad.
        """
        return phase_model(np.asarray(f, float), *self.phase_params)


def fit_resonance_stages(f, s21, tau=0.0):
    """:func:`fit_resonance` with every intermediate stage of the circle fit kept.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz, covering one resonance.
    s21 : ndarray of complex
        Raw transmission in the window.
    tau : float, optional
        Cable delay to remove, in s.

    Returns
    -------
    NotchStages
        The final :class:`NotchFit` with the circle of the delay-corrected
        data, the translated and rotated data and the phase fit.

    Raises
    ------
    NotchFitError
        Where :func:`fit_resonance` returns NaNs.
    ValueError
        Where :func:`refined_fit` raises.
    """
    popt, chi2, ex = _refined_fit(f, s21, tau)
    a, bg_phase, fr, Qr, Qc_abs, alpha = popt
    fit = NotchFit(a, tau, fr, Qr, Qc_abs, alpha, chi2, bg_phase)
    return NotchStages(fit=fit, f=np.asarray(f, float), s21_corrected=ex["s21_corrected"], zc=complex(ex["zc"]),
                       r=float(ex["r"]), z_t=ex["z_t"], circle_t=tuple(float(x) for x in ex["circle_t"]),
                       phase_params=tuple(float(x) for x in ex["phase_params"]))


@dataclass
class TraceFit:
    """Result of :func:`fit_resonance_trace`: a single-resonance sweep with the delay removed."""

    fr: float
    """Resonance frequency in Hz."""
    Qr: float
    """Loaded quality factor."""
    Qc_abs: float
    """``|Qc|``."""
    alpha: float
    """Asymmetry angle in rad."""
    tau: float
    """Cable delay removed from ``s21_corrected``, in s."""
    s21_corrected: np.ndarray = field(repr=False)
    """The whole trace with the delay line (slope and intercept of the edge-band phase fit) removed: the data the
    notch was fitted to. The fitted baseline amplitude ``a`` and residual phase ``bg_phase`` are still in it."""
    window: np.ndarray = field(repr=False)
    """Boolean mask of the points that entered the notch fit (``half_window`` around the dip)."""
    a: float = np.nan
    """Baseline amplitude of the notch fit."""
    bg_phase: float = np.nan
    """Residual baseline phase of the notch fit in rad."""
    chi2: float = np.nan
    """Reduced chi-squared of the notch fit."""
    n_iter: int = 0
    """Delay iterations that were run."""
    converged: bool = False
    """Whether the delay changed by less than the tolerance on the last iteration."""

    def __repr__(self):
        return (f"TraceFit(fr={self.fr / 1e6:.4f} MHz, Qr={self.Qr:.0f}, Qc_abs={self.Qc_abs:.0f}, "
                f"alpha={self.alpha:.4f}, tau={self.tau * 1e9:.2f} ns, n_iter={self.n_iter}, converged={self.converged})")

    @property
    def ok(self):
        """``True`` unless the notch fit failed (NaNs)."""
        return not np.isnan(self.fr)

    @property
    def Qc_dcm(self):
        """Diameter-correction-method coupling Q, ``|Qc| / cos(alpha)``."""
        return self.Qc_abs / np.cos(self.alpha)

    @property
    def Qi(self):
        """Internal quality factor (DCM), ``1 / (1/Qr - 1/Qc_dcm)``."""
        return 1 / (1 / self.Qr - 1 / self.Qc_dcm)

    @property
    def linewidth(self):
        """``fr / Qr`` in Hz."""
        return self.fr / self.Qr

    def as_tuple(self):
        """``(fr, Qr, Qc_abs, alpha)``, what the parameter notebook's ``fit_trace`` returned.

        The same order as :meth:`NotchFit.as_tuple`.
        """
        return self.fr, self.Qr, self.Qc_abs, self.alpha

    def model(self, f):
        """The fitted notch (baseline included) against ``s21_corrected`` at frequencies ``f``.

        Parameters
        ----------
        f : ndarray
            Frequencies in Hz.

        Returns
        -------
        ndarray of complex
        """
        return _t21_p_func(np.asarray(f, float), self.a, self.bg_phase, self.fr, self.Qr, self.Qc_abs, self.alpha)


def fit_resonance_trace(f, s21, edge_band=1.2e6, half_window=1.5e6, exclude_dip=1.5e6, max_iter=8, tol=5e-12):
    """Fit one resonance in a sweep whose cable delay is unknown.

    The delay is a line fitted to the unwrapped phase in a band of
    ``edge_band`` at each edge of the sweep. Because the tail of the
    resonance reaches into the edges, the fitted notch is divided out of the
    trace before the line is fitted, and the two steps are iterated until the
    delay changes by less than ``tol``. On the first pass, before any notch is
    known, the points within ``exclude_dip`` of the minimum of ``|S21|`` are
    left out of the edge bands (which matters for narrow spans). The notch
    itself is fitted with :func:`refined_fit` on the points within
    ``half_window`` of the dip of the delay-corrected trace.

    This is the ``fit_trace`` of the paper's parameter notebook, which
    extracted the parameters of Table I and the configuration files from
    every flux point of the VNA sweeps (with the default arguments).

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    s21 : ndarray of complex
        Raw transmission of one sweep around one resonance.
    edge_band : float, optional
        Width of the band at each edge used for the delay line, in Hz.
    half_window : float, optional
        Half-width of the notch-fit window around the dip, in Hz.
    exclude_dip : float, optional
        Half-width around the raw dip excluded from the edge bands on the first pass, in Hz.
    max_iter : int, optional
        Maximum number of delay iterations.
    tol : float, optional
        Convergence tolerance on the delay, in s.

    Returns
    -------
    TraceFit
        ``fr``, ``Qr``, ``Qc_abs`` and ``alpha`` are NaN (with a
        ``RuntimeWarning``) if the notch fit returned NaNs on the current
        iteration, which ends the iteration as in the original.

    Raises
    ------
    ValueError
        Where :func:`refined_fit` raises (data the notch fit cannot take),
        or if ``max_iter < 1``.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    edge = (f < f[0] + edge_band) | (f > f[-1] - edge_band)
    f_dip = f[np.argmin(np.abs(s21))]
    shape, tau = np.ones_like(s21), np.nan
    fr = Qr = Qc = alpha = a = bg_phase = chi2 = np.nan
    sc = w = None
    converged = False
    it = -1
    for it in range(max_iter):
        band = edge & (np.abs(f - f_dip) >= exclude_dip) if it == 0 else edge
        m, b = np.polyfit(f[band], np.unwrap(np.angle(s21 / shape))[band], 1)
        sc = s21 * np.exp(-1j * (m * f + b))
        w = np.abs(f - f[np.argmin(np.abs(sc))]) < half_window
        try:
            popt, chi2, _ = _refined_fit(f[w], sc[w], 0)
            a, bg_phase, fr, Qr, Qc, alpha = popt
        except NotchFitError as e:
            warnings.warn(f"notch fit failed on delay iteration {it}, returning NaNs ({e})", RuntimeWarning, stacklevel=2)
            fr = Qr = Qc = alpha = a = bg_phase = chi2 = np.nan
        if np.isnan(fr):
            break
        if abs(-m / (2 * np.pi) - tau) < tol:
            converged = True
            break
        shape, tau = notch_model(f, fr, Qr, Qc, alpha), -m / (2 * np.pi)
    return TraceFit(fr=fr, Qr=Qr, Qc_abs=Qc, alpha=alpha, tau=-m / (2 * np.pi), s21_corrected=sc, window=w,
                    a=a, bg_phase=bg_phase, chi2=chi2, n_iter=it + 1, converged=converged)


__all__ = ["notch_model", "phase_model", "calculate_tau", "initial_fit", "fit_phase", "refined_fit", "residues",
           "NotchFitError", "NotchFit", "fit_resonance", "NotchStages", "fit_resonance_stages", "TraceFit",
           "fit_resonance_trace"]
