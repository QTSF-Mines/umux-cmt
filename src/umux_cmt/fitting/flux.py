"""The flux response ``fr(I)`` of a channel and the rf-SQUID parameters it yields.

A current ``I`` in a line coupled to the SQUID by the mutual ``Min`` (the
flux-ramp line, ``Mfr``, in the paper) applies the external flux phase
``phi_e = 2 pi (I + I0) Min / Phi0``. The SQUID screens it: the total flux
phase ``phi`` solves Kepler's equation ``phi + lamb sin(phi) = phi_e``
(:func:`phi_of_phie`, Eq. (8) of the paper in the low-power limit), and the
resonance frequency follows Eq. (2) of the paper,

    fr(I) = f0 + (4 f0**2 / Z1) (Mc**2 / Ls) lamb cos(phi) / (1 + lamb cos(phi)).

:func:`fit_lambda` fits this to the measured ``fr`` at each bias point with a
global optimiser (differential evolution on an L1 objective, bounded by
:func:`guess_params`), and :func:`fit_flux_response` runs it from several
seeds and polishes the best with least squares to get one-sigma errors, the
procedure behind Table I of the paper (Sec. IV B). ``Ls`` is not
identifiable from the response (it only appears as ``Mc**2 / Ls``) and is
fixed.

Two flux-offset conventions appear here and mixing them up shifts every flux
axis by ``~0.3 Phi0``:

* ``I0``, *absolute*: what :func:`fit_lambda` returns and :func:`fr_of_I`
  expects. At ``I = 0`` the SQUID already sees the flux ``I0 Min``.
* ``I_trapped``, *relative*: the offset measured from the ``phi = pi/2``
  operating point, where ``cos(phi) = 0``, the SQUID does not load the
  resonator and ``fr = f0``. That point sits at ``phi_e = pi/2 + lamb``
  (:func:`I_bare`), and ``I_trapped = I0 - I_bare`` wrapped into
  ``+/- Phi0/2`` (:func:`relative_I_trapped`). This is the
  ``trapped_current`` of the configuration files, which the loader applies
  on the ``TrappedFlux`` line on top of the ``phi_e = pi/2 + lamb`` offset,
  and ``phi_tr`` of Table I is ``2 pi I_trapped Mfr / Phi0``
  (:func:`trapped_flux_phase`).

Where this comes from in the original code: the model, the offset
conventions, the guesses and :func:`fit_lambda` are ``lambdafit.py`` of
``resonator_fitting``; :func:`fit_flux_response` and :func:`summarize_channel`
are ``fit_flux`` and ``table_params`` of the paper's parameter notebook. The
original ``resonator_fitting/include/flux_response.py`` held something else
and is not ported: ``refined_flux_fit`` and ``fit_all_flux_responses``, the
full-chip survey behind Fig. 4(c)-(g). That survey ran a single *unseeded*
:func:`fit_lambda` per channel (with its default ``Ls = 24.2 pH``, on bias
currents converted with ``Phi0 = 2.0678e-15`` and a nominal 15 pH), returned
the *relative* trapped current and turned any exception into NaNs; an
unseeded global fit cannot be reproduced bit for bit. To survey a chip, loop
:func:`fit_flux_response` over the channels: it is seeded, and its
:attr:`FluxFit.trapped_current` is the same relative current.

All numerics are those of the code the paper was produced with.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy
import scipy.optimize as op

from ..constants import Phi0

Z1_DEFAULT = 50.0
"""Characteristic impedance of the resonator in Ohm."""
LS_DESIGN = 22.4e-12
"""SQUID loop inductance from the FastHenry design simulation, in H; used by :func:`guess_params`."""
LS_FIT = 24.2e-12
"""SQUID loop inductance the paper's fits and configuration files assume, in H."""


###############################################################################
#                                 THE MODEL                                   #
###############################################################################

def phi_of_phie(phie, lamb):
    """Total flux phase ``phi`` for the external flux phase ``phie``.

    Solves ``phi + lamb sin(phi) = phie`` by Newton-Raphson on the branch
    wrapped into ``[-pi, pi]``, where the iteration is stable for ``lamb < 1``,
    and unwraps back. At most 15 iterations, stopping once the step is below
    ``1e-10``.

    Parameters
    ----------
    phie : float or ndarray
        External flux phase in rad.
    lamb : float
        Hysteresis parameter ``Ls / LJ``.

    Returns
    -------
    float or ndarray
        Total flux phase in rad.
    """
    if lamb < 1e-6:
        return phie

    # Wrap to [-pi, pi]: Newton-Raphson is unconditionally stable there
    phie_wrapped = (phie + np.pi) % (2 * np.pi) - np.pi

    phi_wrapped = np.copy(phie_wrapped)

    for _ in range(15):
        step = (phi_wrapped + lamb * np.sin(phi_wrapped) - phie_wrapped) / (1.0 + lamb * np.cos(phi_wrapped))
        if np.max(np.abs(step)) < 1e-10:
            break
        phi_wrapped -= step

    # Unwrap back to the absolute domain
    phi = phi_wrapped + (phie - phie_wrapped)

    return phi


def fr_of_I(I, I0, Min, Mc, f0, lamb, Ls=LS_FIT, Z1=Z1_DEFAULT):
    """Resonance frequency against the current in a flux line, Eq. (2) of the paper.

    Parameters
    ----------
    I : float or ndarray
        Line current in A.
    I0 : float
        *Absolute* offset current in A (see the module docstring; convert a
        ``trapped_current`` with :func:`absolute_I0` first).
    Min : float
        Mutual inductance of the line to the SQUID in H.
    Mc : float
        Mutual inductance between the SQUID and the resonator in H.
    f0 : float
        Bare resonance frequency in Hz (SQUID not loading the resonator).
    lamb : float
        Hysteresis parameter.
    Ls : float, optional
        SQUID loop inductance in H.
    Z1 : float, optional
        Resonator characteristic impedance in Ohm.

    Returns
    -------
    float or ndarray
        ``fr`` in Hz.
    """
    phie = 2 * np.pi * (I + I0) * Min / Phi0
    phi = phi_of_phie(phie, lamb)
    fr = f0 + (4 * (f0**2) / Z1) * (Mc**2) / Ls * lamb * np.cos(phi) / (1 + lamb * np.cos(phi))

    return fr


###############################################################################
#                           FLUX-OFFSET CONVERSIONS                           #
###############################################################################

def I_bare(Min, lamb):
    """Line current that puts the SQUID at total flux phase ``phi = pi/2``.

    There ``cos(phi) = 0``, the SQUID stops loading the resonator and
    :func:`fr_of_I` returns exactly ``f0``. By Kepler's equation that is the
    external phase ``phie = pi/2 + lamb``.

    Parameters
    ----------
    Min : float
        Mutual inductance of the line in H.
    lamb : float
        Hysteresis parameter.

    Returns
    -------
    float
        Current in A.
    """
    return Phi0 * ((np.pi / 2.0) + lamb) / (2 * np.pi * Min)


def absolute_I0(I_trapped, Min, lamb):
    """Absolute offset current ``I0``, what :func:`fr_of_I` takes, from a relative trapped current.

    The inverse of :func:`relative_I_trapped` up to a whole number of flux
    quanta: ``I0 = I_trapped + I_bare(Min, lamb)``.

    Parameters
    ----------
    I_trapped : float or ndarray
        Trapped current relative to the ``phi = pi/2`` point (the
        ``trapped_current`` of a configuration file), in A.
    Min : float
        Mutual inductance of the line in H.
    lamb : float
        Hysteresis parameter.

    Returns
    -------
    float or ndarray
        Absolute offset current in A.
    """
    return I_trapped + I_bare(Min, lamb)


def relative_I_trapped(I0, Min, lamb):
    """Trapped current relative to the ``phi = pi/2`` point, wrapped into ``+/- Phi0/2``.

    This is the ``trapped_current`` of the configuration files.

    Parameters
    ----------
    I0 : float
        Absolute offset current from :func:`fit_lambda`, in A.
    Min : float
        Mutual inductance of the line in H.
    lamb : float
        Hysteresis parameter.

    Returns
    -------
    float
        Current in A, in ``[-Phi0/(2 Min), Phi0/(2 Min))``.
    """
    I_tr = I0 - I_bare(Min, lamb)
    I_Phi0 = Phi0 / Min
    return (I_tr + I_Phi0 / 2) % I_Phi0 - I_Phi0 / 2


def trapped_flux_phase(I0, Mfr, lamb):
    """Trapped flux phase ``phi_tr = 2 pi I_trapped Mfr / Phi0``, the convention of Table I.

    ``I_trapped`` is :func:`relative_I_trapped`, so ``phi_tr`` is measured
    from the ``phi = pi/2`` point and lies in ``[-pi, pi)``.

    Parameters
    ----------
    I0 : float
        Absolute offset current from :func:`fit_lambda`, in A.
    Mfr : float
        Mutual inductance of the flux-ramp line in H.
    lamb : float
        Hysteresis parameter.

    Returns
    -------
    float
        Phase in rad.
    """
    return 2 * np.pi * relative_I_trapped(I0, Mfr, lamb) * Mfr / Phi0


###############################################################################
#                              INITIAL GUESSES                                #
###############################################################################

def short_signal_periodicity(x):
    """Period of a signal that contains between one and two periods.

    A window is shrunk from the full length down to half of it; the window
    whose Fourier spectrum has the least power above the third harmonic is
    taken to hold a whole number of periods (the SQUID response has power in
    the first harmonics, so those are allowed). The phase of the fundamental
    in that window gives the offset of the oscillation.

    Parameters
    ----------
    x : ndarray
        The signal.

    Returns
    -------
    ppp : int
        Points per period.
    poff : float
        Offset, in points, to the zero phase of the fundamental.
    """
    hhp = np.zeros(len(x)//2)
    phases = np.zeros(len(x)//2)

    for n in range(len(hhp)):
        m = len(x) - n                              # shrink the window by n points
        X = np.fft.rfft(x[:m]) / m                  # Fourier spectrum
        hhp[n] = np.sum(np.abs(X[4:])**2)           # power in the higher bins
        phases[n] = np.angle(X[1])                  # phase of the fundamental

    ppp = len(x) - np.argmin(hhp)                   # window with least power in the higher bins
    poff = ppp*phases[np.argmin(hhp)]/(2*np.pi)

    return ppp, poff


def guess_params(I, fr, Ls=LS_DESIGN, Z1=Z1_DEFAULT):
    """Starting values for :func:`fit_lambda` from the data alone.

    ``lamb`` is the design target 0.33, ``f0`` the mean ``fr``, ``Mc`` follows
    from the peak-to-peak swing of ``fr`` at that ``lamb``, and the period and
    phase of the response (:func:`short_signal_periodicity`) give ``Min`` and
    ``I0``. The data are assumed to span between one and two flux quanta;
    outliers from failed resonance fits can throw the guesses off.

    Parameters
    ----------
    I : ndarray
        Line currents in A.
    fr : ndarray
        Resonance frequencies in Hz.
    Ls : float, optional
        Loop inductance used for the ``Mc`` guess. The default is the
        FastHenry design value, which the original code used here regardless
        of the ``Ls`` of the fit; it only sets the search box of the global
        optimiser.
    Z1 : float, optional
        Resonator characteristic impedance in Ohm.

    Returns
    -------
    (I0, Min, Mc, f0, lamb) : tuple of float
    """
    lambguess = 0.33        # typical design target
    f0guess = np.mean(fr)
    Mcguess = np.sqrt( (np.max(fr)-np.min(fr))*Z1*Ls*(1-lambguess**2) / (8*lambguess*f0guess**2) )

    ppp, poff = short_signal_periodicity(fr)     # assumes between 1 and 2 Phi0 of data
    Minguess = Phi0 / (ppp*(I[1] - I[0]))
    I0guess = I[0] + poff*(I[1]-I[0])

    return I0guess, Minguess, Mcguess, f0guess, lambguess


###############################################################################
#                                  THE FITS                                   #
###############################################################################

def _global_state_advanced(before):
    """Whether numpy's global ``RandomState`` moved on since ``before`` (a ``get_state()``)."""
    after = np.random.get_state()
    return not (before[2] == after[2] and np.array_equal(before[1], after[1]))


def fit_lambda(I, fr, Ls=2.42e-11, Z1=50.0, Mc_max=10e-12, seed=None):
    """Fit :func:`fr_of_I` to a measured flux response with differential evolution.

    The objective is the L1 norm of the residuals (robust to the occasional
    failed resonance fit), minimised over a box built around
    :func:`guess_params`: ``I0`` within 0.2 mA of the guess, ``Min`` within a
    decade, ``Mc`` from a tenth of the guess to ``Mc_max``, ``f0`` within
    10 MHz and ``0 < lamb < 1``.

    Two failure modes to know about, because neither raises:

    * The model has no RF dressing. Driven hard, the real modulation depth
      shrinks; the fit can only follow by shrinking ``lamb`` and compensating
      with ``Mc`` (the depth goes as ``Mc**2 lamb``), until ``Mc`` rails at
      ``Mc_max`` and ``lamb`` collapses. Raising ``Mc_max`` only moves the wall.
    * :func:`guess_params` assumes between one and two flux periods in the
      data; many more periods over the current axis break the period guess
      and everything built on it.

    Parameters
    ----------
    I : ndarray
        Line currents in A.
    fr : ndarray
        Resonance frequencies in Hz.
    Ls : float, optional
        SQUID loop inductance in H (fixed in the fit).
    Z1 : float, optional
        Resonator characteristic impedance in Ohm.
    Mc_max : float, optional
        Upper bound on ``Mc`` in H.
    seed : int, optional
        Seed for the global optimiser. When given, ``numpy.random.seed(seed)``
        is called right before :func:`scipy.optimize.differential_evolution`,
        which, called without its own ``rng``, draws from numpy's global
        ``RandomState``. This is the mechanism the paper's analysis used, and
        it is kept rather than passing ``rng=`` because that would select a
        different generator and change the result bit for bit. It relies on
        SciPy's transitional behaviour (SPEC 7): if a SciPy version stops
        drawing from the global state when ``rng`` is not given, the seed
        stops controlling the fit, and a ``RuntimeWarning`` says so.

    Returns
    -------
    (I0, Min, Mc, f0, lamb) : tuple of float
        ``I0`` is the *absolute* offset current, see the module docstring.

    Warns
    -----
    RuntimeWarning
        If ``seed`` is given but the optimiser did not draw from numpy's
        global ``RandomState`` (the result is then not reproducible).
    """
    # 1. Guesses from the data (the guess uses the design Ls; see guess_params)
    I0_g, Min_g, Mc_g, f0_g, lamb_g = guess_params(I, fr)

    def fr_wrapper(I_data, I0_fit, Min_fit, Mc_fit, f0_fit, lamb_fit):
        return fr_of_I(I_data, I0_fit, Min_fit, Mc_fit, f0_fit, lamb_fit, Ls, Z1)

    # L1 objective for outlier resistance
    def objective(params):
        I0_fit, Min_fit, Mc_fit, f0_fit, lamb_fit = params
        f_model = fr_wrapper(I, I0_fit, Min_fit, Mc_fit, f0_fit, lamb_fit)
        return np.sum(np.abs(fr - f_model))

    # 2. A tight search box around the guesses
    bounds = [
        (I0_g - .2e-3, I0_g + .2e-3),        # I0: +/- 0.2 mA around the guessed offset
        (Min_g * 0.1, Min_g * 10.0),         # Min: a decade either side of the guess
        (Mc_g * 0.1, Mc_max),                # Mc: fits rail here at high drive
        (f0_g - 10e6, f0_g + 10e6),          # f0: +/- 10 MHz around the guess
        (0, 1)                               # lamb: one full period
    ]

    # 3. Global optimiser, drawing from numpy's global RandomState
    if seed is not None:
        np.random.seed(seed)
        state = np.random.get_state()        # read only; for the reproducibility check below
    result = op.differential_evolution(objective, bounds, strategy='best1bin', popsize=15)
    if seed is not None and not _global_state_advanced(state):
        warnings.warn(f"differential_evolution did not draw from numpy's global RandomState (scipy "
                      f"{scipy.__version__}): seed={seed} no longer controls the fit, so it is not reproducible "
                      f"and will not match the paper's values (numpy 2.5.0, scipy 1.18.0)", RuntimeWarning,
                      stacklevel=2)

    I0fit, Minfit, Mcfit, f0fit, lambfit = result.x

    return I0fit, Minfit, Mcfit, f0fit, lambfit


@dataclass
class FluxFit:
    """Result of :func:`fit_flux_response`.

    The parameters are in the order ``(I0, Mfr, Mc, f0, lamb)`` of
    :func:`fr_of_I`, available together as :attr:`params`.
    """

    I0: float
    """Absolute offset current in A."""
    Mfr: float
    """Mutual inductance of the swept line in H."""
    Mc: float
    """SQUID-resonator mutual inductance in H."""
    f0: float
    """Bare resonance frequency in Hz."""
    lamb: float
    """Hysteresis parameter."""
    errors: np.ndarray = field(repr=False)
    """One-sigma errors of ``(I0, Mfr, Mc, f0, lamb)`` from the least-squares polish."""
    tries: np.ndarray = field(repr=False)
    """The differential-evolution solutions of every seed, shape ``(n_seeds, 5)``."""
    rms: float
    """Standard deviation of the residuals ``fr - model`` in Hz."""
    Ls: float = LS_FIT
    """Loop inductance assumed in the fit, in H."""
    Z1: float = Z1_DEFAULT
    """Characteristic impedance assumed in the fit, in Ohm."""
    I: np.ndarray = field(default=None, repr=False)
    """The fitted currents in A."""
    fr: np.ndarray = field(default=None, repr=False)
    """The fitted resonance frequencies in Hz."""

    def __repr__(self):
        return (f"FluxFit(f0={self.f0 / 1e6:.4f} MHz, lamb={self.lamb:.4f}, Mc={self.Mc * 1e12:.3f} pH, "
                f"Mfr={self.Mfr * 1e12:.3f} pH, I_trapped={self.trapped_current * 1e6:.2f} uA, rms={self.rms / 1e3:.1f} kHz)")

    @property
    def params(self):
        """``array([I0, Mfr, Mc, f0, lamb])``."""
        return np.array([self.I0, self.Mfr, self.Mc, self.f0, self.lamb])

    @property
    def LJ(self):
        """Josephson inductance ``Ls / lamb`` in H."""
        return self.Ls / self.lamb

    @property
    def trapped_current(self):
        """Trapped current relative to the ``phi = pi/2`` point (``trapped_current`` of the configs), in A."""
        return relative_I_trapped(self.I0, self.Mfr, self.lamb)

    @property
    def phi_tr(self):
        """Trapped flux phase ``2 pi I_trapped Mfr / Phi0`` in rad."""
        return trapped_flux_phase(self.I0, self.Mfr, self.lamb)

    @property
    def residuals(self):
        """``fr - model(I)`` in Hz."""
        return self.fr - self.model(self.I)

    def model(self, I):
        """The fitted ``fr(I)`` in Hz.

        Parameters
        ----------
        I : float or ndarray
            Line current in A.

        Returns
        -------
        float or ndarray
        """
        return fr_of_I(I, self.I0, self.Mfr, self.Mc, self.f0, self.lamb, Ls=self.Ls, Z1=self.Z1)

    def summary(self):
        """A few lines with the parameters and their errors, as a string."""
        e = self.errors
        return (f"f0   {self.f0 / 1e6:.4f} MHz ({e[3] / 1e3:.1f} kHz)\n"
                f"lamb {self.lamb:.4f} ({e[4]:.4f})\n"
                f"Mc   {self.Mc * 1e12:.3f} pH ({e[2] * 1e12:.3f})\n"
                f"Mfr  {self.Mfr * 1e12:.3f} pH ({e[1] * 1e12:.3f})\n"
                f"I_tr {self.trapped_current * 1e6:.2f} uA   phi_tr {self.phi_tr:.3f} rad\n"
                f"rms  {self.rms / 1e3:.1f} kHz")


def fit_flux_response(I, fr, Ls=LS_FIT, Z1=Z1_DEFAULT, seeds=range(5), Mc_max=10e-12):
    """Flux-response fit of Table I: best of several seeded global fits, polished with least squares.

    :func:`fit_lambda` is run once per seed; the solution with the smallest
    sum of squared residuals starts a least-squares fit in scaled parameters
    (``x_scale='jac'``), whose covariance gives the one-sigma errors. This is
    ``fit_flux`` of the paper's parameter notebook.

    Parameters
    ----------
    I : ndarray
        Line currents in A.
    fr : ndarray
        Resonance frequencies in Hz, one per current.
    Ls : float, optional
        SQUID loop inductance in H, fixed.
    Z1 : float, optional
        Resonator characteristic impedance in Ohm.
    seeds : iterable of int, optional
        Seeds of the global fits, see :func:`fit_lambda`.
    Mc_max : float, optional
        Upper bound on ``Mc`` in the global fits, in H.

    Returns
    -------
    FluxFit

    Notes
    -----
    ``I`` and ``fr`` enter the fit as given, without a conversion of their
    dtype, as in the original. The paper's values come from float64 arrays
    (what the VNA files hold); a float32 ``fr``, for example, changes the
    guesses and with them the result.
    """
    def fr_model(q, I):
        return fr_of_I(I, *q, Ls=Ls, Z1=Z1)

    tries = []
    for seed in seeds:
        tries.append(fit_lambda(I, fr, Ls=Ls, Z1=Z1, Mc_max=Mc_max, seed=seed))
    q0 = min(tries, key=lambda q: np.sum((fr - fr_model(q, I))**2))
    scale = np.array([1e-6, 1e-12, 1e-12, 1e6, 1.0])
    r = op.least_squares(lambda x: (fr_model(x * scale, I) - fr) / 1e3, q0 / scale, x_scale='jac')
    cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun**2) / (len(fr) - 5)
    q, err = r.x * scale, np.sqrt(np.diag(cov)) * scale
    rms = np.std(fr - fr_model(q, I))
    return FluxFit(I0=q[0], Mfr=q[1], Mc=q[2], f0=q[3], lamb=q[4], errors=err, tries=np.array(tries, float), rms=rms,
                   Ls=Ls, Z1=Z1, I=np.asarray(I), fr=np.asarray(fr))


def summarize_channel(Qr_arr, Qc_abs_arr, alpha_arr, flux_fit):
    """The parameter set of one channel for Table I and the configuration file.

    The resonator parameters are the medians over the flux points of the
    notch fits; ``Qc`` is converted to the DCM value ``median(|Qc|) / cos(median(alpha))``
    and ``Qi`` follows from it. The SQUID parameters come from the
    flux-response fit. This is ``table_params`` of the paper's parameter
    notebook, plus ``Qc_abs`` and ``trapped_current``.

    Parameters
    ----------
    Qr_arr, Qc_abs_arr, alpha_arr : ndarray
        ``Qr``, ``|Qc|`` and ``alpha`` of the notch fit at every flux point.
    flux_fit : FluxFit or tuple
        The flux-response fit, or ``((I0, Mfr, Mc, f0, lamb), errors)``.

    Returns
    -------
    dict
        ``Qr``, ``Qc`` (DCM), ``Qi``, ``alpha``, ``f0``, ``lamb``, ``Mc``,
        ``Mfr``, ``phi_tr`` and ``err`` (the flux-fit errors), plus the raw
        ``Qc_abs`` and the ``trapped_current`` for the configuration file.
    """
    alpha = np.median(alpha_arr)
    Qr = np.median(Qr_arr)
    Qc_abs = np.median(Qc_abs_arr)
    Qc = Qc_abs / np.cos(alpha)     # DCM
    if isinstance(flux_fit, FluxFit):
        (I0, Mfr, Mc, f0, lamb), err = flux_fit.params, flux_fit.errors
    else:
        (I0, Mfr, Mc, f0, lamb), err = flux_fit
    return dict(Qr=Qr, Qc=Qc, Qi=1 / (1 / Qr - 1 / Qc), alpha=alpha, f0=f0, lamb=lamb, Mc=Mc, Mfr=Mfr, err=err,
                phi_tr=2 * np.pi * relative_I_trapped(I0, Mfr, lamb) * Mfr / Phi0,
                Qc_abs=Qc_abs, trapped_current=relative_I_trapped(I0, Mfr, lamb))


__all__ = ["phi_of_phie", "fr_of_I", "I_bare", "absolute_I0", "relative_I_trapped", "trapped_flux_phase",
           "short_signal_periodicity", "guess_params", "fit_lambda", "FluxFit", "fit_flux_response",
           "summarize_channel", "LS_FIT", "LS_DESIGN", "Z1_DEFAULT"]
