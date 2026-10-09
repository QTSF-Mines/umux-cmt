"""Finding and fitting every resonance of a multiplexer comb.

A wide VNA sweep of a uMux chip shows one notch per channel. The dips are
located in ``|S21|`` (dB) by their prominence, pairs closer than the minimum
channel spacing are resolved in favour of the deeper dip, and each resonance
is then fitted on its own in a window of fixed width with
:func:`umux_cmt.fitting.notch.refined_fit`, using one cable delay for the
whole sweep. This is the ``find_all_peaks`` and ``fit_all_resonators`` survey
of the original code, behind the full-chip distributions of Fig. 4(a)-(b) in
Sec. IV B of the paper; the single-channel parameters used for the
simulations (Table I) come from
:func:`umux_cmt.fitting.notch.fit_resonance_trace` on narrow sweeps instead.

How the paper's Fig. 4(a)-(b) were made: the July 2026 sweep of the whole
chip (an earlier cooldown; not among the data shipped here) was first
corrected by a line fitted to the unwrapped phase of each trace over its
whole span, slope *and* intercept, and the resonances were then fitted with
no further delay and 4 MHz windows::

    m, b = np.polyfit(f, np.unwrap(np.angle(s21)), 1)     # phase line of the whole trace
    tau = -m / (2 * np.pi)
    s21_line = s21 * np.exp(1j * (2 * np.pi * tau * f - b))
    comb = fit_all_resonances(f, s21_line, 0.0, width=4e6)

(the original corrected all traces with one two-dimensional ``polyfit``).
The default window here, 3 MHz, with a delay from
:func:`~umux_cmt.fitting.notch.calculate_tau` (a slope only), is what the
examples use on the shipped 09/26 comb sweep; those settings do not
reproduce Fig. 4(a)-(b).

All numerics are those of the original code; only plotting and printing
were removed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .notch import refined_fit


def find_resonances(f, s21, dist=1.0e6, prominence=10.0):
    """Indices of the resonance dips in a wide sweep.

    ``|S21|`` in dB is flipped so that dips become peaks and
    :func:`scipy.signal.find_peaks` is run with the given prominence in dB
    (and a minimum separation of 10 samples). Of two peaks closer than
    ``dist`` the shallower one is dropped. This is ``find_all_peaks`` of the
    original code.

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    s21 : ndarray of complex
        Transmission.
    dist : float, optional
        Minimum separation of two resonances in Hz.
    prominence : float, optional
        Minimum depth of a dip in dB.

    Returns
    -------
    ndarray of int
        Sample indices of the dips, in increasing frequency.
    """
    # 1. dB and flipped so that dips become peaks
    s21_db = 20 * np.log10(np.abs(s21))
    s21_db_flipped = s21_db.max() - s21_db

    # 2. Peaks above the prominence threshold
    peaks_full, _ = find_peaks(s21_db_flipped, prominence=prominence, distance=10)

    collided_peak_indices = set()

    # 3. Of two peaks closer than dist keep the deeper one
    for i in range(len(peaks_full) - 1):
        peak1_idx = peaks_full[i]
        peak2_idx = peaks_full[i+1]

        if np.abs(f[peak2_idx] - f[peak1_idx]) < dist:
            if s21_db_flipped[peak1_idx] > s21_db_flipped[peak2_idx]:
                collided_peak_indices.add(peak2_idx)
            else:
                collided_peak_indices.add(peak1_idx)

    # 4. Drop the shallower of each collided pair
    final_peaks_list = [p for p in peaks_full if p not in collided_peak_indices]
    peaks = np.array(final_peaks_list, dtype=int)

    return peaks


@dataclass
class CombFit:
    """Result of :func:`fit_all_resonances`: one entry per detected resonance.

    The arrays are the nine lists the original ``fit_all_resonators``
    returned (``Qi_abs`` is its ``Qis``); failed fits are NaN in every array
    except :attr:`peaks`. The internal Q of the original survey, ``Qi_abs``,
    uses ``|Qc|``; the DCM internal Q, which :attr:`NotchFit.Qi
    <umux_cmt.fitting.notch.NotchFit.Qi>` and :attr:`TraceFit.Qi
    <umux_cmt.fitting.notch.TraceFit.Qi>` report and Fig. 4 shows, is
    :attr:`Qi_dcm`.
    """

    peaks: np.ndarray
    """Sample index of each dip in the sweep."""
    a: np.ndarray
    """Baseline amplitude."""
    tau: np.ndarray
    """Cable delay removed before each fit, in s: the ``tau`` passed in, NaN where the fit failed."""
    fr: np.ndarray
    """Resonance frequencies in Hz."""
    Qr: np.ndarray
    """Loaded quality factors."""
    Qc_abs: np.ndarray
    """``|Qc|``."""
    Qi_abs: np.ndarray
    """``1 / (1/Qr - 1/|Qc|)``, the internal Q of the original survey (not DCM; see :attr:`Qi_dcm`)."""
    bw: np.ndarray
    """Bandwidths ``fr / Qr`` in Hz."""
    alpha: np.ndarray
    """Asymmetry angles in rad."""
    chi2: np.ndarray
    """Reduced chi-squared of each fit."""

    def __len__(self):
        return len(self.peaks)

    @property
    def ok(self):
        """Mask of the fits that succeeded."""
        return ~np.isnan(self.fr)

    @property
    def Qc_dcm(self):
        """Diameter-correction-method coupling Q, ``|Qc| / cos(alpha)``."""
        return self.Qc_abs / np.cos(self.alpha)

    @property
    def Qi_dcm(self):
        """Internal quality factor (DCM), ``1 / (1/Qr - 1/Qc_dcm)``."""
        return 1 / (1 / self.Qr - 1 / self.Qc_dcm)

    @property
    def spacing(self):
        """Spacing of neighbouring resonances in Hz (sorted by frequency, failed fits dropped)."""
        return np.diff(np.sort(self.fr[self.ok]))

    def summary(self):
        """A short table (DCM ``Qc`` and ``Qi``), as a string."""
        lines = [f"{'#':>3} {'fr (MHz)':>10} {'Qr':>7} {'|Qc|':>7} {'alpha':>7} {'Qc DCM':>7} {'Qi DCM':>9} "
                 f"{'BW (kHz)':>9} {'chi2':>7}"]
        Qc_dcm, Qi_dcm = self.Qc_dcm, self.Qi_dcm
        for i in range(len(self)):
            lines.append(f"{i:3d} {self.fr[i] / 1e6:10.3f} {self.Qr[i]:7.0f} {self.Qc_abs[i]:7.0f} {self.alpha[i]:7.3f} "
                         f"{Qc_dcm[i]:7.0f} {Qi_dcm[i]:9.0f} {self.bw[i] / 1e3:9.0f} {self.chi2[i]:7.1f}")
        return "\n".join(lines)


def fit_all_resonances(f, s21, tau, dist=1.0e6, width=3e6, prominence=10.0):
    """Find and fit every resonance of a wide sweep (``fit_all_resonators`` of the original code).

    Each dip from :func:`find_resonances` is fitted with
    :func:`~umux_cmt.fitting.notch.refined_fit` on the points within
    ``width / 2`` of it, after removing the delay ``tau``. See the module
    docstring for the settings behind the paper's Fig. 4(a)-(b).

    Parameters
    ----------
    f : ndarray
        Frequencies in Hz.
    s21 : ndarray of complex
        Transmission.
    tau : float
        Cable delay of the sweep in s, e.g. from
        :func:`umux_cmt.fitting.notch.calculate_tau`, or ``0`` for data whose
        phase line has already been removed.
    dist : float, optional
        Minimum separation of two resonances in Hz.
    width : float, optional
        Width of the fit window centred on each dip, in Hz.
    prominence : float, optional
        Minimum depth of a dip in dB.

    Returns
    -------
    CombFit
        Fits that return NaNs (see :func:`~umux_cmt.fitting.notch.refined_fit`)
        are NaN in every array.

    Raises
    ------
    ValueError
        Where :func:`~umux_cmt.fitting.notch.refined_fit` raises for a
        window, as the original did.
    """
    peaks = find_resonances(f, s21, dist=dist, prominence=prominence)

    aas, taus, frs, Qrs, Qcs, Qis, alphas, chi2s, bws = [], [], [], [], [], [], [], [], []

    for peak_idx in peaks:
        center_f = f[peak_idx]
        f_start = center_f - width / 2
        f_stop = center_f + width / 2

        mask = (f >= f_start) & (f <= f_stop)
        f_region = f[mask]
        s21_region = s21[mask]

        a, tau_fit, fr, Qr, Qc, alpha, chi2 = refined_fit(f_region, s21_region, tau)

        bw = fr / Qr
        Qi = 1 / (1/Qr - 1/Qc)

        aas.append(a)
        taus.append(tau_fit)
        frs.append(fr)
        Qrs.append(Qr)
        Qcs.append(Qc)
        alphas.append(alpha)
        chi2s.append(chi2)
        bws.append(bw)
        Qis.append(Qi)

    return CombFit(peaks=peaks, a=np.array(aas, float), tau=np.array(taus, float), fr=np.array(frs, float),
                   Qr=np.array(Qrs, float), Qc_abs=np.array(Qcs, float), Qi_abs=np.array(Qis, float),
                   bw=np.array(bws, float), alpha=np.array(alphas, float), chi2=np.array(chi2s, float))


__all__ = ["find_resonances", "fit_all_resonances", "CombFit"]
