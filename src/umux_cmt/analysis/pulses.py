"""Folding, edge alignment and calibration of square-pulse records (Sec. V C of the paper).

In the pulse measurement a square wave (``PERIOD`` long, duty 1/2) plays on
the flux line at a nominal *offset* and *amplitude*, and the transmission
``z(t) = I + iQ`` of one probe tone is digitised, untriggered, for thousands
of pulses. The reduction to one averaged period per record is:

* **Period.** The true period differs slightly from ``PERIOD * FS`` samples,
  which would smear the edges over thousands of pulses. It is measured
  (:func:`estimate_period`) from the phase drift of the first ``K`` harmonics
  of the record between its first and last quarter.
* **Folding** (:func:`fold_pulses`): all pulses are averaged into ``nbin``
  bins of one period.
* **Alignment** (:func:`align_to_rising_edge`): the fold is rotated to put
  the rising edge at bin 0. A coarse edge is the best match to a ``+-1``
  square wave; it lands late, because the ring-up drags the match, so each
  edge is placed at its *departure* (:func:`departure_bin`): the last bin
  before the trace leaves the preceding plateau by more than 6 sigma of the
  plateau noise or 0.5 % of the step. The two edges are exactly half a
  period apart, so the earlier detection is kept for both, and of the two
  edges the rising one is the edge after which the trace moves *away* from
  the static (amplitude 0) level of that offset.
* **Plateaus** (:func:`plateau_levels`): the mean over the last ``n_plat``
  bins of each half period is the static response to that flux level; the
  scatter about the plateau means is the noise per bin
  (:func:`plateau_noise`).
* **Flux of each level, not fitted.** The source output recorded directly on
  the digitiser (the bare-source records) gives volts per nominal amplitude
  and per nominal offset, and the baseline (:func:`source_levels`,
  :func:`source_scales`). Volts become flux through the flux-line current
  and the mutual inductance of the configuration,
  ``Phi / Phi0 = v * Mfr / (Phi0 * (R_loop + R_src))``
  (:class:`LevelCalibration`), so that a record at nominal offset ``o`` and
  amplitude ``a`` steps between ``ko*o + b`` and ``ko*o + b + ka*a``.

The comparison with the CMT then fits nothing but the readout:

* **Calibration of the IQ plane** on the plateaus, where the field is in its
  steady state: ``z = A * S21(Phi) + B`` with the static model ``S21`` at the
  flux of each level (:func:`calibration_levels`, :func:`fit_levels`). Each
  record gives its on level; the off plateaus of the records of one offset
  and its static point share one flux and are averaged into one off level.
  The fit is repeated with each offset held out, and the scatter of the off
  levels within an offset (:func:`off_level_scatter`) shows how reproducible
  a level is. Residuals are quoted relative to the diameter of the resonance
  circle in the IQ plane (:func:`iq_circle_diameter`).
* **Edge residuals.** A model trace, mapped into volts with ``A`` and ``B``,
  is averaged over the measured bins (:func:`binned_model`) and compared with
  the fold over the first ``EDGE`` seconds after each edge (:func:`edge_rms`).
  One time shift common to all records (the readout delay and the position of
  bin 0) minimises the summed squared residual (:func:`edge_residual_sum`,
  :func:`fit_edge_shift`).

The module-level constants are the settings of the measurement in the
paper (40 MS/s, 40 us period, 10 us plateau and 3 us edge windows, the grid
of nominal levels, and the records whose amplitude change had not yet taken
effect). :func:`fold_pulse_records` and :func:`source_record_levels` run the
whole reduction over a folder of raw records; the shipped demo files
``pulses_folded.npz`` and ``source_levels.npz`` are their output.
"""
import glob
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import Phi0 as _Phi0
from ..io import load_record

FS = 40e6
"""Sample rate of the digitiser in Hz."""
PERIOD = 40e-6
"""True period of the square wave in s (the records' own metadata is wrong)."""
PLATEAU = 10e-6
"""Length of the plateau window at the end of each half period, in s."""
EDGE = 3e-6
"""Length of the window after each edge in which models are compared with the folds, in s."""
SUB = 5
"""Sub-samples per bin when a model is averaged over the measured bins (:func:`binned_model`)."""
NBIN = int(round(PERIOD * FS))
"""Bins per folded period (25 ns)."""
N_PLAT = int(round(PLATEAU * FS))
"""Bins per plateau window."""
SHIFT_COARSE = (-300e-9, 300e-9, 5e-9)
"""``numpy.arange`` arguments (start, stop, step, in s) of the coarse scan of the common time shift."""
SHIFT_FINE = (-5e-9, 5e-9, 0.25e-9)
"""``numpy.arange`` arguments of the fine scan, added to the best shift of the coarse scan."""

OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0]
"""Nominal offsets of the pulse records, in the source's flux units."""
AMPS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2]
"""Nominal amplitudes of the pulse records; amplitude 0 is the static record of an offset."""
DUPES = {(0.0, 0.75): 0.5, (0.25, 0.25): 0.1, (0.5, 0.75): 0.5, (1.0, 0.05): 0.0}
"""(offset, amplitude) of records taken before the amplitude change had taken effect, mapped to the
amplitude they actually hold. They duplicate another record and are skipped."""

SRC_AMPS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
"""Nominal amplitudes of the bare-source records."""
SOURCE_OFFSET = 0.5
"""Nominal offset of all bare-source records."""
R_LOOP = 1983.0
"""Resistance of the flux line and its series resistors in Ohm (DMM)."""
R_SRC = 50.0
"""Output impedance of the source in Ohm (assumed)."""


# ----------------------------------------------------------------------------- one record

def estimate_period(z, P0=NBIN, K=20):
    """Period of a repetitive record in samples, from the phase drift of its harmonics.

    Harmonic ``k`` of the nominal period ``P0`` turns by ``2 pi k (3N) (1/P - 1/P0)``
    between the first and the last quarter (``N`` samples each) of the
    record. The drifts of the first ``K`` harmonics are combined, weighted by
    their power, into one estimate of the true period ``P``.

    Parameters
    ----------
    z : ndarray of complex
        The record.
    P0 : float, optional
        Nominal period in samples.
    K : int, optional
        Number of harmonics.

    Returns
    -------
    float
        Period in samples.
    """
    N = len(z) // 4
    w1 = np.exp(-2j * np.pi * np.arange(N) / P0)
    w, num, den = np.ones(N), 0.0, 0.0
    for k in range(1, K + 1):
        w = w * w1
        X1, X2 = (z[:N] * w).sum(), (z[-N:] * w).sum()
        num += k * abs(X1) ** 2 * np.angle(X2 / X1)
        den += (k * abs(X1)) ** 2
    return 1 / (1 / P0 + num / den / (2 * np.pi * 3 * N))


def fold_pulses(z, P, nbin=NBIN):
    """Average every period of a record into one, in ``nbin`` bins.

    Sample ``i`` goes to bin ``floor(((i / P) mod 1) * nbin)`` and each bin is
    the mean of its samples. Over thousands of periods the sample phases
    drift through every bin, so a bin is the average over its 1/``nbin`` of
    the period.

    Parameters
    ----------
    z : ndarray of complex
        The record.
    P : float
        Period in samples (:func:`estimate_period`; need not be an integer).
    nbin : int, optional
        Bins per period.

    Returns
    -------
    ndarray of complex
        The folded period, ``nbin`` values; bin 0 starts at the first sample
        of the record.
    """
    k = np.minimum(((np.arange(len(z)) / P) % 1 * nbin).astype(int), nbin - 1)
    return (np.bincount(k, z.real, nbin) + 1j * np.bincount(k, z.imag, nbin)) / np.bincount(k, minlength=nbin)


def departure_bin(zf, k, nbin=NBIN):
    """Bin at which the trace departs from the plateau preceding a coarse edge ``k``.

    The plateau level and its noise are taken 400 to 200 bins before ``k``
    (10 to 5 us before the edge at 25 ns bins), the step from the mean 500 to
    700 bins after it, and bins ``k - 200`` to ``k + 200`` are scanned for
    the first point further than ``max(6 sigma, 0.5 % of the step)`` from the
    plateau. The bin before it is returned. Indices wrap around the period.

    Parameters
    ----------
    zf : ndarray of complex
        Folded period (:func:`fold_pulses`).
    k : int
        Coarse position of the edge, in bins.
    nbin : int, optional
        Bins per period.

    Returns
    -------
    int
        The departure bin, in ``[0, nbin)``.
    """
    base = zf[(k - 400 + np.arange(200)) % nbin]
    p, sig = base.mean(), np.sqrt(np.mean(np.abs(base - base.mean()) ** 2))
    step = abs(zf[(k + 500 + np.arange(200)) % nbin].mean() - p)
    scan = (k - 200 + np.arange(400)) % nbin
    return (scan[np.argmax(np.abs(zf[scan] - p) > max(6 * sig, 0.005 * step))] - 1) % nbin


def align_to_rising_edge(zf, z_static, nbin=NBIN, n_plat=N_PLAT):
    """Rotate a folded period so that its rising edge is at bin 0.

    Parameters
    ----------
    zf : ndarray of complex
        Folded period (:func:`fold_pulses`).
    z_static : complex
        Static IQ level of the same offset (mean of its amplitude-0 record).
        The rising edge is the one after which the trace moves away from it.
    nbin, n_plat : int, optional
        Bins per period and per plateau window.

    Returns
    -------
    ndarray of complex
        The rotated fold: rising edge at bin 0, falling edge at ``nbin // 2``.
    """
    h = nbin // 2
    s = np.cumsum(np.r_[0, zf, zf])
    k = np.argmax(np.abs(2 * (s[h:h + nbin] - s[:nbin]) - zf.sum()))   # coarse edge: best match to a +-1 square wave
    ka, kb = departure_bin(zf, k, nbin), departure_bin(zf, k + h, nbin)
    ka = (ka + min((kb - ka) % nbin - h, 0)) % nbin                     # the duty is exactly 1/2: keep the earlier edge
    za, zb = np.roll(zf, -ka), np.roll(zf, -(ka + h))
    far = lambda z: abs(z[h - n_plat:h].mean() - z_static)               # noqa: E731  the on plateau is the one away from the static level
    return za if far(za) >= far(zb) else zb


def plateau_levels(zf, nbin=NBIN, n_plat=N_PLAT):
    """Mean of the aligned fold over the last ``n_plat`` bins of each half period.

    Parameters
    ----------
    zf : ndarray of complex
        Aligned fold (:func:`align_to_rising_edge`).
    nbin, n_plat : int, optional
        Bins per period and per plateau window.

    Returns
    -------
    (on, off)
        The level with the pulse on (before the falling edge) and off (before
        the rising edge), complex.
    """
    return zf[nbin // 2 - n_plat:nbin // 2].mean(), zf[nbin - n_plat:].mean()


def plateau_noise(zf, nbin=NBIN, n_plat=N_PLAT):
    """Noise per bin of a folded average, from its scatter on the plateaus.

    Parameters
    ----------
    zf : ndarray of complex
        Aligned fold (:func:`align_to_rising_edge`).
    nbin, n_plat : int, optional
        Bins per period and per plateau window.

    Returns
    -------
    float
        rms distance of the plateau bins from their plateau's mean
        (:func:`plateau_levels`), in the units of ``zf``.
    """
    on, off = plateau_levels(zf, nbin, n_plat)
    r = np.r_[zf[nbin // 2 - n_plat:nbin // 2] - on, zf[nbin - n_plat:] - off]
    return np.sqrt(np.mean(np.abs(r) ** 2))


def bin_centres(nbin=NBIN, period=PERIOD):
    """Time of each bin centre of a folded period.

    Parameters
    ----------
    nbin : int, optional
        Bins per period.
    period : float, optional
        Period in s.

    Returns
    -------
    ndarray of float
        ``(i + 0.5) * period / nbin`` for every bin ``i``, in s from the
        rising edge of an aligned fold.
    """
    return (np.arange(nbin) + 0.5) * period / nbin


def plateau_mask(nbin=NBIN, period=PERIOD, plateau=PLATEAU):
    """The bins of the two plateau windows: the last ``plateau`` seconds of each half period.

    Parameters
    ----------
    nbin : int, optional
        Bins per period.
    period, plateau : float, optional
        Period and length of each window, in s.

    Returns
    -------
    ndarray of bool
        ``True`` for the bins whose centre (:func:`bin_centres`) lies in a
        window; the same bins :func:`plateau_levels` averages.
    """
    tb = bin_centres(nbin, period)
    return ((tb >= period / 2 - plateau) & (tb < period / 2)) | (tb >= period - plateau)


def edge_mask(nbin=NBIN, period=PERIOD, edge=EDGE):
    """The bins of the two edge windows: the first ``edge`` seconds after each edge.

    Parameters
    ----------
    nbin : int, optional
        Bins per period.
    period, edge : float, optional
        Period and length of each window, in s.

    Returns
    -------
    ndarray of bool
        ``True`` for the bins whose centre (:func:`bin_centres`) is less than
        ``edge`` after the rising edge (time 0) or the falling edge
        (``period / 2``) of an aligned fold.
    """
    tb = bin_centres(nbin, period)
    return (tb < edge) | ((tb >= period / 2) & (tb < period / 2 + edge))


def binned_model(model, shift, nbin=NBIN, period=PERIOD, sub=SUB):
    """Average a time-domain model over each measured bin.

    The sample phases drift through each bin over a record, so a measured bin
    is the average over a slice of the period; the model is averaged over the
    same slices, ``sub`` sub-samples per bin, after a shift in time.

    Parameters
    ----------
    model : callable
        ``t -> S21`` (or any complex trace), periodic in ``t`` (s) with the
        pulse period, e.g. :func:`umux_cmt.simulate.periodic_interpolant` of a
        simulation with the rising edge at ``t = 0``.
    shift : float
        Delay of the model relative to the fold, in s.
    nbin : int, optional
        Bins per period.
    period : float, optional
        Period in s.
    sub : int, optional
        Sub-samples per bin.

    Returns
    -------
    ndarray of complex
        ``nbin`` values; ``A * binned_model(...) + B`` compares with a fold.
    """
    q = (np.arange(nbin)[:, None] + (np.arange(sub) + 0.5) / sub) * (period / nbin) - shift
    return model(q).mean(axis=1)


# ----------------------------------------------------------------------------- flux of each level

def source_levels(x):
    """Low and high level of a recorded square wave.

    Parameters
    ----------
    x : ndarray of float
        Samples of the source output, V.

    Returns
    -------
    (lo, hi)
        The medians of the samples below and at or above mid-range,
        ``(max(x) + min(x)) / 2``.
    """
    mid = (x.max() + x.min()) / 2
    return np.median(x[x < mid]), np.median(x[x >= mid])


def source_scales(lo, hi, v_zero, amps, offset=SOURCE_OFFSET):
    """Volts per nominal amplitude and per nominal offset of the source.

    Parameters
    ----------
    lo, hi : ndarray
        Low and high levels of the source records at each nominal amplitude.
    v_zero : float
        Baseline: the level of the source with nothing playing (median of
        the amplitude-0 record).
    amps : sequence of float
        Nominal amplitudes of the records.
    offset : float, optional
        Nominal offset all the records share.

    Returns
    -------
    (v_amp, v_off)
        Slope of the step height ``hi - lo`` against the nominal amplitude,
        and the mean low level above the baseline per unit of nominal offset.
    """
    v_amp = np.polyfit(amps, hi - lo, 1)[0]
    v_off = (lo.mean() - v_zero) / offset
    return v_amp, v_off


def flux_per_volt(Mfr, R_loop=R_LOOP, R_src=R_SRC, Phi0=_Phi0):
    """Flux at the SQUID, in ``Phi0``, per volt of source output.

    The source drives a current ``v / (R_loop + R_src)`` through the flux
    line, which couples ``Mfr`` times that current into the SQUID.

    Parameters
    ----------
    Mfr : float
        Mutual inductance of the flux-ramp line, H (from the configuration).
    R_loop, R_src : float, optional
        Resistance of the line with its series resistors and output
        impedance of the source, Ohm.
    Phi0 : float, optional
        Flux quantum, Wb.

    Returns
    -------
    numpy.float64
        ``Mfr / Phi0 / (R_loop + R_src)``, in ``Phi0`` per V. It is a
        ``numpy.float64`` (as in the paper's analysis, where ``Mfr`` came
        from the configuration as one), so a product with a single-precision
        level such as the ``v_zero`` of :func:`source_record_levels` is
        computed in double precision. A Python float would leave it in single
        precision under NumPy 2's promotion rules.
    """
    return np.float64(Mfr) / Phi0 / (R_loop + R_src)


@dataclass
class LevelCalibration:
    """Flux of the pulse levels in ``Phi0``, from the source levels and the flux-line mutual.

    A record at nominal offset ``o`` and amplitude ``a`` steps between
    ``flux(o, 0) = ko*o + b`` and ``flux(o, a) = ko*o + b + ka*a``. The offset
    and amplitude are separate settings of the source and do not share a
    scale. The three scales are stored as ``numpy.float64`` (whatever they
    are given as), so that :meth:`flux` of single-precision input is
    computed in double precision, as in the paper's analysis.
    """

    ka: float
    """Flux per unit of nominal amplitude."""
    ko: float
    """Flux per unit of nominal offset."""
    b: float
    """Flux at zero offset (the source baseline)."""

    def __post_init__(self):
        self.ka, self.ko, self.b = np.float64(self.ka), np.float64(self.ko), np.float64(self.b)

    def flux(self, offset, amp):
        """Flux of a level.

        Parameters
        ----------
        offset, amp : float or ndarray
            Nominal offset and amplitude; ``amp = 0`` is the off level of the
            offset.

        Returns
        -------
        numpy.float64 or ndarray of float64
            ``ko * offset + b + ka * amp``, in ``Phi0``.
        """
        return self.ko * offset + self.b + self.ka * amp

    @classmethod
    def from_source(cls, levels, Mfr, R_loop=R_LOOP, R_src=R_SRC, Phi0=_Phi0):
        """Build from the output of :func:`source_record_levels` (or the saved ``source_levels.npz``).

        ``ka``, ``ko`` and ``b`` are ``v_amp``, ``v_off`` and ``v_zero`` times
        :func:`flux_per_volt`. ``v_zero`` is the median of a single-precision
        record and is widened to double precision first, so that all three
        products are computed in double precision.

        Parameters
        ----------
        levels : mapping
            With the entries ``v_amp``, ``v_off`` and ``v_zero`` (V).
        Mfr, R_loop, R_src, Phi0 : float
            As in :func:`flux_per_volt`.

        Returns
        -------
        LevelCalibration
        """
        phi_per_V = flux_per_volt(Mfr, R_loop, R_src, Phi0)
        return cls(ka=float(levels["v_amp"]) * phi_per_V, ko=float(levels["v_off"]) * phi_per_V,
                   b=float(levels["v_zero"]) * phi_per_V)


# ----------------------------------------------------------------------------- calibration of the IQ plane

def rms(r):
    """Root mean square magnitude of an array of residuals.

    Parameters
    ----------
    r : ndarray (complex or real)

    Returns
    -------
    float
        ``sqrt(mean(|r|**2))``.
    """
    return np.sqrt(np.mean(np.abs(r) ** 2))


def calibration_levels(folds, offsets=OFFSETS, nbin=NBIN, n_plat=N_PLAT):
    """The plateau levels the gain and offset of the IQ plane are calibrated on.

    Every record contributes its on plateau. The off plateaus of the records
    of one offset and the static level of that offset all sit at the same
    flux, so they are averaged into one off level per offset. The paper's 31
    records at 5 offsets give 36 levels.

    Parameters
    ----------
    folds : mapping
        :func:`fold_pulse_records` output or the saved ``pulses_folded.npz``:
        ``offset``, ``amp``, ``z`` (aligned folds), ``static_offset`` and
        ``static_z``.
    offsets : sequence of float, optional
        Offsets to use; records at other offsets are left out.
    nbin, n_plat : int, optional
        Bins per period and per plateau window.

    Returns
    -------
    (offset, amp, z, off_levels)
        ``offset`` and ``amp`` (float arrays) are the nominal offset and
        amplitude of each level and ``z`` (complex, V) its measured value:
        first the off level of each offset (amplitude 0) in the order of
        ``offsets``, then the on level of each record in the order of
        ``folds``. ``off_levels`` maps each offset to the values averaged into
        its off level: the static level followed by the off plateau of each
        record.
    """
    off, amp, z = folds["offset"], folds["amp"], folds["z"]
    static = dict(zip(folds["static_offset"], folds["static_z"]))
    off_levels = {o: np.r_[static[o], [plateau_levels(z[i], nbin, n_plat)[1] for i in np.flatnonzero(off == o)]]
                  for o in offsets}
    rows = ([(o, 0.0, off_levels[o].mean()) for o in offsets]
            + [(off[i], amp[i], plateau_levels(z[i], nbin, n_plat)[0]) for i in range(len(z)) if off[i] in offsets])
    cal_off, cal_amp, cal_z = (np.array(x) for x in zip(*rows))
    return cal_off, cal_amp, cal_z, off_levels


@dataclass
class LevelFit:
    """Gain and offset of the IQ plane fitted to the calibration levels (:func:`fit_levels`)."""

    A: complex
    """Complex gain of the readout chain, V per unit of ``S21``."""
    B: complex
    """Offset of the IQ plane, V."""
    residual: np.ndarray
    """``z - (A * s21 + B)`` of every level, from the fit to all levels, in the order of the levels."""
    held_out: np.ndarray
    """Residual of each level from the fit to the levels of all other offsets, grouped by offset in the order of
    ``offsets``: the order in which the paper takes its rms. ``held_out_index`` gives the level of each entry."""
    held_out_index: np.ndarray
    """Index into the levels of each entry of ``held_out``."""

    @property
    def held_out_by_level(self):
        """``held_out`` in the order of the levels (NaN for a level whose offset was not held out)."""
        out = np.full(len(self.residual), np.nan, dtype=self.held_out.dtype)
        out[self.held_out_index] = self.held_out
        return out


def fit_levels(s21, z, offset, offsets=OFFSETS):
    """Fit ``z = A * s21 + B`` to the calibration levels, to all of them and with each offset held out.

    A linear least-squares fit of the two complex numbers. Holding out the
    levels of one offset at a time and predicting them from a fit to the
    others tests the calibration out of sample.

    Parameters
    ----------
    s21 : ndarray of complex
        Model ``S21`` of each level, e.g. the static response at the flux
        :meth:`LevelCalibration.flux` of its offset and amplitude.
    z : ndarray of complex
        Measured levels, V (:func:`calibration_levels`).
    offset : ndarray of float
        Nominal offset of each level.
    offsets : sequence of float, optional
        Offsets to hold out, in order.

    Returns
    -------
    LevelFit
        ``A``, ``B``, the in-sample residuals and the held-out residuals. The
        paper quotes ``rms(fit.residual)`` and ``rms(fit.held_out)``
        (:func:`rms`) in mV, and the former also relative to
        :func:`iq_circle_diameter`.
    """
    s21, z, offset = np.asarray(s21), np.asarray(z), np.asarray(offset)
    M = np.c_[s21, np.ones(len(z))]

    def fit(keep):
        (A, B), *_ = np.linalg.lstsq(M[keep], z[keep], rcond=None)
        return A, B, z - M @ [A, B]

    A, B, residual = fit(np.ones(len(z), bool))
    held = np.concatenate([fit(offset != o)[2][offset == o] for o in offsets])
    index = np.concatenate([np.flatnonzero(offset == o) for o in offsets])
    return LevelFit(A=A, B=B, residual=residual, held_out=held, held_out_index=index)


def off_level_scatter(off_levels):
    """Scatter of the off levels within each offset: how reproducible a plateau level is.

    Parameters
    ----------
    off_levels : dict
        Offset -> the values averaged into its off level
        (:func:`calibration_levels`).

    Returns
    -------
    float
        rms distance of every value from the mean of its offset, V.
    """
    return rms(np.concatenate([v - v.mean() for v in off_levels.values()]))


def iq_circle_diameter(A, kappa_c, kappa, alpha):
    """Diameter of the resonance circle in the IQ plane.

    ``S21 = 1 - exp(i alpha) / cos(alpha) * (kappa_c / 2) / (i Delta + kappa / 2)``
    traces a circle of diameter ``kappa_c / (kappa cos(alpha))`` as the
    detuning ``Delta`` varies; the readout scales it by ``|A|``.

    Parameters
    ----------
    A : complex
        Gain of the readout chain (:func:`fit_levels`).
    kappa_c, kappa : float
        Coupling and total loss rate of the resonator, 1/s.
    alpha : float
        Asymmetry angle of the notch, rad.

    Returns
    -------
    float
        ``|A| kappa_c / (kappa cos(alpha))``, in the units of ``A``.
    """
    return abs(A) * kappa_c / (kappa * np.cos(alpha))


# ----------------------------------------------------------------------------- edge residuals

def edge_rms(z, model, A, B, shift, nbin=NBIN, period=PERIOD, edge=EDGE, sub=SUB):
    """rms residual of a model against one fold over the edge windows.

    Parameters
    ----------
    z : ndarray of complex
        Aligned fold (:func:`align_to_rising_edge`), V.
    model : callable
        Periodic model trace of the record, ``t -> S21`` (:func:`binned_model`).
    A, B : complex
        Gain and offset of the IQ plane (:func:`fit_levels`).
    shift : float
        Delay of the model, s (:func:`fit_edge_shift`).
    nbin, period, edge, sub
        As in :func:`edge_mask` and :func:`binned_model`.

    Returns
    -------
    float
        ``sqrt(mean(|z - (A * binned_model + B)|**2))`` over the bins of
        :func:`edge_mask`, V.
    """
    in_edge = edge_mask(nbin, period, edge)
    return np.sqrt(np.mean(np.abs(z - (A * binned_model(model, shift, nbin, period, sub) + B))[in_edge] ** 2))


def edge_residual_sum(zs, models, A, B, shift, nbin=NBIN, period=PERIOD, edge=EDGE, sub=SUB):
    """Summed squared residual of a set of models against their folds over the edge windows.

    The records are summed one after the other in the order given (a plain
    Python sum, as in the paper's analysis, so the result is reproduced to
    the last bit only in the same order).

    Parameters
    ----------
    zs : sequence of ndarray of complex
        Aligned folds, V.
    models : sequence of callable
        The periodic model trace of each fold, in the same order.
    A, B, shift, nbin, period, edge, sub
        As in :func:`edge_rms`.

    Returns
    -------
    float
        Sum over the records of ``sum(|z - (A * binned_model + B)|**2)`` over
        the bins of :func:`edge_mask`, V^2.
    """
    zs, models = list(zs), list(models)
    if len(zs) != len(models):
        raise ValueError(f"{len(zs)} folds but {len(models)} models")
    in_edge = edge_mask(nbin, period, edge)
    return sum(np.sum(np.abs(z - (A * binned_model(model, shift, nbin, period, sub) + B))[in_edge] ** 2)
               for z, model in zip(zs, models))


def fit_edge_shift(zs, models, A, B, coarse=SHIFT_COARSE, fine=SHIFT_FINE, nbin=NBIN, period=PERIOD, edge=EDGE, sub=SUB):
    """The time shift, common to all records, that minimises the summed squared edge residual.

    Bin 0 of an aligned fold is its departure bin, a point defined by the
    data, and the readout delays the trace; one shift for all records
    accounts for both without changing the shape of any trace. The shift is
    scanned on the grid ``numpy.arange(*coarse)`` and then on
    ``best + numpy.arange(*fine)`` (:func:`edge_residual_sum`); the first
    minimum of each scan is kept.

    Parameters
    ----------
    zs, models, A, B
        As in :func:`edge_residual_sum`.
    coarse : tuple, optional
        ``(start, stop, step)`` of the coarse scan, s. The paper scans -300 to
        300 ns in 5 ns steps.
    fine : tuple or None, optional
        ``(start, stop, step)`` of the fine scan relative to the best coarse
        shift (the paper: -5 to 5 ns in 0.25 ns steps); ``None`` keeps the
        coarse result.
    nbin, period, edge, sub
        As in :func:`edge_rms`.

    Returns
    -------
    float
        The shift, s.
    """
    zs, models = list(zs), list(models)
    grid = np.arange(*coarse)
    best = grid[np.argmin([edge_residual_sum(zs, models, A, B, s, nbin, period, edge, sub) for s in grid])]
    if fine is None:
        return best
    grid = best + np.arange(*fine)
    return grid[np.argmin([edge_residual_sum(zs, models, A, B, s, nbin, period, edge, sub) for s in grid])]


# ----------------------------------------------------------------------------- folders of records

def _tag(x):
    return str(x).replace(".", "p")


def pulse_record_name(offset, amp):
    """File pattern of the pulse record at a nominal offset and amplitude.

    Parameters
    ----------
    offset, amp : float
        Nominal offset and amplitude.

    Returns
    -------
    str
        A ``glob`` pattern, e.g. ``IQ_0p5phi0_offset_0p25phi0amp_*.npz``.
    """
    return f"IQ_{_tag(offset)}phi0_offset_{_tag(amp)}phi0amp_*.npz"


def source_record_name(amp, offset=SOURCE_OFFSET):
    """File pattern of the bare-source record at a nominal amplitude.

    Parameters
    ----------
    amp : float
        Nominal amplitude.
    offset : float, optional
        Nominal offset of the record.

    Returns
    -------
    str
        A ``glob`` pattern, e.g. ``*0p5phi0offset_0p25phi0amp*.npz``.
    """
    return f"*{_tag(offset)}phi0offset_{_tag(amp)}phi0amp*.npz"


def source_baseline_name():
    """File pattern of the bare-source record with nothing playing (amplitude 0).

    It is matched on the amplitude alone, at any offset, as in the paper's
    analysis.

    Returns
    -------
    str
        ``*_0p0phi0amp*.npz``.
    """
    return f"*_{_tag(0.0)}phi0amp*.npz"


def _find(directory, pattern):
    """The file of ``directory`` that matches ``pattern``, found as the paper's analysis found it.

    ``glob.glob`` semantics: a wildcard does not match a leading dot, so hidden
    files such as the ``._*`` files macOS leaves on shared drives are ignored.
    If several files match, the first one ``glob`` lists is used, as in the
    original analysis, with a warning.
    """
    matches = glob.glob(os.path.join(glob.escape(os.fspath(directory)), pattern))
    if not matches:
        raise FileNotFoundError(f"no file matches {pattern!r} in {directory}")
    if len(matches) > 1:
        warnings.warn(f"{len(matches)} files match {pattern!r} in {directory}; using {os.path.basename(matches[0])}, "
                      "the first one listed, as the original analysis did", stacklevel=2)
    return Path(matches[0])


def fold_pulse_records(pulse_dir, offsets=OFFSETS, amps=AMPS, dupes=DUPES, fs=FS, period=PERIOD, plateau=PLATEAU):
    """Fold and align every pulse record of a folder.

    For each offset the amplitude-0 record gives the static IQ level; every
    other record is folded at its measured period and aligned on its rising
    edge. Records listed in ``dupes`` are skipped.

    Parameters
    ----------
    pulse_dir : path
        Folder of raw records named as :func:`pulse_record_name`.
    offsets, amps : sequence of float, optional
        Nominal grid; ``amps`` must contain 0.
    dupes : dict, optional
        Records to skip.
    fs, period, plateau : float, optional
        Sample rate (Hz), pulse period (s) and plateau window (s).

    Returns
    -------
    dict
        Arrays for ``numpy.savez``: ``offset``, ``amp``, ``period_samples``,
        ``noise`` (one entry per pulsed record), ``z`` of shape
        ``(n_records, nbin)`` with the aligned folds, ``static_offset`` and
        ``static_z`` (the static level of each offset), ``t`` the bin
        centres in s, and the scalars ``fs``, ``period``, ``plateau``,
        ``nbin``, ``n_plat``.
    """
    nbin, n_plat = int(round(period * fs)), int(round(plateau * fs))
    rows, static_offset, static_z = [], [], []
    for o in offsets:
        static = None
        for a in amps:
            if (o, a) in dupes:
                continue
            z = load_record(_find(pulse_dir, pulse_record_name(o, a))).z
            if a == 0.0:
                static = z.mean()
                static_offset.append(o)
                static_z.append(static)
                continue
            if static is None:
                raise ValueError(f"the amplitude-0 record of offset {o} must come before its pulsed records")
            P = estimate_period(z, nbin)
            zf = align_to_rising_edge(fold_pulses(z, P, nbin), static, nbin, n_plat)
            rows.append((o, a, P, zf, plateau_noise(zf, nbin, n_plat)))
    offset, amp, P, zf, noise = zip(*rows)
    return dict(offset=np.array(offset, float), amp=np.array(amp, float), period_samples=np.array(P, float),
                z=np.array(zf), noise=np.array(noise, float), static_offset=np.array(static_offset, float),
                static_z=np.array(static_z), t=(np.arange(nbin) + 0.5) / fs,
                fs=fs, period=period, plateau=plateau, nbin=nbin, n_plat=n_plat)


def record_index(folds, offset, amp):
    """Row of a set of folds holding one nominal (offset, amplitude).

    Parameters
    ----------
    folds : mapping
        :func:`fold_pulse_records` output or the saved ``pulses_folded.npz``.
    offset, amp : float
        Nominal offset and amplitude.

    Returns
    -------
    int
        The row of ``folds["z"]``.

    Raises
    ------
    KeyError
        If no record (e.g. a skipped duplicate) or more than one matches.
    """
    i = np.flatnonzero((folds["offset"] == offset) & (folds["amp"] == amp))
    if len(i) != 1:
        raise KeyError(f"{len(i)} records at offset {offset}, amplitude {amp}")
    return int(i[0])


def source_record_levels(source_dir, amps=SRC_AMPS, offset=SOURCE_OFFSET, n_excerpt=3 * NBIN):
    """Levels of every bare-source record of a folder, and the volt scales of the source.

    Parameters
    ----------
    source_dir : path
        Folder of raw single-channel (``Q``) records named as
        :func:`source_record_name`, and the baseline record
        (:func:`source_baseline_name`).
    amps : sequence of float, optional
        Nominal amplitudes to use.
    offset : float, optional
        Nominal offset of the records.
    n_excerpt : int, optional
        Samples of each record kept for plotting.

    Returns
    -------
    dict
        ``amps``, ``lo``, ``hi`` (V), ``v_zero`` (V, single precision as
        recorded), ``v_amp``, ``v_off`` (V per nominal unit,
        :func:`source_scales`), ``offset``, ``fs``, and the first
        ``n_excerpt`` samples of each record as ``excerpt`` (``(len(amps),
        n_excerpt)``, float32) and of the baseline record as
        ``excerpt_zero``.
    """
    levels, excerpt = [], []
    for a in amps:
        with np.load(_find(source_dir, source_record_name(a, offset))) as d:
            q = d["Q"]
            fs = float(d["fs"])
        levels.append(source_levels(q.astype(float)))
        excerpt.append(q[:n_excerpt].astype(np.float32))
    lo, hi = np.array(levels).T
    with np.load(_find(source_dir, source_baseline_name())) as d:
        q0 = d["Q"]
    v_zero = np.median(q0)
    v_amp, v_off = source_scales(lo, hi, v_zero, amps, offset)
    return dict(amps=np.array(amps, float), lo=lo, hi=hi, v_zero=v_zero, v_amp=v_amp, v_off=v_off, offset=offset, fs=fs,
                excerpt=np.array(excerpt), excerpt_zero=q0[:n_excerpt].astype(np.float32))


__all__ = ["FS", "PERIOD", "PLATEAU", "EDGE", "SUB", "NBIN", "N_PLAT", "SHIFT_COARSE", "SHIFT_FINE",
           "OFFSETS", "AMPS", "DUPES", "SRC_AMPS", "SOURCE_OFFSET", "R_LOOP", "R_SRC",
           "estimate_period", "fold_pulses", "departure_bin", "align_to_rising_edge", "plateau_levels", "plateau_noise",
           "bin_centres", "plateau_mask", "edge_mask", "binned_model",
           "source_levels", "source_scales", "flux_per_volt", "LevelCalibration",
           "rms", "calibration_levels", "LevelFit", "fit_levels", "off_level_scatter", "iq_circle_diameter",
           "edge_rms", "edge_residual_sum", "fit_edge_shift",
           "pulse_record_name", "source_record_name", "source_baseline_name", "fold_pulse_records", "record_index",
           "source_record_levels"]
