"""Folding, noise and IQ-plane calibration of flux-ramp records (Sec. V B of the paper).

A flux-ramp record is the demodulated transmission ``z(t) = I(t) + i Q(t)`` of
one probe tone while a sawtooth flux ramp runs on the bias line, digitised
for a number of ramp periods *without* a trigger on the ramp. The functions
here reduce such a record to what the paper compares with the CMT:

* **Folding.** All periods are averaged into one (:func:`fold_record`), in
  bins of one sample or ``max_bins`` per period, whichever is coarser. The
  phase ``u`` in ``[0, 1)`` of each bin is the mean phase of the samples in
  it. A ramp that runs slightly off its programmed rate (the 1 kHz record of
  the paper) would smear the reset when folded at the nominal period; its
  period is measured from the reset bursts instead (:func:`reset_period`),
  which also puts phase 0 at the reset.
* **Noise of the average** (:func:`even_odd_noise`): the rms difference of
  the even- and odd-period averages, divided by two.
* **Calibration of the IQ plane.** The measurement model is
  ``z(u) = A * S21(u) + B`` with the complex gain ``A`` of the readout chain
  (one rotation and one scaling of the IQ plane) and the demodulator's DC
  offset ``B``; for a given model both follow by linear least squares
  (:func:`fit_gain_offset`).
* **Alignment in phase.** Since the digitizer is not triggered on the ramp,
  the reset sits at an unknown phase of each record. :func:`best_phase_shift`
  slides a periodic model along the record and keeps the shift with the
  smallest residual; the shift moves the trace, it does not change its shape.
* **Residuals** are rms distances in the IQ plane (:func:`rms_distance`),
  quoted in units of the rms excursion of a reference trace
  (:func:`rms_excursion`).

All functions are pure numpy and return plain arrays or tuples.
"""
import numpy as np

from ..io import load_record


def fold_periods(z, period, first=0.0, max_bins=2000):
    """Average all periods of a record into one, binned in phase.

    Parameters
    ----------
    z : ndarray of complex
        The record ``I + iQ``.
    period : float
        Period in samples (need not be an integer).
    first : float, optional
        Sample index of phase 0 (e.g. the first reset); default is the start
        of the record.
    max_bins : int, optional
        Upper limit on the number of bins; below it there is one bin per
        sample of the period.

    Returns
    -------
    (u, z_fold)
        Mean phase in ``[0, 1)`` of the samples in each bin, and their mean
        value. Both have ``int(min(max_bins, period))`` entries.
    """
    u = ((np.arange(len(z)) - first) / period) % 1
    nb = int(min(max_bins, period))
    k = (u * nb).astype(int)
    n = np.bincount(k, minlength=nb)
    return np.bincount(k, u, nb) / n, (np.bincount(k, z.real, nb) + 1j * np.bincount(k, z.imag, nb)) / n


def reset_period(z, fs, rate, smooth=8):
    """Period and phase of a ramp from its reset bursts.

    The reset of the sawtooth is the largest jump of the trace within each
    nominal period. The jumps are located on the record smoothed over
    ``smooth`` samples, and a straight line through their positions gives
    the true period (slope) and the position of the first reset (intercept).

    Parameters
    ----------
    z : ndarray of complex
        The record.
    fs : float
        Sample rate in Hz.
    rate : float
        Nominal ramp rate in Hz; it fixes the search window of one period.
    smooth : int, optional
        Length of the moving average, in samples.

    Returns
    -------
    (period, first)
        Period in samples and the sample index of the first reset, for
        :func:`fold_periods`.
    """
    n = int(fs / rate)
    jump = np.abs(np.diff(np.convolve(z, np.ones(smooth) / smooth, "valid")))
    resets = [p * n + smooth // 2 + np.argmax(jump[p * n:(p + 1) * n]) for p in range(len(z) // n)]
    period, first = np.polyfit(np.arange(len(resets)), resets, 1)
    return float(period), float(first)


def fold_record(z, fs, rate, from_resets=False, max_bins=2000):
    """Fold a flux-ramp record at its nominal rate, or at the period measured from its resets.

    Parameters
    ----------
    z : ndarray of complex
        The record ``I + iQ``.
    fs : float
        Sample rate in Hz.
    rate : float
        Programmed ramp rate in Hz.
    from_resets : bool, optional
        Measure the period from the reset bursts (:func:`reset_period`) and
        put phase 0 at the first reset, instead of folding at ``fs / rate``
        from the start of the record. Use it for a ramp that does not run at
        its programmed rate.
    max_bins : int, optional
        Upper limit on the number of bins per period.

    Returns
    -------
    (u, z_fold)
        See :func:`fold_periods`.
    """
    fs, rate = float(fs), float(rate)
    if from_resets:
        period, first = reset_period(z, fs, rate)
        return fold_periods(z, period, first, max_bins)
    return fold_periods(z, fs / rate, max_bins=max_bins)


def fold_ramp_records(paths, reset_rates=(1e3,), max_bins=2000):
    """Fold a set of record files, keyed by their programmed rate.

    Parameters
    ----------
    paths : iterable of path
        ``.npz`` records readable by :func:`umux_cmt.io.load_record`.
    reset_rates : sequence of float, optional
        Rates (Hz) whose records are folded on their reset bursts
        (``from_resets=True`` in :func:`fold_record`). The paper folds only
        the 1 kHz record this way.
    max_bins : int, optional
        Upper limit on the number of bins per period (:func:`fold_periods`);
        the paper's 2000 bins resolve the reset at every rate.

    Returns
    -------
    dict
        ``{rate: (u, z_fold)}``, with ``rate`` the programmed rate of each
        record in Hz and ``(u, z_fold)`` as returned by :func:`fold_record`.
    """
    out = {}
    for path in paths:
        rec = load_record(path)
        out[rec.rate] = fold_record(rec.z, rec.fs, rec.rate, from_resets=rec.rate in reset_rates, max_bins=max_bins)
    return out


def even_odd_noise(z, period, max_bins=2000):
    """Noise of the folded average, from the even- and odd-period halves.

    The record is folded twice, once over its even and once over its odd
    periods (at the nominal ``period``, from the start of the record). The two
    averages differ by noise only. Each half averages half the periods, so
    its noise variance is twice that of the full average; the difference of
    the two independent halves has twice that again, four times the variance
    (twice the rms) of the full average. The rms of the difference divided by
    two is therefore the noise of the full average.

    Parameters
    ----------
    z : ndarray of complex
        The record.
    period : float
        Period in samples.
    max_bins : int, optional
        Upper limit on the number of bins per period.

    Returns
    -------
    float
        rms noise per bin of the folded average, in the units of ``z``.
        Divide by :func:`rms_excursion` of a trace to quote it relative to
        the signal.
    """
    i = np.arange(len(z))
    nb = int(min(max_bins, period))
    k = ((i / period) % 1 * nb).astype(int)
    even = (i // period) % 2 == 0
    half = [(np.bincount(k[m], z.real[m], nb) + 1j * np.bincount(k[m], z.imag[m], nb)) / np.bincount(k[m], minlength=nb)
            for m in (even, ~even)]
    return np.sqrt(np.mean(np.abs(half[0] - half[1]) ** 2)) / 2


def rms_excursion(z):
    """rms distance of a trace from its mean in the IQ plane: the size of the signal.

    The paper quotes every flux-ramp residual and noise level in units of the
    excursion of the folded 1 kHz record.

    Parameters
    ----------
    z : ndarray of complex
        A trace ``I + iQ``, e.g. a fold from :func:`fold_record`.

    Returns
    -------
    float
        ``sqrt(mean(|z - mean(z)|**2))``, in the units of ``z``.
    """
    return np.sqrt(np.mean(np.abs(z - z.mean()) ** 2))


def rms_distance(z_meas, z_model, scale=1.0):
    """rms distance between two traces in the IQ plane, divided by ``scale``.

    Parameters
    ----------
    z_meas, z_model : ndarray of complex
        Measured and model trace at the same points (``z_model`` may also be
        a scalar).
    scale : float, optional
        Normalisation. With the :func:`rms_excursion` of the 1 kHz fold this is
        the dimensionless residual the paper quotes; with the default 1 the
        result is in the units of ``z_meas``.

    Returns
    -------
    float
        ``sqrt(mean(|z_meas - z_model|**2)) / scale``.
    """
    return np.sqrt(np.mean(np.abs(z_meas - z_model) ** 2)) / scale


def fit_gain_offset(model_values, z_meas, return_fit=False):
    """Complex gain and offset mapping a model onto a measured trace, ``z = A * model + B``.

    A linear least-squares fit of the two complex numbers (four real
    parameters) to all points at once; the model is evaluated by the caller
    at the phases of the measured points.

    Parameters
    ----------
    model_values : ndarray of complex
        Model ``S21`` at the measured points.
    z_meas : ndarray of complex
        Measured ``I + iQ`` at the same points.
    return_fit : bool, optional
        Also return the fitted trace ``M @ [A, B]`` at the measured points.
        It equals ``A * model_values + B`` up to the last bit (the matrix
        product sums in a different order); the in-sample residuals of the
        paper's calibration tables are computed from it.

    Returns
    -------
    (A, B) or (A, B, z_fit)
        Complex gain and offset, and the fitted trace if requested.
    """
    M = np.c_[model_values, np.ones(len(z_meas))]
    (A, B), *_ = np.linalg.lstsq(M, z_meas, rcond=None)
    if return_fit:
        return A, B, M @ [A, B]
    return A, B


def periodic_phase_interpolant(u, z):
    """Periodic interpolant ``q -> z(q mod 1)`` of one period of samples at phases ``u`` in ``[0, 1)``.

    The period is replicated once on each side, so a lookup anywhere is a
    plain linear interpolation, also across the wrap from the last sample
    to the first. For samples on a time axis use
    :func:`umux_cmt.simulate.periodic_interpolant`.

    Parameters
    ----------
    u : ndarray of float
        Increasing phases of the samples within one period, in ``[0, 1)``
        (e.g. ``t * rate`` of a simulation over one ramp period, or the
        phases of a fold).
    z : ndarray of complex
        Values at ``u``.

    Returns
    -------
    callable
        ``q -> complex ndarray``, the linear interpolant of the real and
        imaginary parts at ``q mod 1``; ``q`` is any array of phases.
    """
    uu, zz = np.r_[u - 1, u, u + 1], np.r_[z, z, z]
    return lambda q: np.interp(q % 1, uu, zz.real) + 1j * np.interp(q % 1, uu, zz.imag)


def best_phase_shift(z_meas, u_meas, model_periodic, A, B, *, scale, n=4000):
    """Phase shift of a periodic model that best matches a measured fold.

    The model is evaluated at ``u_meas - s`` for ``n`` shifts ``s`` over one
    period and mapped into volts with the fixed ``A`` and ``B``; the shift
    with the smallest :func:`rms_distance` is kept (the first one on a tie).
    Nothing but the position of the trace in phase is adjusted.

    Note the order of the measured arrays: the trace comes first and its
    phases second, the reverse of the ``(u, z)`` that the folds return.
    Passing them in fold order raises a ``TypeError``.

    Parameters
    ----------
    z_meas : ndarray of complex
        Measured fold (:func:`fold_record`).
    u_meas : ndarray of float
        Its phases in ``[0, 1)``.
    model_periodic : callable
        ``q -> S21``, periodic in ``q`` with period 1
        (:func:`periodic_phase_interpolant`).
    A, B : complex
        Gain and offset from :func:`fit_gain_offset`.
    scale : float
        Normalisation of the residual (:func:`rms_distance`), applied to every
        trial before the comparison, as in the paper's analysis. It is
        required because it can matter at the last bit: dividing by a
        constant can round two nearly equal residuals to the same number, and
        the earlier shift then wins. Pass the :func:`rms_excursion` of the
        1 kHz fold to reproduce the paper (and to get the residual it quotes);
        pass 1.0 for residuals in volts.
    n : int, optional
        Number of trial shifts over the period.

    Returns
    -------
    (shift, err)
        Best shift in periods (in ``[0, 1)``) and the residual there.
    """
    if np.iscomplexobj(u_meas):
        raise TypeError("best_phase_shift takes the measured trace first and its phases second, (z_meas, u_meas, ...); "
                        "the folds return them the other way round, (u, z)")
    shifts = np.arange(0, 1, 1 / n)
    errs = [rms_distance(z_meas, A * model_periodic(u_meas - s) + B, scale) for s in shifts]
    i = int(np.argmin(errs))
    return shifts[i], errs[i]


__all__ = ["fold_periods", "reset_period", "fold_record", "fold_ramp_records", "even_odd_noise", "rms_excursion",
           "rms_distance", "fit_gain_offset", "periodic_phase_interpolant", "best_phase_shift"]
