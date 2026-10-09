"""Flux-ramp demodulation, Eq. (5) of the paper.

A sawtooth flux ramp of ``n_phi0`` flux quanta per period at rate ``f_ramp``
modulates the resonance, and with it any real readout signal ``y(t)`` derived
from ``S21(t)`` (a quadrature, or the probe phase), at

    f_m = n_phi0 * f_ramp.

The response is periodic in the *total* flux, so a detector flux ``Phi_in``
added to the ramp shifts the modulation earlier in the ramp: its phase moves
by ``-2*pi*Phi_in/Phi0`` for a rising ramp (and by ``+2*pi*Phi_in/Phi0`` for a
falling one). The modulation phase is recovered once per ramp period by
projecting the samples of that period onto reference waves at ``f_m``::

    dtheta = arctan2( sum_i y(t_i) sin(2 pi f_m t_i),  sum_i y(t_i) cos(2 pi f_m t_i) ),

and :func:`phase_to_flux` turns it into the detector flux in ``Phi0`` (up to
a constant offset, which a record without signal calibrates).

Samples within ``discard`` seconds of each reset can be dropped, which is how
readout systems deal with the transient that follows the reset (Secs. I and V
of the paper). Over an incomplete period the two reference waves are no
longer orthogonal, and harmonics of ``f_m`` (the flux response of a SQUID is
far from sinusoidal) leak into the fundamental. The default here therefore
fits the fundamental, a constant and optionally the next harmonics by linear
least squares instead of taking the raw sums; over a complete, uniformly
sampled period the fit and the sums give the same phase.
"""
import numpy as np


def demodulate(y, t, f_ramp, n_phi0, t_reset=0.0, discard=0.0, method="lstsq", n_harmonics=1, unwrap=True):
    """Modulation phase of ``y(t)`` for every complete ramp period.

    Parameters
    ----------
    y : ndarray
        Real readout signal sampled at ``t`` (a quadrature, or the probe phase).
    t : ndarray
        Sample times in s, increasing and uniformly spaced.
    f_ramp : float
        Flux-ramp rate in Hz.
    n_phi0 : float
        Flux quanta swept per ramp period; sets ``f_m = n_phi0 * f_ramp``.
    t_reset : float, optional
        Time of one ramp reset in s; the others are ``t_reset + k / f_ramp``.
    discard : float, optional
        Seconds after each reset to leave out.
    method : {'lstsq', 'sum'}, optional
        ``'sum'`` is Eq. (5) literally. ``'lstsq'`` fits
        ``a cos(2 pi f_m tau) + b sin(2 pi f_m tau) + c`` (plus ``n_harmonics - 1``
        higher harmonics) to the kept samples and returns ``arctan2(b, a)``;
        identical to the sums over a full period, and free of the
        reference-wave bias over a partial one.
    n_harmonics : int, optional
        Harmonics of ``f_m`` in the least-squares basis. Including the second
        and third harmonic removes most of their leakage into the fundamental
        when samples are discarded.
    unwrap : bool, optional
        Unwrap the recovered phase across periods.

    Returns
    -------
    t_period : ndarray
        Centre time of every demodulated period, in s.
    dtheta : ndarray
        Modulation phase per period, in rad; see :func:`phase_to_flux`.
    """
    y, t = np.asarray(y, float), np.asarray(t, float)
    T = 1 / f_ramp
    f_m = n_phi0 * f_ramp
    dt = np.median(np.diff(t))
    k = np.floor((t - t_reset) / T + 1e-9).astype(int)
    tau_all = (t - t_reset) - k * T
    keep = tau_all >= discard - 1e-12

    periods = [p for p in range(k.min(), k.max() + 1)
               if t_reset + p * T >= t[0] - 1.000001 * dt and t_reset + (p + 1) * T <= t[-1] + 1.000001 * dt]

    t_period, dtheta = [], []
    for p in periods:
        m = keep & (k == p)
        if not np.any(m):
            continue
        tau = tau_all[m]
        w = 2 * np.pi * f_m * tau
        if method == "sum":
            s, c = np.sum(y[m] * np.sin(w)), np.sum(y[m] * np.cos(w))
        elif method == "lstsq":
            cols = [np.cos(h * w) for h in range(1, n_harmonics + 1)] + [np.sin(h * w) for h in range(1, n_harmonics + 1)]
            M = np.c_[np.column_stack(cols), np.ones(m.sum())]
            coef, *_ = np.linalg.lstsq(M, y[m], rcond=None)
            c, s = coef[0], coef[n_harmonics]
        else:
            raise ValueError("method must be 'lstsq' or 'sum'")
        dtheta.append(np.arctan2(s, c))
        t_period.append(t_reset + (p + 0.5) * T)
    dtheta = np.array(dtheta)
    if unwrap and len(dtheta):
        dtheta = np.unwrap(dtheta)
    return np.array(t_period), dtheta


def phase_to_flux(dtheta, rising=True):
    """Detector flux in ``Phi0`` from the modulation phase of :func:`demodulate`.

    For a ramp that rises during each period the flux is ``-dtheta / 2pi``; for
    a falling ramp it is ``+dtheta / 2pi``. The result carries a constant offset
    set by where the ramp starts and by the trapped flux; subtract the
    demodulated phase of a record without a detector signal to remove it.
    """
    return (-1.0 if rising else 1.0) * np.asarray(dtheta) / (2 * np.pi)


def probe_phase(s21, center=None):
    """Probe phase ``theta(t)`` of a transmission record, Eq. (4): the angle of ``S21`` about ``center``.

    With ``center`` the centre of the resonance circle in the IQ plane (the
    default, estimated as the mean of the record), ``theta`` swings back and
    forth once per flux quantum of the SQUID as the resonance passes the probe
    tone and returns; this is the periodic signal Eq. (5) demodulates. It winds
    through a full turn only if the modulation carries the resonance all the
    way across the tone and the phase is unwrapped, as it is here.
    """
    s21 = np.asarray(s21)
    if center is None:
        center = s21.mean()
    return np.unwrap(np.angle(s21 - center))
