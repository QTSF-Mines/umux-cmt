"""Ideal flux drives: sawtooth ramps, square pulses and exponential pulses.

Every function returns a callable ``t -> flux`` in whatever unit the amplitude
is given in (the simulation helpers expect units of ``Phi0``). The callables
accept scalars or arrays of time in s and are periodic, so they can be passed
through :meth:`umux_cmt.flux_line.FluxLineTF.apply_periodic`.
"""
import numpy as np


def sawtooth(rate, amplitude=1.0, offset=0.0):
    """Ideal flux-ramp sawtooth: ``offset`` at each reset, rising ``amplitude`` per period.

    Parameters
    ----------
    rate : float
        Ramp repetition rate in Hz. The reset is at ``t = 0`` and every period after.
    amplitude : float, optional
        Flux swept per period (``n_Phi0`` of Eq. (3) of the paper).
    offset : float, optional
        Flux at the reset.
    """
    return lambda t: offset + amplitude * ((t * rate) % 1)


def square(period, low, high, duty=0.5):
    """Square wave: ``high`` for the first ``duty`` of each period (rising edge at ``t = 0``), then ``low``."""
    return lambda t: np.where((t / period) % 1 < duty, high, low)


def sawtooth_finite_flyback(rate, amplitude, t_flyback, offset=0.0):
    """Sawtooth whose reset takes ``t_flyback`` seconds instead of being instantaneous."""
    def func(t):
        phase = (t * rate) % 1.0
        flyback_frac = np.clip(t_flyback * rate, 0.001, 0.99)
        up = amplitude * (phase / (1 - flyback_frac))
        down = amplitude * (1.0 - (phase - (1 - flyback_frac)) / flyback_frac)
        return offset + np.where(phase < (1 - flyback_frac), up, down)
    return func


def exponential_pulse(amplitude, t_rise, t_fall, period, t_delay=0.0, offset=0.0):
    """Repeating detector-like pulse ``exp(-t/t_fall) - exp(-t/t_rise)``, normalised to ``amplitude`` at its peak."""
    t_rise = max(t_rise, 1e-12)
    if t_rise == t_fall:
        t_fall += 1e-12
    t_peak = (t_rise * t_fall / (t_fall - t_rise)) * np.log(t_fall / t_rise)
    norm = np.exp(-t_peak / t_fall) - np.exp(-t_peak / t_rise)

    def func(t):
        t_mod = (t - t_delay) % period
        return offset + amplitude * (np.exp(-t_mod / t_fall) - np.exp(-t_mod / t_rise)) / norm
    return func


def flux_to_current(flux_func, M, Phi0):
    """Turn a flux drive in units of ``Phi0`` into the line current in A: ``I = flux * Phi0 / M``."""
    return lambda t: flux_func(t) * Phi0 / M
