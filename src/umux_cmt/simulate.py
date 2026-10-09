"""High-level simulations on a :class:`~umux_cmt.device.Multiplexer`.

Three ways to ask the model for ``S21``:

* :func:`steady_state_sweep` -- a frequency sweep, each point a short time-domain
  solve run to its steady state (the paper's virtual VNA, Sec. V A). This is
  the full CMT, including the readout-power dependence through the stored
  energy.
* :func:`static_response` -- the quasi-static ``S21`` as a function of flux at
  one probe frequency, from the fixed point of the equation of motion. It is
  what the resonator does when the flux changes slowly compared with its
  ring-up time, and it is tabulated once so that it can be evaluated millions
  of times (calibrations, static controls).
* :func:`simulate` -- the time-domain response to any flux drive, Secs. V B and
  V C, started from the static steady state.
"""
from dataclasses import dataclass

import numpy as np

from .constants import Phi0
from .io import dbm_to_field
from .solver import Solver


def _probe_field(P_chip_dBm):
    return dbm_to_field(P_chip_dBm)


def _line_state(device):
    """Current of every flux line of every channel, so a simulation can leave the device as it found it."""
    return [(line, line.I) for ch in device.channels for line in ch.flux_lines.values()]


def _restore_line_state(saved):
    for line, I in saved:
        line.I = I


def steady_state_sweep(device, f_probe, P_chip_dBm, flux_ramp_current=0.0, input_current=0.0,
                       rtol=1e-5, atol=1e-9, n_t=201, settle_first=20.0, settle_next=8.0, atol_rel=None):
    """Steady-state ``S21`` of the device over a set of probe frequencies.

    Every frequency point is a time-domain CMT solve run to its steady state:
    ``settle_first / kappa`` from an empty cavity for the first point, then
    ``settle_next / kappa`` starting from the field of the previous point.
    With the defaults that is well under a millisecond per point.

    Parameters
    ----------
    device : Multiplexer
    f_probe : array_like
        Probe frequencies in Hz.
    P_chip_dBm : float
        Probe power at the device in dBm.
    flux_ramp_current, input_current : float, optional
        Static currents on the ``'FluxRamp'`` and ``'Input'`` lines in A.
    rtol, atol : float, optional
        Solver tolerances. The defaults are the ones the paper's sweeps were
        run with. ``atol`` is in units of the field, ``sqrt(J)``; at probe powers
        below about -90 dBm at the chip the field on resonance is itself only a
        few 1e-10, so the default absolute tolerance is then no longer small
        compared with the solution. The effect on the paper's sweeps is a shift
        of the fitted resonance of about 0.5 kHz (0.1% of the linewidth); pass
        ``atol_rel`` to scale the tolerance with the field instead.
    n_t : int, optional
        Output points per solve (only the last is used).
    settle_first, settle_next : float, optional
        Settling time in units of ``1/kappa`` for the first point (from an
        empty cavity) and for every following point (from the previous field).
        With ``settle_next = 8`` about ``exp(-4) = 2%`` of the change between
        neighbouring points is still unsettled, which drags the lineshape
        slightly in the sweep direction on coarse grids.
    atol_rel : float, optional
        If given, overrides ``atol`` with ``atol_rel`` times the field on
        resonance, ``sqrt(kappa_c/2) a_in / (kappa/2)`` (as :func:`simulate` does).

    Returns
    -------
    ndarray of complex
        ``S21`` at each probe frequency.
    """
    f_probe = np.atleast_1d(np.asarray(f_probe, float))
    a_in = _probe_field(P_chip_dBm)
    saved = _line_state(device)
    _, ki, kc = device.compile_static_parameters(0.0)
    kappa = np.min(ki + kc)                 # slowest channel sets the settling time
    alpha = device.get_alpha_vec()
    if atol_rel is not None:
        atol = atol_rel * np.max(np.sqrt(kc / 2) * a_in / ((ki + kc) / 2))
    out, a0 = [], None
    for fp in f_probe:
        delta, ki, kc = device.compile_static_parameters(2 * np.pi * fp, lambda t: flux_ramp_current, lambda t: input_current)
        sol = Solver(np.linspace(0, (settle_first if a0 is None else settle_next) / kappa, n_t), delta, ki, kc, a_in, alpha=alpha)
        if a0 is not None:
            sol.a0 = a0
        sol.run_solver(rtol=rtol, atol=atol)
        a0 = sol.sol.y[:, -1]
        out.append(sol.get_a_out()[-1] / a_in)
    _restore_line_state(saved)
    return np.array(out)


@dataclass
class StaticResponse:
    """Quasi-static ``S21`` of one channel tabulated against the flux on one line.

    Call it with a flux in ``Phi0`` (scalar or array) to interpolate.
    """

    phi: np.ndarray
    """Flux grid in Phi0."""
    s21: np.ndarray
    """Complex S21 on the grid."""
    f_probe: float
    P_chip_dBm: float
    line: str

    def __call__(self, phi):
        return np.interp(phi, self.phi, self.s21.real) + 1j * np.interp(phi, self.phi, self.s21.imag)


def static_response(device, f_probe, P_chip_dBm, phi_grid=None, channel=0, line="FluxRamp", n_iter=30):
    """Quasi-static ``S21`` of one channel against the flux on one of its lines.

    Setting ``da/dt = 0`` in the equation of motion gives::

        a = sqrt(kappa_c/2) a_in / (i Delta(Phi, Er) + kappa/2),   Er = |a|**2,
        S21 = 1 - (e^{i alpha}/cos alpha) sqrt(kappa_c/2) a / a_in.

    ``Delta`` depends on ``Er`` through the RF dressing, so ``a`` is found by
    iterating the first equation from ``Er = 0``. This is the response to a
    flux that changes slowly compared with the ring-up time, and it is what
    a flux-ramp measurement sees at low ramp rates.

    Parameters
    ----------
    device : Multiplexer
    f_probe : float
        Probe frequency in Hz.
    P_chip_dBm : float
        Probe power at the device in dBm.
    phi_grid : array_like, optional
        Flux grid in Phi0 (default ``linspace(-0.5, 2.5, 3001)``).
    channel : int, optional
        Which channel (the others are ignored; the response is of this notch alone).
    line : str, optional
        Flux line to sweep.
    n_iter : int, optional
        Fixed-point iterations for the stored energy.

    Returns
    -------
    StaticResponse
    """
    phi_grid = np.linspace(-0.5, 2.5, 3001) if phi_grid is None else np.asarray(phi_grid, float)
    ch = device.channels[channel]
    fl = ch.line(line)
    saved = fl.I
    kc, ki = ch.get_kappa_c(), ch.get_kappa_i()
    kappa = ki + kc
    alpha = ch.alpha
    a_in = _probe_field(P_chip_dBm)
    wp = 2 * np.pi * f_probe
    s21_grid = np.empty(len(phi_grid), complex)
    for i, phi in enumerate(phi_grid):
        fl.I = phi * Phi0 / fl.M
        Er = 0.0
        for _ in range(n_iter):
            a = np.sqrt(kc / 2) * a_in / (1j * (wp - ch.get_omega_r(Er)) + kappa / 2)
            Er = abs(a) ** 2
        s21_grid[i] = 1 - np.exp(1j * alpha) / np.cos(alpha) * np.sqrt(kc / 2) * a / a_in
    fl.I = saved
    return StaticResponse(phi=phi_grid, s21=s21_grid, f_probe=f_probe, P_chip_dBm=P_chip_dBm, line=line)


@dataclass
class TimeDomainResult:
    """Result of :func:`simulate`."""

    t: np.ndarray
    """Output times in s (the requested grid)."""
    s21: np.ndarray
    """Complex transmission ``S21(t)`` of the whole device."""
    a: np.ndarray
    """Field of every channel, shape ``(n_channels, len(t))``, in sqrt(J)."""
    a_in: float
    """Incident field amplitude in sqrt(W)."""
    f_probe: float
    P_chip_dBm: float

    @property
    def energy(self):
        """Stored energy ``|a|**2`` in J per channel."""
        return np.abs(self.a) ** 2

    @property
    def I(self):
        return self.s21.real

    @property
    def Q(self):
        return self.s21.imag


def simulate(device, f_probe, P_chip_dBm, t_out, flux_ramp_current_func=None, input_current_func=None,
             settle=5e-6, max_step=20e-9, rtol=1e-6, atol_rel=1e-6, method="BDF", a0=None):
    """Time-domain ``S21`` of the device under a flux drive.

    The solver starts ``settle`` seconds before ``t_out[0]`` from the static
    steady state for the drive at that instant (about ten ring-up times for
    a 580 kHz resonator), integrates through the drive, and returns the
    transmission on ``t_out``. The flux lines of the device are left as they
    were before the call.

    Parameters
    ----------
    device : Multiplexer
    f_probe : float
        Probe frequency in Hz.
    P_chip_dBm : float
        Probe power at the device in dBm.
    t_out : array_like
        Output times in s, increasing. For a periodic drive use one period
        starting at the feature of interest, e.g. ``np.arange(n)/n * T``.
    flux_ramp_current_func, input_current_func : callable, optional
        Line currents ``I(t)`` in A (use :func:`umux_cmt.signals.flux_to_current`
        to convert a drive in Phi0).
    settle : float, optional
        Settling time before ``t_out[0]`` in s.
    max_step : float, optional
        Largest internal step in s; 20 ns resolves the 4.5 MHz ringing of the
        bias line in the paper. Use ``min(20e-9, T/500)`` for fast ramps.
    rtol : float, optional
        Relative tolerance.
    atol_rel : float, optional
        Absolute tolerance as a fraction of the field on resonance.
    method : str, optional
        ``solve_ivp`` method; BDF by default (the equation is stiff).
    a0 : array_like, optional
        Initial field per channel; default is the static steady state at
        ``t_out[0] - settle``.

    Returns
    -------
    TimeDomainResult
    """
    t_out = np.asarray(t_out, float)
    a_in = _probe_field(P_chip_dBm)
    saved = _line_state(device)
    delta, ki, kc = device.compile_static_parameters(2 * np.pi * f_probe, flux_ramp_current_func, input_current_func)
    kappa = ki + kc
    alpha = device.get_alpha_vec()
    a_scale = np.max(np.sqrt(kc / 2) * a_in / (kappa / 2))        # field on resonance

    t_start = t_out[0] - settle
    sol = Solver(np.r_[t_start, t_out], delta, ki, kc, a_in, alpha)
    if a0 is None:
        a0 = np.array([np.sqrt(kc[j] / 2) * a_in / (1j * delta[j](t_start, 0.0) + kappa[j] / 2) for j in range(len(kc))])
    sol.a0 = np.asarray(a0, complex)
    sol.run_solver(method=method, rtol=rtol, atol=atol_rel * a_scale, max_step=max_step)
    _restore_line_state(saved)
    return TimeDomainResult(t=t_out, s21=sol.get_a_out()[1:] / a_in, a=sol.sol.y[:, 1:], a_in=a_in,
                            f_probe=f_probe, P_chip_dBm=P_chip_dBm)


def periodic_interpolant(t, z, period):
    """Wrap one period of samples ``(t, z)`` into a periodic interpolant ``q -> z(q mod period)``."""
    tt, zz = np.r_[t[-1] - period, t, t[0] + period], np.r_[z[-1], z, z[0]]
    return lambda q: np.interp(q % period, tt, zz.real) + 1j * np.interp(q % period, tt, zz.imag)
