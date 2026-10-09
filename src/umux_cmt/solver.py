"""Time-domain integration of the coupled-mode equation of motion, Eq. (23).

For resonators hung off one feedline, each driven by the incident field::

    da_j/dt = -(i*Delta_j(t, |a_j|**2) + kappa_j/2) * a_j + sqrt(kappa_c_j/2) * a_in
    a_out   = a_in * prod_j [1 - (e^{i alpha_j}/cos alpha_j) * sqrt(kappa_c_j/2) * a_j / a_in]

``a_in`` is the incident probe field with ``|a_in|**2`` the probe power in W,
``|a_j|**2`` is the energy stored in resonator ``j`` in J, ``kappa_c`` is the
real part of the complex coupling (diameter correction method) and ``alpha``
its asymmetry angle. The equation is written in the frame rotating at the
probe tone, so the step size is set by the resonator and flux dynamics, not
by the GHz carrier. The default integrator is the implicit BDF method of
``scipy.integrate.solve_ivp``.
"""
import warnings

import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """Integrate Eq. (23) for one or more channels.

    Parameters
    ----------
    t_array : ndarray
        Times at which the solution is returned, in s. Integration runs from
        ``t_array[0]`` to ``t_array[-1]``.
    delta_func : sequence of callables
        One ``delta(t, Er)`` per channel, returning the detuning
        ``omega_p - omega_r`` in rad/s at time ``t`` and stored energy ``Er``
        (from :meth:`umux_cmt.device.Multiplexer.compile_static_parameters`).
    kappa_i, kappa_c : array_like
        Internal and coupling loss rates in rad/s, one per channel.
    a_in : float or complex
        Incident probe field, ``sqrt(P_probe / W)``. (An array is accepted
        for compatibility; only its first element is used.)
    alpha : array_like, optional
        Asymmetry angles in rad, one per channel. Defaults to zero.

    Attributes
    ----------
    a0 : ndarray
        Initial field of every channel. Zero (empty cavity) by default; set it
        before :meth:`run_solver` to start from a steady state.
    sol : OdeResult
        The ``solve_ivp`` result after :meth:`run_solver`; ``sol.y[j]`` is the
        field of channel ``j`` at ``t_array``.
    """

    def __init__(self, t_array, delta_func, kappa_i, kappa_c, a_in, alpha=None):
        self.t_array = np.asarray(t_array, dtype=float)
        self.delta_func = delta_func
        self.kappa_i = np.atleast_1d(kappa_i)
        self.kappa_c = np.atleast_1d(kappa_c)
        self.a_in = np.atleast_1d(a_in)
        self.kappa = self.kappa_i + self.kappa_c
        self.alpha = np.zeros_like(self.kappa) if alpha is None else np.atleast_1d(alpha).astype(float)
        self.g = np.sqrt(self.kappa_c / 2)
        self.rot = np.exp(1j * self.alpha) / np.cos(self.alpha)
        self.a0 = np.zeros(len(self.kappa), dtype=complex)
        self.sol = None

    def dadt(self, t, a):
        """Right-hand side of Eq. (23) for the field vector ``a``."""
        a = np.atleast_1d(a)
        delta = np.array([f(t, e) for f, e in zip(self.delta_func, np.abs(a) ** 2)])

        if np.max(np.abs(delta)) > 1e12:
            warnings.warn(f"Huge detuning detected: {delta}", RuntimeWarning, stacklevel=2)
        if not np.all(np.isfinite(delta)):
            warnings.warn(f"Non-finite detuning at t={t}: delta={delta}, a={a}; returning zero derivative",
                          RuntimeWarning, stacklevel=2)
            return np.zeros_like(a)

        return -1j * delta * a - (self.kappa / 2) * a + self.g * self.a_in[0]

    def run_solver(self, method='BDF', rtol=1e-6, atol=1e-13, max_step=np.inf):
        """Integrate from ``a0`` over ``t_array``.

        Parameters
        ----------
        method : str, optional
            Any ``solve_ivp`` method. The equation is stiff, so the implicit
            BDF method is the default.
        rtol, atol : float, optional
            Relative and absolute tolerances. ``atol`` is in units of the
            field, ``sqrt(J)``; a good choice is ``1e-6`` times the field on
            resonance, ``sqrt(kappa_c/2) * a_in / (kappa/2)``.
        max_step : float, optional
            Largest internal step in s. Set it to resolve the fastest feature
            of the flux drive (e.g. 20 ns for the 4.5 MHz ringing of a bias line).

        Returns
        -------
        OdeResult
            Also stored as ``self.sol``.
        """
        self.sol = solve_ivp(
            self.dadt, (self.t_array[0], self.t_array[-1]), self.a0, t_eval=self.t_array,
            method=method, rtol=rtol, atol=atol, max_step=max_step,
        )
        if not self.sol.success:
            warnings.warn(f"solve_ivp did not succeed: {self.sol.message}", RuntimeWarning, stacklevel=2)
        return self.sol

    def get_a_out(self):
        """Outgoing feedline field ``a_out(t)`` at ``t_array``, Eq. (24)."""
        t = 1 - (self.rot * self.g)[:, np.newaxis] * self.sol.y / self.a_in[0]
        return self.a_in[0] * np.prod(t, axis=0)

    def get_s21(self):
        """Transmission ``S21(t) = a_out(t) / a_in`` at ``t_array``, Eq. (25)."""
        return self.get_a_out() / self.a_in[0]

    def get_energy(self):
        """Stored energy ``|a_j(t)|**2`` in J, shape ``(n_channels, len(t_array))``."""
        return np.abs(self.sol.y) ** 2


def steady_state_field(delta, kappa, kappa_c, a_in):
    """Steady-state field of Eq. (23) for a fixed detuning: ``sqrt(kc/2)*a_in / (i*delta + kappa/2)``."""
    return np.sqrt(kappa_c / 2) * a_in / (1j * delta + kappa / 2)
