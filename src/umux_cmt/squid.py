"""The rf-SQUID: screening response, flux lines and the Kepler equation.

The resonator of a uMux channel is terminated by an inductor ``Lc`` that is
screened by an rf-SQUID through the mutual inductance ``Mc``. The screening
changes the effective termination inductance,

    Leff = Lc - (Mc**2 / Ls) * S(phi_ext, phi_rf),

where ``S`` is the dimensionless *screening response* of the SQUID. In the
low-power limit ``S = lamb*cos(phi_tot) / (1 + lamb*cos(phi_tot))`` with
``phi_tot`` the total flux phase from Kepler's equation, Eq. (1) and (8) of
the paper. At finite readout power the response is dressed by the RF flux,
and this module evaluates it with the series expansion of Wegner, Enss and
Kempf (2022), Eq. (10) of the paper, which is a function of the *external*
flux and therefore needs no Kepler solve inside the equation of motion.

Notation follows the paper: ``Ls`` and ``LJ`` are the geometric and Josephson
inductances of the SQUID, ``lamb = Ls/LJ`` is the hysteresis parameter
(``lambda`` is reserved in Python). Capitalised ``Phi_*`` are fluxes in Wb,
lower-case ``phi_*`` are phases ``2*pi*Phi/Phi0``.
"""
import numpy as np
import scipy.special as sp

from .constants import Phi0

###############################################################################
#                          SQUID SCREENING MODEL                              #
###############################################################################
#
# Table 1 of M. Wegner, C. Enss and S. Kempf, Supercond. Sci. Technol. 35,
# 075011 (2022), reproduced verbatim: a 10th-order Taylor expansion in lamb of
# the screening response of a non-hysteretic rf-SQUID driven at finite RF flux,
# quoted as valid for lamb < 0.6.
#
# The expansion is a sum of 36 terms
#
#     p_ij = a_ij * lamb**b_ij * J1(c_ij*phi_rf) * cos(c_ij*phi_ext),
#
# entering the termination inductance as
#
#     Leff = Lc - (Mc**2/Ls) * (2*lamb/phi_rf) * sum_ij p_ij.
#
# Verified against a numerically exact reference in tests/test_core_equivalence.py.
#
# Accuracy note. In the zero-RF-flux limit the exact response follows from
# Kepler's equation as a Kapteyn series, S = sum_c 2(-1)^(c+1) J_c(c*lamb)
# cos(c*phi_ext), whose Taylor coefficients agree with the table only through
# the lamb**3 terms; from lamb**4 upward the tabulated values differ from the
# exact ones (e.g. the lamb**5 cos(phi_ext) coefficient is 1/16 here and 1/192
# exactly). The consequence is a smooth error of 0.07% of the modulation swing
# at lamb = 0.2, 0.27% at lamb = 0.32 (about 2 kHz of resonance shift for the
# channel of the paper, under 1% of its linewidth) and 1.4% at lamb = 0.5. The
# table is kept exactly as the paper's results were computed with it;
# `screening_exact` gives the exact low-power reference for comparison.

WEGNER_A = np.array([
    1, -1/2, -1/8, 3/8, 1/8, -5/16, 1/16, -5/32, 9/32, -5/64, 3/16, -17/64,
    -15/512, 57/512, -115/512, 133/512, 21/512, -77/512, 137/512, -267/1024,
    21/1024, -35/512, 103/512, -651/2048, 547/2048, -63/2048, 27/256,
    -1089/4096, 193/512, -1139/4096, -105/8192, 435/8192, -2595/16384,
    5705/16384, -7317/16384, 4807/16384,
])
WEGNER_B = np.array([0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                     8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10])
WEGNER_C = np.array([1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 6, 1, 3, 5, 7, 2, 4, 6, 8,
                     1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11])

LAMB_MAX = 0.6
"""Upper limit of the hysteresis parameter for which the series is valid."""


def screening_weights(lamb):
    """Collapse the 36 table terms onto the 11 flux harmonics ``c = 1..11``.

    Each term depends on the flux only through ``J1(c*phi_rf)`` and
    ``cos(c*phi_ext)``, so terms sharing ``c`` can be summed once per SQUID::

        S = sum_c w_c(lamb) * [2*J1(c*phi_rf)/phi_rf] * cos(c*phi_ext),
        w_c = lamb * sum_{c_ij = c} a_ij * lamb**b_ij.

    This is an exact regrouping, not a further approximation.

    Parameters
    ----------
    lamb : float
        Hysteresis parameter ``Ls/LJ``. Must be below ``LAMB_MAX``.

    Returns
    -------
    (w, c) : tuple of ndarray
        Weights ``w_c`` and the harmonic indices ``c`` as floats.
    """
    if lamb > LAMB_MAX:
        raise ValueError(
            f"lamb = {lamb:.3f} exceeds {LAMB_MAX}, beyond the validity of the "
            "Wegner series; the screening response would be wrong by several "
            "percent or more.")
    harmonics = np.unique(WEGNER_C)
    w = lamb * np.array([np.sum(WEGNER_A[WEGNER_C == c] * lamb ** WEGNER_B[WEGNER_C == c])
                         for c in harmonics])
    return w, harmonics.astype(float)


def screening_factor(phi_ext, phi_rf, weights):
    """Dimensionless screening response ``S = (Lc - Leff) * Ls / Mc**2``.

    The ``0/0`` limit at zero readout power is taken analytically: since
    ``2*J1(c*phi_rf)/phi_rf -> c``, ``phi_rf = 0`` is a valid argument.

    Parameters
    ----------
    phi_ext : float or ndarray
        External flux phase ``2*pi*Phi_ext/Phi0``.
    phi_rf : float or ndarray
        RF flux amplitude phase ``2*pi*Phi_rf/Phi0``.
    weights : tuple
        Output of :func:`screening_weights` for this SQUID.

    Returns
    -------
    float or ndarray
        The screening response, broadcast over the inputs.
    """
    w, c = weights
    phi_ext = np.asarray(phi_ext, dtype=float)
    phi_rf = np.asarray(phi_rf, dtype=float)
    scalar = phi_ext.ndim == 0 and phi_rf.ndim == 0

    x = phi_rf[..., np.newaxis] * c
    tiny = np.abs(x) < 1e-9
    safe = np.where(tiny, 1.0, x)
    kernel = np.where(tiny, c, 2.0 * sp.j1(safe) / safe * c)

    S = np.sum(w * kernel * np.cos(phi_ext[..., np.newaxis] * c), axis=-1)
    return float(S) if scalar else S


def solve_kepler(phi_ext, lamb, tol=1e-12, max_iter=25):
    """Solve ``phi_tot + lamb*sin(phi_tot) = phi_ext`` by Newton-Raphson.

    This is Kepler's equation, Eq. (8) of the paper without the RF term. It is
    iterated on the branch wrapped into ``[-pi, pi]``, where the iteration is
    unconditionally stable for ``lamb < 1``, and then unwrapped back.

    Parameters
    ----------
    phi_ext : float or ndarray
        External flux phase.
    lamb : float
        Hysteresis parameter (``< 1``, non-hysteretic SQUID).

    Returns
    -------
    float or ndarray
        Total flux phase ``phi_tot``.
    """
    if lamb < 1e-12:
        return phi_ext
    phi_ext = np.asarray(phi_ext, dtype=float)
    wrapped = (phi_ext + np.pi) % (2 * np.pi) - np.pi
    phi_tot = np.copy(wrapped)
    for _ in range(max_iter):
        step = ((phi_tot + lamb * np.sin(phi_tot) - wrapped)
                / (1.0 + lamb * np.cos(phi_tot)))
        phi_tot -= step
        if np.max(np.abs(step)) < tol:
            break
    return phi_tot + (phi_ext - wrapped)


def screening_exact(phi_ext, phi_rf, lamb, n_period=64):
    """Numerically exact screening response, for any ``lamb < 1``.

    Solves Kepler's equation over one RF period and takes the fundamental
    ``sin(wt)`` Fourier component of the screening current ``Ic*sin(phi_tot)``;
    the harmonics do not meet the resonance condition and are discarded, exactly
    as in the derivation of the series. The integrand is analytic and periodic,
    so the trapezoid rule converges spectrally and 64 points reach machine
    precision. This is the oracle the Wegner series is tested against; it is
    too slow to use inside the equation of motion.
    """
    phi_ext = np.asarray(phi_ext, dtype=float)
    if abs(phi_rf) < 1e-9:
        phi_tot = solve_kepler(phi_ext, lamb)
        return lamb * np.cos(phi_tot) / (1.0 + lamb * np.cos(phi_tot))
    wt = 2 * np.pi * np.arange(n_period) / n_period
    phi_tot = solve_kepler(phi_ext[..., np.newaxis] + phi_rf * np.sin(wt), lamb)
    return lamb * 2.0 * np.mean(np.sin(phi_tot) * np.sin(wt), axis=-1) / phi_rf


###############################################################################
#                                 FLUX LINE                                   #
###############################################################################

class FluxLine:
    """A current-carrying line coupled to a SQUID loop by a mutual inductance.

    Setting ``line.I`` (in A) sets the flux ``line.phi = M * I`` (in Wb) that
    the line delivers to the loop, and vice versa. Any value accepted by
    ``numpy.asanyarray`` works, so a line can hold a whole array of currents at
    once for vectorised sweeps.

    Parameters
    ----------
    name : str
        Line name, e.g. ``'FluxRamp'``, ``'Input'`` or ``'TrappedFlux'``.
    M : float
        Mutual inductance to the loop in H.
    initial_current : float, optional
        Line current in A.
    """

    def __init__(self, name, M, initial_current=0.0):
        self.name = name
        self.M = M
        self._I = 0.0
        self._phi = 0.0
        self.I = initial_current

    def __repr__(self):
        return f"FluxLine({self.name!r}, M={self.M * 1e12:.2f} pH)"

    @property
    def I(self):
        """Line current in A."""
        return self._I

    @I.setter
    def I(self, new_I):
        self._I = np.asanyarray(new_I)
        self._phi = self.M * self._I

    @property
    def phi(self):
        """Flux delivered to the loop in Wb."""
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        self._phi = np.asanyarray(new_phi)
        self._I = self._phi / self.M if self.M != 0 else np.zeros_like(self._phi)

    @property
    def n_phi0(self):
        """Flux delivered to the loop in units of ``Phi0``."""
        return self._phi / Phi0

    @n_phi0.setter
    def n_phi0(self, n):
        self.phi = np.asanyarray(n) * Phi0


###############################################################################
#                                  rf-SQUID                                   #
###############################################################################

class rfSQUID:
    """A non-hysteretic rf-SQUID: a loop of inductance ``Ls`` closed by a
    junction of Josephson inductance ``LJ``, driven by any number of flux lines.

    The SQUID knows nothing about the resonator. It reports its screening
    response :meth:`get_screening_factor` for the current external flux (the
    static offset ``Phi_ext`` plus the flux of every line) and a given RF flux
    amplitude; the :class:`~umux_cmt.channel.Channel` turns that into an
    effective inductance and a resonant frequency.

    Parameters
    ----------
    Ls : float
        SQUID loop geometric inductance in H.
    lamb : float, optional
        Hysteresis parameter ``Ls/LJ``. Takes precedence over ``LJ``.
    LJ : float, optional
        Josephson inductance ``Phi0/(2*pi*Ic)`` in H. Ignored if ``lamb`` is
        given.
    Phi_ext : float, optional
        Static flux offset in Wb, on top of the flux lines. The configuration
        loader sets it to ``Phi0*(pi/2 + lamb)/(2*pi)``, the state in which the
        SQUID does not load the resonator (``cos(phi_tot) = 0``), so that the
        bare resonance ``f0`` and the trapped flux are referenced to that point.
    """

    def __init__(self, Ls, lamb=None, LJ=None, Phi_ext=0.0):
        self.name = "SQUID"
        self.Ls = Ls
        if lamb is not None:
            self.lamb = lamb
            self.LJ = Ls / lamb
        elif LJ is not None:
            self.LJ = LJ
            self.lamb = Ls / LJ
        else:
            raise ValueError("give either lamb or LJ")

        self.Phi_ext = Phi_ext
        self.flux_lines = {}
        self._weights = (None, None)
        self._weights_lamb = None

    def __repr__(self):
        lines = ", ".join(self.flux_lines)
        return f"rfSQUID(Ls={self.Ls * 1e12:.2f} pH, lamb={self.lamb:.4f}, lines=[{lines}])"

    def add_flux_line(self, name, M, initial_current=0.0):
        """Attach a current-carrying line coupled to the loop by mutual ``M``.

        Returns
        -------
        FluxLine
            The created line, also available as ``self.flux_lines[name]``.
        """
        self.flux_lines[name] = FluxLine(name, M, initial_current)
        return self.flux_lines[name]

    def get_Phi_ext(self):
        """Total quasi-static external flux in Wb: the offset plus every line."""
        Phi = self.Phi_ext
        for line in self.flux_lines.values():
            Phi = Phi + line.phi
        return Phi

    @property
    def phi_ext(self):
        """Total external flux phase ``2*pi*Phi_ext/Phi0``."""
        return 2.0 * np.pi * self.get_Phi_ext() / Phi0

    def get_phi_tot(self):
        """Total flux phase from Kepler's equation at the current external flux
        (low-power limit). Useful for diagnostics; not used by the solver."""
        return solve_kepler(self.phi_ext, self.lamb)

    def get_screening_factor(self, Phi_rf=0.0):
        """Screening response ``S = (Lc - Leff) * Ls / Mc**2`` from the Wegner series.

        A function of the external flux, not the total flux, so it carries the
        readout-power dependence with no Kepler solve and no self-consistency
        loop.

        Parameters
        ----------
        Phi_rf : float or ndarray, optional
            RF flux amplitude at the SQUID in Wb. Defaults to the low-power
            limit.

        Returns
        -------
        float or ndarray
            The screening response.
        """
        if self._weights_lamb != self.lamb:
            self._weights = screening_weights(self.lamb)
            self._weights_lamb = self.lamb
        k = 2.0 * np.pi / Phi0
        return screening_factor(k * self.get_Phi_ext(),
                                k * np.asarray(Phi_rf, dtype=float),
                                self._weights)
