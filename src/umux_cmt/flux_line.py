"""The flux-ramp bias line as a linear filter from source current to coil current.

The flux ramp is generated at room temperature and reaches the SQUID through
a long bias line with parasitic reactances. The line rings after every sharp
edge (Sec. IV C of the paper). Measured as a differential two-port with a VNA,
the line gives a differential impedance ``Zdd(f)``; driven through a series
resistor ``R``, the coil current decays at the complex frequencies ``p_k``
where ``R + Zdd(p_k) = 0``, and the all-pole transfer function

    H(s) = prod_k (1 - s / p_k)**-1,        H(0) = 1,

filters the source current into the coil current, Eq. (30). ``H(0) = 1``
means the DC flux calibration passes through unchanged.

:class:`FluxLineTF` holds the poles, evaluates ``H(f)``, and filters any
periodic drive (:meth:`FluxLineTF.apply_periodic`) or one-shot drive
(:meth:`FluxLineTF.apply`). :func:`poles_from_s2p` extracts the poles from a
Touchstone file with the AAA rational approximation (``scipy >= 1.15``).
"""
from pathlib import Path

import numpy as np

from .io import read_s2p


class FluxLineTF:
    """All-pole transfer function of the flux line, from source to coil current.

    Parameters
    ----------
    poles_hz : array_like of complex
        Poles in Hz (``p_k / 2pi``), as complex-conjugate pairs with negative
        real part. For the NIST line of the paper the dominant pair sits at
        ``-0.40 +- 4.53j`` MHz.
    """

    def __init__(self, poles_hz):
        self.poles = np.asarray(poles_hz, complex)

    def __repr__(self):
        pos = self.poles[self.poles.imag > 0]
        desc = ", ".join(f"{p.imag / 1e6:.2f} MHz (Q {p.imag / (-2 * p.real):.1f})" for p in pos[np.argsort(pos.imag)])
        return f"FluxLineTF({len(self.poles)} poles: {desc})"

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_file(cls, path):
        """Load poles saved by :meth:`save` (an ``.npz`` with ``poles_MHz``)."""
        with np.load(path) as d:
            return cls(d["poles_MHz"] * 1e6)

    @classmethod
    def from_s2p(cls, path, R_series, f_min=80e3, f_max=40e6, rtol=1e-2):
        """Extract the poles from a two-port measurement of the line; see :func:`poles_from_s2p`."""
        return cls(poles_from_s2p(path, R_series, f_min, f_max, rtol)["poles_hz"])

    def save(self, path, **meta):
        """Save the poles (in MHz) and any metadata to an ``.npz``."""
        np.savez(path, poles_MHz=self.poles / 1e6, **meta)

    # ------------------------------------------------------------ descriptors

    @property
    def mode_frequencies(self):
        """Frequencies of the modes (positive imaginary parts) in Hz, ascending."""
        return np.sort(self.poles.imag[self.poles.imag > 0])

    def mode_table(self):
        """``(f, Q, tau)`` of every mode: frequency in Hz, quality factor and 1/e decay time in s."""
        pos = self.poles[self.poles.imag > 0]
        pos = pos[np.argsort(pos.imag)]
        f = pos.imag
        Q = pos.imag / (-2 * pos.real)
        tau = 1 / (-2 * np.pi * pos.real)
        return np.c_[f, Q, tau]

    @property
    def t_decay(self):
        """Slowest 1/e amplitude decay time of the line in s."""
        return np.max(-1 / (2 * np.pi * self.poles.real))

    # -------------------------------------------------------------- transfer

    def H(self, f_hz):
        """Complex transfer function ``H(f) = prod_k 1 / (1 - i f / p_k)`` at frequencies in Hz."""
        return np.prod(1 / (1 - 1j * np.asarray(f_hz)[..., None] / self.poles), axis=-1)

    def apply_periodic(self, source, period, dt=0.25e-9):
        """Steady-state coil drive for a periodic source drive.

        One period of ``source`` is sampled at mid-sample points (so that a
        step at ``t = 0`` is centred between two samples), every harmonic is
        multiplied by ``H(f)``, and the result is returned as a periodic
        interpolant. Because the filter acts on the periodic signal, the
        result is the steady state the line settles into when driven for ever.

        Parameters
        ----------
        source : callable
            ``t -> drive`` in any unit, periodic with ``period``.
        period : float
            Period in s (``1/rate`` for a flux ramp).
        dt : float, optional
            Sample spacing in s; rounded to a power of two samples per period.
            0.25 ns resolves the line's modes up to tens of MHz.

        Returns
        -------
        callable
            ``t -> coil drive``, periodic, accepting scalars or arrays.
        """
        n = int(2 ** np.ceil(np.log2(period / dt)))
        u = (np.arange(n) + 0.5) / n
        f = np.fft.rfftfreq(n, d=period / n)
        coil = np.fft.irfft(np.fft.rfft(source(u * period)) * self.H(f), n)
        u, coil = np.r_[u[-1] - 1, u, u[0] + 1], np.r_[coil[-1], coil, coil[0]]   # wrap once, so each lookup is a plain interp
        return lambda t: np.interp((t / period) % 1, u, coil)

    def step(self, t):
        """Response to a unit step at ``t = 0`` (closed form), for ``t`` in s."""
        p = self.poles * 2 * np.pi
        A = np.array([np.prod(-p) / np.prod(pk - np.delete(p, k)) for k, pk in enumerate(p)])
        t = np.asarray(t, float)
        y = np.sum(A / p * (np.exp(np.outer(t, p)) - 1), axis=-1).real
        return np.where(t >= 0, y, 0.0)

    def apply(self, source, t_span, dt=None):
        """Coil drive for a one-shot source drive at rest before ``t_span[0]``.

        The source is sampled every ``dt`` and treated as piecewise linear;
        the filter is applied exactly to that interpolant. Use
        :meth:`apply_periodic` for repetitive drives.
        """
        from scipy.signal import lfilter

        p = self.poles * 2 * np.pi
        A = np.array([np.prod(-p) / np.prod(pk - np.delete(p, k)) for k, pk in enumerate(p)])
        dt = dt or 1 / (100 * np.max(np.abs(self.poles)))
        t0, t1 = t_span
        tt = np.arange(t0, t1 + dt, dt)
        x = source(tt)
        y = np.zeros(len(x))
        for pk, Ak in zip(p, A):
            E = np.exp(pk * dt)
            I0 = (E - 1) / pk
            I1 = (E - 1 - pk * dt) / (pk ** 2 * dt)
            y += lfilter([Ak * I1, Ak * (I0 - I1)], [1, -E], x - x[0]).real
        yy = y + x[0]
        return lambda t: np.interp(np.asarray(t), tt, yy)


def differential_impedance(f, S11, S21, S12, S22, Z0=50.0):
    """Differential-mode reflection and impedance of a balanced pair measured as a two-port, Eq. (28).

    Returns ``(Sdd, Sdc, Zdd)``: the differential reflection, the differential-to-common
    mode conversion (a balance check) and the differential impedance ``2*Z0*(1+Sdd)/(1-Sdd)``.
    """
    Sdd = (S11 - S12 - S21 + S22) / 2
    Sdc = (S11 + S12 - S21 - S22) / 2
    Zdd = 2 * Z0 * (1 + Sdd) / (1 - Sdd)
    return Sdd, Sdc, Zdd


def poles_from_impedance(f, Zdd, R_series, f_min=80e3, f_max=40e6, rtol=1e-2):
    """Natural frequencies of the driven line: the zeros of ``R_series + Zdd(s)``.

    A rational function is fitted to ``R + Zdd`` on the imaginary axis (both
    signs of frequency, so the fit is real) with the AAA algorithm, and its
    roots with negative real part that come in conjugate pairs are kept.

    Returns
    -------
    dict
        ``poles_hz`` (upper-half-plane poles and their conjugates, in Hz),
        ``poles_MHz_upper`` (the distinct modes), ``n_terms`` of the fit and
        ``max_rel_err`` of the fit on the data.
    """
    from scipy.interpolate import AAA

    f = np.asarray(f, float)
    m = (f >= f_min) & (f <= f_max)
    s = np.r_[1j * f[m], -1j * f[m]] / 1e6            # units of 2 pi MHz
    Zop = np.r_[Zdd[m], np.conj(Zdd[m])] + R_series
    r = AAA(s, Zop, rtol=rtol)
    roots = r.roots()
    poles = np.array(sorted([z for z in roots if z.real < 0 and z.imag > 0 and np.min(np.abs(roots - np.conj(z))) < 0.02 * abs(z)],
                            key=lambda z: z.imag))
    poles_hz = np.r_[poles, np.conj(poles)] * 1e6
    return dict(poles_hz=poles_hz, poles_MHz_upper=poles, n_terms=len(r.poles()),
                max_rel_err=float(np.max(np.abs(r(s) - Zop) / np.abs(Zop))), fit=r)


def poles_from_s2p(path, R_series, f_min=80e3, f_max=40e6, rtol=1e-2):
    """Read a Touchstone two-port of the line and extract the coil-current poles.

    Parameters
    ----------
    path : str or Path
        ``.s2p`` file, with port 1 and 2 on the two legs of the twisted pair
        and the shields on the cryostat.
    R_series : float
        Resistance in series with the pair when the line is driven (Ohm); the
        source sees ``R_series + Zdd``.
    f_min, f_max : float, optional
        Fit window in Hz.
    rtol : float, optional
        Relative tolerance of the AAA fit.

    Returns
    -------
    dict
        As :func:`poles_from_impedance`, plus ``f``, ``Sdd``, ``Sdc`` and ``Zdd``.
    """
    f, S = read_s2p(path)
    Sdd, Sdc, Zdd = differential_impedance(f, S[..., 0, 0], S[..., 1, 0], S[..., 0, 1], S[..., 1, 1])
    out = poles_from_impedance(f, Zdd, R_series, f_min, f_max, rtol)
    out.update(f=f, Sdd=Sdd, Sdc=Sdc, Zdd=Zdd)
    return out
