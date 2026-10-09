"""Device configuration files (TOML) -> :class:`~umux_cmt.device.Multiplexer`, and back.

A configuration holds every parameter of a multiplexer chip in four tables.
Per-channel keys may be a single number (shared by all channels) or a list
with one entry per channel::

    [globals]
    Phi0 = 2.068e-15

    [simulation]            # defaults for simulations, not used by the loader
    nPhi0 = 1               # flux-ramp amplitude in Phi0
    freq_fr = 1000.0        # flux-ramp rate in Hz
    P_rf_dBm = -88.9        # probe power at the chip in dBm

    [resonators]
    num_channels = 1
    f0 = 5515688502.86      # bare resonance (SQUID not loading the resonator), Hz; from the flux-response fit
    Qr = 9514.86            # loaded Q from the S21 circle fit
    Qc = 10435.94           # |Qc_hat| from the circle fit (converted to Qc = |Qc_hat|/cos(alpha))
    Qi = 103803.44          # recomputed from Qr and Qc, kept for reference
    alpha = -0.0864         # asymmetry angle, rad
    Cc = 9.4e-15            # coupling capacitance, F (metadata)
    Lc = 4.91e-11           # terminating inductance, H

    [squids]
    Ls = 2.42e-11           # loop inductance, H
    Lj = 7.6e-11            # Josephson inductance, H (ignored if lamb is given)
    lamb = 0.3183           # hysteresis parameter Ls/LJ
    Mc = 3.69e-12           # SQUID-resonator mutual, H
    Mfr = 1.44e-11          # flux-ramp line mutual, H
    Min = 2.46e-10          # detector input mutual, H
    Mtr = 1.44e-11          # mutual of the 'TrappedFlux' line, H (optional)
    trapped_current = -6.4e-5   # current on the 'TrappedFlux' line, A (optional)

``Qr``, ``Qc`` and ``alpha`` come from the circle fit of the measured S21 at
every flux point; ``f0`` (the resonance with the SQUID not loading the
resonator) and the SQUID parameters come from the flux-response fit, see
:mod:`umux_cmt.fitting`.
``Qc`` in the file is the fitted magnitude ``|Qc_hat|``; the loader converts
it to the real-part coupling Q of the diameter correction method,
``Qc = |Qc_hat| / cos(alpha)``, and recomputes ``Qi`` from ``Qr`` and that
``Qc``. Every SQUID is offset to ``phi_ext = pi/2 + lamb``, the unloaded state
(``cos(phi_tot) = 0``) that ``f0`` and ``trapped_current`` are referenced to.
"""
import tomllib
from pathlib import Path

import numpy as np

from .channel import Channel
from .device import Multiplexer
from .squid import rfSQUID


def _to_array(data, size):
    if isinstance(data, (list, np.ndarray)):
        return np.array(data)
    return np.full(size, data)


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Config:
    """A parsed configuration file with every per-channel quantity as an array.

    Attributes mirror the TOML keys (``f0s``, ``Qrs``, ``Qcs``, ``Qis``,
    ``alphas``, ``Lcs``, ``Ccs``, ``Lss``, ``Ljs``, ``lambs``, ``Mcs``,
    ``Mfrs``, ``Mins``, ``Mtrs``, ``Itrs``) plus ``Phi0``, ``num_channels`` and
    the ``[simulation]`` defaults ``nPhi0``, ``freq_fr`` and ``P_rf_dBm``.
    ``Qcs`` and ``Qis`` hold the *converted* values described in the module
    docstring; the raw file content is in ``raw``.
    """

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            self.raw = tomllib.load(f)
        cfg = self.raw

        self.Phi0 = cfg["globals"]["Phi0"]

        sim = cfg.get("simulation", {})
        self.nPhi0 = sim.get("nPhi0")
        self.freq_fr = sim.get("freq_fr")
        self.P_rf_dBm = sim.get("P_rf_dBm")

        res = cfg["resonators"]
        n = self.num_channels = res["num_channels"]
        self.f0s = _to_array(res["f0"], n)
        self.Qrs = _to_array(res["Qr"], n)
        self.Qcs = _to_array(res["Qc"], n)
        self.Qis = _to_array(res["Qi"], n)
        self.Ccs = _to_array(res["Cc"], n) if "Cc" in res else np.full(n, np.nan)
        self.Lcs = _to_array(res["Lc"], n)

        self.alphas = np.zeros(n)
        if "alpha" in res:
            self.alphas = _wrap_angle(_to_array(res["alpha"], n))
            self.Qcs = self.Qcs / np.cos(self.alphas)
            self.Qis = 1 / (1 / self.Qrs - 1 / self.Qcs)

        sqd = cfg["squids"]
        self.Lss = _to_array(sqd["Ls"], n)
        self.Ljs = _to_array(sqd["Lj"], n) if sqd.get("Lj") is not None else None
        if sqd.get("lamb"):
            self.lambs = _to_array(sqd["lamb"], n)
        elif self.Ljs is not None:
            self.lambs = self.Lss / self.Ljs
        else:
            raise ValueError("[squids] needs lamb or Lj")
        self.Mcs = _to_array(sqd["Mc"], n)
        self.Mfrs = _to_array(sqd["Mfr"], n)
        self.Mins = _to_array(sqd["Min"], n)
        self.Mtrs = _to_array(sqd["Mtr"], n) if sqd.get("Mtr") is not None else None
        self.Itrs = _to_array(sqd["trapped_current"], n) if sqd.get("trapped_current") is not None else None

    def __repr__(self):
        return f"Config({self.path.name!r}, {self.num_channels} channel(s))"

    def build(self, channels=None, name=None):
        """Build the :class:`~umux_cmt.device.Multiplexer`.

        Parameters
        ----------
        channels : int or sequence of int, optional
            Build only the first ``channels`` channels, or only the listed
            channel indices. Default: all.
        name : str, optional
            Name of the device; defaults to the file stem.
        """
        if channels is None:
            indices = range(self.num_channels)
        elif np.isscalar(channels):
            indices = range(int(channels))
        else:
            indices = list(channels)

        umux = Multiplexer(name=name or self.path.stem)
        for n in indices:
            squid = rfSQUID(Ls=self.Lss[n], lamb=self.lambs[n])
            squid.Phi_ext = self.Phi0 * ((np.pi / 2.0) + self.lambs[n]) / (2 * np.pi)

            squid.add_flux_line("FluxRamp", M=self.Mfrs[n], initial_current=0.0)
            if self.Mtrs is not None:
                squid.add_flux_line("TrappedFlux", M=self.Mtrs[n], initial_current=0.0)
            squid.add_flux_line("Input", M=self.Mins[n], initial_current=0.0)

            Qr, Qc = self.Qrs[n], self.Qcs[n]
            Qi = 1 / (1 / Qr - 1 / Qc) if (Qr > 0 and Qc > 0 and (1 / Qr - 1 / Qc) > 1e-12) else 1e12

            channel = Channel(f0=self.f0s[n], Qi=Qi, Qc=Qc, Lc=self.Lcs[n], Mc=self.Mcs[n], squid=squid,
                              alpha=self.alphas[n], Z0=umux.Z0, name=f"Ch{n + 1}", Cc=self.Ccs[n])

            if self.Itrs is not None and "TrappedFlux" in squid.flux_lines:
                squid.flux_lines["TrappedFlux"].I = self.Itrs[n]

            umux.add(channel)
        return umux


def load_config(path, channels=None, name=None):
    """Read a TOML configuration and build the device (shorthand for ``Config(path).build()``)."""
    return Config(path).build(channels=channels, name=name)


CONFIG_TEMPLATE = """[globals]
Phi0 = {Phi0}

[simulation]
nPhi0 = {nPhi0}
freq_fr = {freq_fr}
P_rf_dBm = {P_rf_dBm}

[resonators]
num_channels = 1
f0 = {f0}
Qr = {Qr}
Qc = {Qc}
Qi = {Qi}
alpha = {alpha}
Cc = {Cc}
Lc = {Lc}

[squids]
Ls = {Ls}
Lj = {Lj}
lamb = {lamb}
Mc = {Mc}
Mfr = {Mfr}
Min = {Min}
Mtr = {Mtr}
trapped_current = {trapped_current}
"""


def write_config(path, f0, Qr, Qc_abs, alpha, lamb, Mc, Mfr, trapped_current, Ls=24.2e-12, Lc=4.91e-11, Cc=9.4e-15,
                 Min=2.46e-10, Phi0=2.068e-15, P_rf_dBm=-89.0, freq_fr=1000.0, nPhi0=1):
    """Write a single-channel configuration from fitted parameters.

    Parameters
    ----------
    path : str or Path
        Output file.
    f0, lamb, Mc, Mfr, trapped_current : float
        From the flux-response fit (:func:`umux_cmt.fitting.fit_flux_response`
        and :func:`umux_cmt.fitting.relative_I_trapped`).
    Qr, Qc_abs, alpha : float
        From the S21 circle fit. ``Qc_abs`` is the fitted ``|Qc_hat|``; the
        loader converts it to the DCM ``Qc``.
    Ls : float, optional
        SQUID loop inductance assumed in the flux-response fit.
    Lc, Cc, Min : float, optional
        Terminating inductance, coupling capacitance and detector-input
        mutual, usually design values.
    P_rf_dBm : float, optional
        Probe power at the chip at which the parameters were extracted
        (recorded in ``[simulation]``).

    Returns
    -------
    str
        The file content.
    """
    text = CONFIG_TEMPLATE.format(
        Phi0=Phi0, nPhi0=nPhi0, freq_fr=freq_fr, P_rf_dBm=P_rf_dBm,
        f0=f0, Qr=Qr, Qc=Qc_abs, Qi=1 / (1 / Qr - np.cos(alpha) / Qc_abs), alpha=alpha, Cc=Cc, Lc=Lc,
        Ls=Ls, Lj=Ls / lamb, lamb=lamb, Mc=Mc, Mfr=Mfr, Min=Min, Mtr=Mfr, trapped_current=trapped_current,
    )
    Path(path).write_text(text)
    return text


def write_multichannel_config(path, f0, Qr, Qc_abs, alpha, lamb, Mc, Mfr, trapped_current, Ls=24.2e-12, Lc=4.91e-11,
                              Cc=9.4e-15, Min=2.46e-10, Phi0=2.068e-15, P_rf_dBm=-89.0, freq_fr=1000.0, nPhi0=1):
    """Write a configuration with one entry per channel (arrays of equal length)."""
    f0, Qr, Qc_abs, alpha, lamb, Mc, Mfr, Itr = (np.asarray(x, float) for x in (f0, Qr, Qc_abs, alpha, lamb, Mc, Mfr, trapped_current))
    n = len(f0)
    if not all(len(x) == n for x in (Qr, Qc_abs, alpha, lamb, Mc, Mfr, Itr)):
        raise ValueError("all per-channel arrays must have the same length")

    def arr(x):
        return "[ " + ", ".join(repr(float(v)) for v in x) + " ]"

    Qi = 1 / (1 / Qr - np.cos(alpha) / Qc_abs)
    text = (
        "[globals]\n"
        f"Phi0 = {Phi0}\n\n"
        "[simulation]\n"
        f"nPhi0 = {nPhi0}\n"
        f"freq_fr = {freq_fr}\n"
        f"P_rf_dBm = {P_rf_dBm}\n\n"
        "[resonators]\n"
        f"num_channels = {n}\n"
        f"f0 = {arr(f0)}\n"
        f"Qr = {arr(Qr)}\n"
        f"Qc = {arr(Qc_abs)}\n"
        f"Qi = {arr(Qi)}\n"
        f"alpha = {arr(alpha)}\n"
        f"Cc = {Cc}\n"
        f"Lc = {Lc}\n\n"
        "[squids]\n"
        f"Ls = {Ls}\n"
        f"Lj = {arr(Ls / lamb)}\n"
        f"lamb = {arr(lamb)}\n"
        f"Mc = {arr(Mc)}\n"
        f"Mfr = {arr(Mfr)}\n"
        f"Min = {Min}\n"
        f"Mtr = {arr(Mfr)}\n"
        f"trapped_current = {arr(Itr)}\n"
    )
    Path(path).write_text(text)
    return text
