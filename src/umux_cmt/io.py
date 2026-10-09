"""Readers for the measurement files shipped with the examples.

* VNA flux/power sweeps: ``.npz`` with ``frequencies`` (Hz), ``flux_bias``
  (nominal flux in Phi0), ``vna_powers`` (dBm at the VNA output) and
  ``s21_data`` of shape ``(power, flux, frequency)``.
* Digitizer records: ``.npz`` with ``I`` and ``Q`` (V), ``fs`` (sample rate
  in Hz) and ``f`` (the programmed drive rate in Hz).
* Touchstone ``.s2p`` two-port files from a VNA.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VnaSweep:
    """A VNA flux (and power) sweep around one or more resonances."""

    f: np.ndarray
    """Frequencies in Hz."""
    flux: np.ndarray
    """Nominal flux bias of each trace, in Phi0 (at the nominal mutual of the bias coil)."""
    powers: np.ndarray
    """VNA output power of each trace in dBm."""
    s21: np.ndarray
    """Complex S21, shape ``(len(powers), len(flux), len(f))``."""
    path: str = ""

    @property
    def shape(self):
        return self.s21.shape

    def power_index(self, p_dbm):
        """Index of the power closest to ``p_dbm``."""
        return int(np.argmin(np.abs(self.powers - p_dbm)))

    def flux_index(self, n_phi0):
        """Index of the flux bias closest to ``n_phi0``."""
        return int(np.argmin(np.abs(self.flux - n_phi0)))

    def bias_current(self, M_nominal, Phi0=2.068e-15):
        """Coil current of each trace in A, for the nominal mutual the flux axis was programmed with."""
        return self.flux * Phi0 / M_nominal


def load_vna_sweep(path):
    """Read a VNA flux/power sweep ``.npz`` into a :class:`VnaSweep`.

    Parameters
    ----------
    path : str or Path
        The ``.npz`` file.

    Returns
    -------
    VnaSweep
        ``f``, ``flux`` and ``powers`` are converted to float64 and ``s21``
        is returned as stored. The lab files hold float64 frequencies and
        flux biases, for which the conversion is a no-op, so the fits see
        exactly the arrays the paper's notebooks read; ``vna_powers`` is
        stored as int64 in some files.
    """
    with np.load(path) as d:
        return VnaSweep(f=d["frequencies"].astype(float), flux=d["flux_bias"].astype(float),
                        powers=np.atleast_1d(d["vna_powers"]).astype(float), s21=d["s21_data"], path=str(path))


@dataclass
class Record:
    """A digitizer record of a flux-ramp or pulse measurement."""

    z: np.ndarray
    """Complex ``I + 1j*Q`` samples in V (``Q`` is zero for single-channel records)."""
    fs: float
    """Sample rate in Hz."""
    rate: float
    """Programmed drive rate in Hz (ramp rate, or pulse repetition rate as logged)."""
    path: str = ""

    @property
    def t(self):
        """Sample times in s."""
        return np.arange(len(self.z)) / self.fs

    @property
    def n_periods(self):
        """Programmed periods in the record."""
        return len(self.z) * self.rate / self.fs


def load_record(path):
    """Read an ``I``/``Q`` digitizer ``.npz`` into a :class:`Record`."""
    with np.load(path) as d:
        I = d["I"].astype(float) if "I" in d else np.zeros(len(d["Q"]))
        Q = d["Q"].astype(float) if "Q" in d else np.zeros(len(I))
        return Record(z=I + 1j * Q, fs=float(d["fs"]), rate=float(d["f"]), path=str(path))


def read_s2p(path):
    """Read a Touchstone ``.s2p`` file (``# Hz S dB R 50`` or ``MA``/``RI`` formats).

    Returns
    -------
    (f, S)
        Frequencies in Hz and the complex S-matrix of shape ``(len(f), 2, 2)``,
        with ``S[:, i, j] = S_{i+1, j+1}``.
    """
    rows, fmt, unit = [], "DB", "HZ"
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s[0] == "!":
            continue
        if s[0] == "#":
            tok = s[1:].upper().split()
            unit = next((t for t in tok if t in ("HZ", "KHZ", "MHZ", "GHZ")), "HZ")
            fmt = next((t for t in tok if t in ("DB", "MA", "RI")), "DB")
            continue
        rows.append([float(x) for x in s.split()[:9]])
    d = np.array(rows, float)
    f = d[:, 0] * {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[unit]

    def cplx(a, b):
        if fmt == "DB":
            return 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
        if fmt == "MA":
            return a * np.exp(1j * np.deg2rad(b))
        return a + 1j * b

    # Touchstone 2-port column order: S11, S21, S12, S22
    S11, S21, S12, S22 = (cplx(d[:, i], d[:, i + 1]) for i in (1, 3, 5, 7))
    S = np.empty((len(f), 2, 2), complex)
    S[:, 0, 0], S[:, 1, 0], S[:, 0, 1], S[:, 1, 1] = S11, S21, S12, S22
    return f, S


class ValueStore:
    """Numbers one notebook records for another, kept in a small JSON file.

    The examples pass a few fitted quantities between notebooks this way (the
    line attenuation, the ramp amplitude, the flux-line poles). Each entry
    stores the number, a formatted string and an optional note::

        store = ValueStore("results/values.json")
        store.record("Atten", -62.9, "{:.1f}", "line attenuation fitted in 05_power_dependence")
        A = store.value("Atten", default=-62.9)
    """

    def __init__(self, path):
        self.path = Path(path)

    def _load(self):
        import json
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def record(self, name, value, fmt="{:g}", note=""):
        """Store ``value`` under ``name``; returns the formatted text.

        The file is rewritten atomically under a lock, so notebooks that run
        at the same time do not lose each other's entries.
        """
        import json
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock, "w") as lf:
            try:
                import fcntl
                fcntl.flock(lf, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            d = self._load()
            d[name] = {"value": float(value), "text": fmt.format(value), "note": note}
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        return d[name]["text"]

    def value(self, name, default=None):
        """The number stored under ``name``, or ``default`` if none has been recorded."""
        d = self._load()
        if name in d:
            return d[name]["value"]
        if default is None:
            raise KeyError(f"{name!r} has not been recorded yet; run the notebook that records it first")
        return default

    def text(self, name):
        """The formatted string stored under ``name``."""
        return self._load()[name]["text"]

    def __contains__(self, name):
        return name in self._load()

    def __repr__(self):
        d = self._load()
        return f"ValueStore({str(self.path)!r}, {len(d)} values)"

    def table(self):
        """All stored values as a text table."""
        d = self._load()
        width = max((len(k) for k in d), default=4)
        return "\n".join(f"{k:<{width}}  {v['text']:>12}  {v.get('note', '')}" for k, v in sorted(d.items()))


def dbm_to_watt(p_dbm):
    """Power in W for a level in dBm."""
    return 10 ** ((np.asarray(p_dbm, float) - 30) / 10)


def watt_to_dbm(p_w):
    """Level in dBm for a power in W."""
    return 10 * np.log10(np.asarray(p_w, float)) + 30


def dbm_to_field(p_dbm):
    """Incident field amplitude ``a_in = sqrt(P)`` for a probe power in dBm, so that ``|a_in|**2`` is in W."""
    return np.sqrt(10 ** ((p_dbm - 30) / 10))
