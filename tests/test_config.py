"""Configuration files: loading, the |Qc_hat| -> Qc conversion, writing and round trips."""
from pathlib import Path

import numpy as np
import tomllib

import umux_cmt as uc

CONFIGS = Path(__file__).parents[1] / "configs"


def test_loader_converts_coupling_to_dcm():
    cfg = uc.Config(CONFIGS / "nist_res2_2026-09-26.toml")
    raw = cfg.raw["resonators"]
    alpha = (raw["alpha"] + np.pi) % (2 * np.pi) - np.pi
    assert cfg.Qcs[0] == raw["Qc"] / np.cos(alpha)
    assert cfg.Qis[0] == 1 / (1 / raw["Qr"] - 1 / cfg.Qcs[0])
    dev = cfg.build()
    ch = dev[0]
    assert ch.Qc == cfg.Qcs[0] and ch.f0 == raw["f0"] and ch.alpha == alpha
    assert ch.squid.lamb == cfg.raw["squids"]["lamb"]
    assert ch.line("TrappedFlux").I == cfg.raw["squids"]["trapped_current"]
    assert ch.line("TrappedFlux").M == ch.line("FluxRamp").M
    assert abs(ch.linewidth - ch.f0 / ch.Qr) < 1e-6


def test_write_config_round_trip(tmp_path):
    src = uc.Config(CONFIGS / "nist_res2_2026-09-26.toml")
    raw_r, raw_s = src.raw["resonators"], src.raw["squids"]
    path = tmp_path / "written.toml"
    uc.write_config(path, f0=raw_r["f0"], Qr=raw_r["Qr"], Qc_abs=raw_r["Qc"], alpha=raw_r["alpha"], lamb=raw_s["lamb"],
                    Mc=raw_s["Mc"], Mfr=raw_s["Mfr"], trapped_current=raw_s["trapped_current"], Ls=raw_s["Ls"],
                    P_rf_dBm=src.P_rf_dBm, freq_fr=src.freq_fr)
    with open(path, "rb") as f:
        written = tomllib.load(f)
    for section in ("globals", "simulation", "resonators", "squids"):
        for key, val in src.raw[section].items():
            assert written[section][key] == val, (section, key)
    a, b = src.build()[0], uc.load_config(path)[0]
    assert a.get_omega_r(0.0) == b.get_omega_r(0.0)
    assert a.get_kappa_c() == b.get_kappa_c() and a.get_kappa_i() == b.get_kappa_i()


def test_multichannel_config_round_trip(tmp_path):
    src = uc.Config(CONFIGS / "nist_res2_2026-09-26.toml")
    r, s = src.raw["resonators"], src.raw["squids"]
    path = tmp_path / "chip.toml"
    uc.write_multichannel_config(path, f0=[r["f0"], r["f0"] + 7e6], Qr=[r["Qr"]] * 2, Qc_abs=[r["Qc"]] * 2, alpha=[r["alpha"]] * 2,
                                 lamb=[s["lamb"]] * 2, Mc=[s["Mc"]] * 2, Mfr=[s["Mfr"]] * 2, trapped_current=[s["trapped_current"]] * 2,
                                 Ls=s["Ls"])
    dev = uc.load_config(path)
    assert len(dev) == 2
    assert dev[1].f0 == r["f0"] + 7e6
    assert dev[0].get_omega_r(0.0) == src.build()[0].get_omega_r(0.0)
    sub = uc.load_config(path, channels=[1])
    assert len(sub) == 1 and sub[0].f0 == r["f0"] + 7e6
    delta, ki, kc = dev.compile_static_parameters(2 * np.pi * r["f0"])
    assert len(delta) == 2 and ki.shape == (2,)


def test_circle_fit_on_noise_free_points():
    """Points exactly on a circle give an eigenvalue that is zero up to round-off; the fit must still be right."""
    from umux_cmt.fitting import fit_circle
    for n in (201, 351, 1001, 3001):
        th = np.linspace(0, 1.7 * np.pi, n)
        xc, yc, r = fit_circle(0.546 + 0.456 * np.cos(th), 0.039 + 0.456 * np.sin(th))
        assert abs(xc - 0.546) < 1e-9 and abs(yc - 0.039) < 1e-9 and abs(r - 0.456) < 1e-9


def test_value_store_round_trip(tmp_path):
    store = uc.io.ValueStore(tmp_path / "v.json")
    store.record("A", -62.9018, "{:.1f}", "note")
    store.record("B", 3, "{:d}")
    assert store.value("A") == -62.9018 and store.text("A") == "-62.9" and "B" in store
    assert store.value("missing", default=1.5) == 1.5
    assert not (tmp_path / "v.json.tmp").exists()
