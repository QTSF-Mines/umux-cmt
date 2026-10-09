"""Bit-faithfulness of the simulation core against the framework the paper was produced with.

The files in ``tests/reference`` were captured from the original code
(``umux_modeling/LW_uMUX_src`` + the final notebooks) with the same numpy/scipy
versions. Every numerical path of the public package is checked here: the
device (resonance vs flux and stored energy, loss rates, calibration), the
steady-state sweep, the quasi-static response, the flux-line filter, the
ramp and pulse simulations, and the pole extraction from the two-port file.
"""
from pathlib import Path

import numpy as np
import pytest

import umux_cmt as uc
from umux_cmt import signals

HERE = Path(__file__).parent
REF = HERE / "reference"
ROOT = HERE.parent
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"

exact = dict(rtol=0, atol=0)


def _load(name):
    return np.load(REF / name)


@pytest.fixture(scope="module")
def dev0926():
    return uc.load_config(CONFIGS / "nist_res2_2026-09-26.toml")


@pytest.fixture(scope="module")
def dev1004():
    return uc.load_config(CONFIGS / "nist_res2_2026-10-04.toml")


@pytest.mark.parametrize("tag,cfg", [("0926", "nist_res2_2026-09-26.toml"), ("1004", "nist_res2_2026-10-04.toml")])
def test_device(tag, cfg):
    r = _load(f"device_{tag}.npz")
    dev = uc.load_config(CONFIGS / cfg)
    ch = dev.channels[0]
    _, ki, kc = dev.compile_static_parameters(0.0)
    np.testing.assert_allclose(ki, r["kappa_i"], **exact)
    np.testing.assert_allclose(kc, r["kappa_c"], **exact)
    np.testing.assert_allclose(dev.get_alpha_vec(), r["alpha"], **exact)
    assert ch.Ceff == r["Ceff"] and ch.Lres == r["Lres"]
    assert ch.squid.Phi_ext == r["Phi_ext"]
    assert ch.line("TrappedFlux").I == r["Itr"][0]
    wr = np.zeros_like(r["omega_r"])
    for i, e in enumerate(r["Er"]):
        for j, p in enumerate(r["phi"]):
            ch.line("FluxRamp").I = p * uc.Phi0 / ch.line("FluxRamp").M
            wr[i, j] = ch.get_omega_r(e)
    ch.line("FluxRamp").I = 0.0
    np.testing.assert_allclose(wr, r["omega_r"], **exact)
    delta, _, _ = dev.compile_static_parameters(2 * np.pi * 5515.8e6, flux_ramp_current_func=lambda t: 1e-5 * np.sin(t * 1e5))
    dvals = np.array([delta[0](t, 3e-19) for t in r["t_delta"]])
    np.testing.assert_allclose(dvals, r["delta_vals"], **exact)
    # the reference captured Leff after that last detuning evaluation left the ramp line at I(t_delta[-1])
    assert ch.get_Leff(0.0) == r["Leff0"]


def test_steady_state_sweep(dev0926):
    r = _load("steady_state_sweep_0926.npz")
    s21 = uc.steady_state_sweep(dev0926, r["f"], float(r["P_chip"]), flux_ramp_current=0.0)
    np.testing.assert_allclose(s21, r["s21"], **exact)


def test_power_grid_subset(dev0926):
    r = _load("power_grid_0926.npz")
    for j, I in enumerate(r["I"]):
        for i, P in enumerate(r["P"]):
            s21 = uc.steady_state_sweep(dev0926, r["f"][j], float(P), flux_ramp_current=float(I))
            np.testing.assert_allclose(s21, r["s21"][j, i], **exact)


@pytest.mark.parametrize("tag", ["0926", "1004"])
def test_static_response(tag, dev0926, dev1004):
    r = _load(f"static_grid_{tag}.npz")
    dev = dev0926 if tag == "0926" else dev1004
    before = dev.channels[0].line("FluxRamp").I
    sr = uc.static_response(dev, float(r["f_probe"]), float(r["P_chip"]), phi_grid=r["phi"])
    np.testing.assert_allclose(sr.s21, r["s21"], **exact)
    assert dev.channels[0].line("FluxRamp").I == before      # the sweep restores the line it swept


def test_flux_line_transfer_function():
    r = _load("flux_line_tf.npz")
    tf = uc.FluxLineTF(r["poles_hz"])
    np.testing.assert_allclose(tf.H(r["f"]), r["H"], **exact)
    amp = float(r["ramp_amp"])
    for rate, key in ((100e3, "100k"), (500e3, "500k")):
        coil = tf.apply_periodic(signals.sawtooth(rate, amp), 1 / rate)
        # the reference used the rate form of the same FFT filter: differences are at the ulp level
        np.testing.assert_allclose(coil(r[f"t_ramp{key}"]), r[f"coil_ramp{key}"], rtol=0, atol=1e-12)
    T = float(r["T_pulse"])
    coil = tf.apply_periodic(signals.square(T, 0.0, 0.25), T)
    np.testing.assert_allclose(coil(r["t_pulse"]), r["coil_pulse_rise"], **exact)
    np.testing.assert_allclose(coil(r["t_pulse"] + T / 2), r["coil_pulse_fall"], **exact)


def test_ramp_simulations(dev0926):
    r = _load("ramp_sims_0926.npz")
    tf = uc.FluxLineTF(_load("flux_line_tf.npz")["poles_hz"])
    amp = float(_load("flux_line_tf.npz")["ramp_amp"])
    phi_reset, f_probe, P = float(r["phi_reset"]), float(r["f_probe"]), float(r["P_chip"])
    Mfr = dev0926.channels[0].line("FluxRamp").M
    for rate, key in ((100e3, "100k"), (500e3, "500k"), (1e3, "1k")):
        T = 1 / rate
        for use_tf in (False, True):
            tag = f"{key}_{'tf' if use_tf else 'ideal'}"
            if f"s21_{tag}" not in r:
                continue
            ramp = tf.apply_periodic(signals.sawtooth(rate, amp), T) if use_tf else signals.sawtooth(rate, amp)
            flux = lambda t, ramp=ramp: phi_reset + ramp(t)        # noqa: E731
            res = uc.simulate(dev0926, f_probe, P, r[f"t_{tag}"], signals.flux_to_current(flux, Mfr, uc.Phi0),
                              settle=5e-6, max_step=min(20e-9, T / 500))
            tol = dict(rtol=1e-7, atol=1e-9) if use_tf else exact
            np.testing.assert_allclose(res.s21, r[f"s21_{tag}"], **tol)


def test_pulse_simulations(dev1004):
    r = _load("pulse_sims_1004.npz")
    tf = uc.FluxLineTF(_load("flux_line_tf.npz")["poles_hz"])
    ka, ko, b, T = (float(r[k]) for k in ("ka", "ko", "b", "T"))
    f_probe, P = float(r["f_probe"]), float(r["P_chip"])
    Mfr = dev1004.channels[0].line("FluxRamp").M
    for (o, a) in [(0.5, 0.25), (1.0, 0.1)]:
        lo, hi = ko * o + b + ka * 0.0, ko * o + b + ka * a
        for use_tf in (False, True):
            tag = f"{str(o).replace('.', 'p')}_{str(a).replace('.', 'p')}_{'tf' if use_tf else 'ideal'}"
            src = signals.square(T, lo, hi)
            flux = tf.apply_periodic(src, T) if use_tf else src
            res = uc.simulate(dev1004, f_probe, P, r[f"t_{tag}"], signals.flux_to_current(flux, Mfr, uc.Phi0),
                              settle=5e-6, max_step=20e-9)
            np.testing.assert_allclose(res.s21, r[f"s21_{tag}"], **exact)


@pytest.mark.skipif(not (DATA / "flux_line_2026-10-02" / "flux_line_two_port.s2p").exists(), reason="demo data not present")
def test_poles_from_s2p():
    r = _load("flux_line_aaa.npz")
    out = uc.poles_from_s2p(DATA / "flux_line_2026-10-02" / "flux_line_two_port.s2p", R_series=float(r["R_load"]))
    np.testing.assert_allclose(out["Zdd"], r["Zdd"], **exact)
    np.testing.assert_allclose(out["Sdc"], r["Sdc"], **exact)
    np.testing.assert_allclose(out["poles_MHz_upper"], r["poles_MHz"], **exact)
    assert out["n_terms"] == int(r["n_terms"])
    stored = uc.FluxLineTF.from_file(DATA / "flux_line_2026-10-02" / "flux_line_poles.npz")
    np.testing.assert_allclose(np.sort_complex(stored.poles), np.sort_complex(out["poles_hz"]), rtol=1e-12)


def test_screening_series_against_exact_reference():
    """Across the device lamb range the Wegner series tracks the exact model to <0.3%."""
    phi_ext = np.linspace(-np.pi, np.pi, 721)
    for lamb in (0.206, 0.245, 0.315):
        w = uc.screening_weights(lamb)
        pk_pk = np.ptp(uc.screening_exact(phi_ext, 0.0, lamb))
        for ratio in (0.0, 0.1, 0.19, 0.3, 0.5, 0.61, 0.8, 1.0):
            phi_rf = 2 * np.pi * ratio
            err = np.max(np.abs(uc.screening_factor(phi_ext, phi_rf, w) - uc.screening_exact(phi_ext, phi_rf, lamb)))
            assert err < 3e-3 * pk_pk, (lamb, ratio, err / pk_pk)


def test_series_refuses_outside_its_validity():
    for lamb in (0.61, 0.8, 0.998):
        with pytest.raises(ValueError):
            uc.screening_weights(lamb)


def test_calibration_lands_on_f0(dev0926, dev1004):
    """At the calibration flux and zero power, omega_r must land exactly on f0."""
    for dev in (dev0926, dev1004):
        ch = dev.channels[0]
        saved = {name: line.I for name, line in ch.flux_lines.items()}
        for line in ch.flux_lines.values():
            line.I = 0.0
        assert abs(ch.get_omega_r(0.0) / (2 * np.pi) - ch.f0) < 1e-6
        for name, I in saved.items():
            ch.flux_lines[name].I = I


def test_power_suppresses_modulation(dev0926):
    """Peak-to-peak omega_r modulation falls monotonically with readout power (RF dressing, Fig. 7 of the paper)
    and is almost gone when the RF flux reaches the first zero of J1 (Phi_rf ~ 0.61 Phi0)."""
    ch = dev0926.channels[0]
    line = ch.line("FluxRamp")
    I_fr = np.linspace(-1.5, 1.5, 61) * uc.Phi0 / line.M
    spans = []
    for dBm in (-100, -85, -80, -75, -70):
        P_rf = 10 ** (dBm / 10) / 1000
        Er = 2 * ch.Qr ** 2 * P_rf / (ch.Qc * 2 * np.pi * ch.f0)
        f_r = []
        for I in I_fr:
            line.I = I
            f_r.append(ch.get_omega_r(Er) / (2 * np.pi))
        line.I = 0.0
        spans.append(np.ptp(f_r))
    assert all(a > b for a, b in zip(spans, spans[1:])), spans
    assert spans[-1] < 0.15 * spans[0], spans
    # at the J1 zero the modulation (first harmonic) is suppressed by two orders of magnitude
    Phi_rf = 0.61 * uc.Phi0
    Er_zero = (Phi_rf / ch.Mc) ** 2 * ch.Z0 / (16 * ch.f0)
    f_r = []
    for I in I_fr:
        line.I = I
        f_r.append(ch.get_omega_r(Er_zero) / (2 * np.pi))
    line.I = 0.0
    assert np.ptp(f_r) < 0.05 * spans[0]
