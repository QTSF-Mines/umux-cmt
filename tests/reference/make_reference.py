"""Capture reference values from the ORIGINAL code the paper was produced with.

This script wrote the first set of reference files in this folder (on 2026-10-08, in about 30 s):
``device_0926``, ``device_1004``, ``steady_state_sweep_0926``, ``power_grid_0926``, ``static_grid_0926``,
``flux_line_tf``, ``flux_line_aaa``, ``ramp_sims_0926``, ``pulse_sims_1004``, ``static_grid_1004``,
``fits_0926``, ``fits_1004``, ``comb_0926``, ``iq_folds_0926`` and ``pulse_folds_1004`` (``.npz``). The
tests compare ``umux_cmt`` against them exactly. The fitting references (section H) come from
``resonator_fitting/include`` (``resonance_fitting.refined_fit``/``calculate_tau``, ``lambdafit``,
``all_resonances``) driven by verbatim copies of the ``notch``, ``fit_trace`` and ``fit_flux`` cells of the
paper's parameter notebook (``umux_modeling/LW_uMUX_src/jupyter/final/43_characterization/res2_params.ipynb``);
``fit_flux`` here also returns its five differential-evolution solutions. ``make_fitting_coverage.py`` adds the
whole-sweep fitting references.

The original code and the raw measurement files are not part of this package. They live in the lab repository
that holds this one (``umux_modeling/LW_uMUX_src``, ``resonator_fitting``, ``NIST_uMux_measurment_data``); set
``UMUX_ORIGINAL_ROOT`` to it if this checkout is not inside it. The files are written to this folder, or to
``UMUX_REFERENCE_OUT``. Run it with the original environment (Python 3.14, numpy 2.5.0, scipy 1.18.0, as the
tests) and without writing bytecode into the original tree::

    PYTHONDONTWRITEBYTECODE=1 UMUX_REFERENCE_OUT=/tmp/ref python tests/reference/make_reference.py

The script only reads the original code and data. Apart from the paths above it is unchanged from the run that
wrote the files.
"""
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.environ.get("UMUX_ORIGINAL_ROOT", str(Path(__file__).resolve().parents[3]))
SRC = REPO + "/umux_modeling/LW_uMUX_src"
DATA = REPO + "/NIST_uMux_measurment_data/"
OUT = os.environ.get("UMUX_REFERENCE_OUT", str(Path(__file__).resolve().parent)) + "/"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, SRC)
sys.path.insert(0, REPO)
os.chdir(SRC)

from include.config_loader import ConfigLoader  # noqa: E402
from include.engine import Solver  # noqa: E402
from resonator_fitting.include import resonance_fitting as rf, lambdafit as lf, all_resonances as ar  # noqa: E402

ATTEN = -62.90189079698751          # values.json Atten
EXTRACT_VNA_DBM = -26.0
P_CHIP = EXTRACT_VNA_DBM + ATTEN     # -88.90189...
RAMP_AMP = 0.945856249388244         # values.json RampAmpPhi
PHI_RESET = 0.1845                   # values.json IQPhiReset
Mfr_nominal = 14.4e-12
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# A/B. Device: omega_r grid, kappas, calibration constants, for both configs
# ----------------------------------------------------------------------------------------------
def device_reference(cfg_name, tag):
    loader = ConfigLoader(f"configs/{cfg_name}")
    umux = loader.build_multiplexer()
    ch = umux.channels[0][0]
    squid = ch.squids[0]
    _, ki, kc = umux.compile_static_parameters(0.0)
    alpha = umux.get_alpha_vec()
    phi = np.linspace(-0.6, 1.6, 221)
    Er = np.array([0.0, 1e-20, 1e-19, 6e-19, 2e-18])
    wr = np.zeros((len(Er), len(phi)))
    for i, e in enumerate(Er):
        for j, p in enumerate(phi):
            squid.flux_lines["FluxRamp"].I = p * loader.Phi0 / loader.Mfrs[0]
            wr[i, j] = ch.get_omega_r(e)
    squid.flux_lines["FluxRamp"].I = 0.0
    # detuning function evaluated through compile_static_parameters
    delta, _, _ = umux.compile_static_parameters(2 * np.pi * 5515.8e6, flux_ramp_current_func=lambda t: 1e-5 * np.sin(t * 1e5))
    tt = np.linspace(0, 1e-4, 50)
    dvals = np.array([delta[0](t, 3e-19) for t in tt])
    np.savez(OUT + f"device_{tag}.npz", phi=phi, Er=Er, omega_r=wr, kappa_i=ki, kappa_c=kc, alpha=alpha,
             f0=loader.f0s, Qi=loader.Qis, Qc=loader.Qcs, Qr=loader.Qrs, lamb=loader.lambs, Mc=loader.Mcs, Mfr=loader.Mfrs,
             Mtr=loader.Mtrs, Itr=loader.Itrs, Ceff=ch.Ceff, Lres=ch.Lres, Phi_ext=squid.Phi_ext, Phi0=loader.Phi0,
             Leff0=ch.get_Leff(0.0), t_delta=tt, delta_vals=dvals, P_rf_dBm=loader.P_rf_dBm)
    log(f"device {tag}: f0 {loader.f0s[0]:.3f}, ki {ki[0]:.6e}, kc {kc[0]:.6e}, Ceff {ch.Ceff:.6e}, Lres {ch.Lres:.6e}")
    return loader, umux, ch


loader, umux, res = device_reference("NIST_uMux_res2_2026-09-26.toml", "0926")
loader_oct, umux_oct, res_oct = device_reference("NIST_uMux_res2_2026-10-04.toml", "1004")

# ----------------------------------------------------------------------------------------------
# C. Steady-state sweep (51_static/s21_simulation s21_cmt)
# ----------------------------------------------------------------------------------------------
_, ki, kc = umux.compile_static_parameters(0.0)
kappa = (ki + kc)[0]


def s21_cmt(umux, kappa, f_sim, P, I):
    a_in = np.sqrt(10 ** ((P - 30) / 10))
    out, a0 = [], None
    for fp in f_sim:
        delta, ki, kc = umux.compile_static_parameters(2 * np.pi * fp, lambda t: I)
        sol = Solver(np.linspace(0, (20 if a0 is None else 8) / kappa, 201), delta, ki, kc, a_in, alpha=umux.get_alpha_vec())
        if a0 is not None:
            sol.a0 = a0
        sol.run_solver(rtol=1e-5, atol=1e-9)
        a0 = sol.sol.y[:, -1]
        out.append(sol.get_a_out()[-1] / a_in)
    return np.array(out)


res.squids[0].flux_lines["FluxRamp"].I = 0.0
fr = res.get_omega_r(0.0) / (2 * np.pi)
f_sim = np.linspace(fr - 3e6, fr + 3e6, 1201)
s21_sim = s21_cmt(umux, kappa, f_sim, P_CHIP, 0.0)
np.savez(OUT + "steady_state_sweep_0926.npz", f=f_sim, s21=s21_sim, P_chip=P_CHIP, I_bias=0.0, fr=fr)
log(f"steady-state sweep: fr {fr/1e6:.4f} MHz, min |S21| {np.abs(s21_sim).min():.4f}")

# power grid subset: 3 fluxes x 4 powers x 351 points, with refined_fit fr (power_simulation)
I_list = [x * loader.Phi0 / Mfr_nominal for x in (0.0, 0.24, 0.5)]
P_list = np.array([-110.0, -90.0, -80.0, -70.0])
fr_grid = np.zeros((3, len(P_list)))
s21_grid_pw = np.zeros((3, len(P_list), 351), complex)
f_grid_pw = np.zeros((3, 351))
for j, I in enumerate(I_list):
    res.squids[0].flux_lines["FluxRamp"].I = I
    fr_lin = res.get_omega_r(0.0) / (2 * np.pi)
    f_s = np.linspace(fr_lin - 2e6, fr_lin + 1.5e6, 351)
    f_grid_pw[j] = f_s
    for i, P in enumerate(P_list):
        s = s21_cmt(umux, kappa, f_s, P, I)
        s21_grid_pw[j, i] = s
        fr_grid[j, i] = rf.refined_fit(f_s, s, tau=0)[2]
res.squids[0].flux_lines["FluxRamp"].I = 0.0
np.savez(OUT + "power_grid_0926.npz", I=np.array(I_list), P=P_list, f=f_grid_pw, s21=s21_grid_pw, fr_fit=fr_grid)
log("power grid subset done")

# ----------------------------------------------------------------------------------------------
# D. Static (quasi-static) S21 grid, 52_IQ cell 14
# ----------------------------------------------------------------------------------------------
F_PROBE = 5515.8e6
ramp_line = res.squids[0].flux_lines["FluxRamp"]
Phi0, Mfr = loader.Phi0, loader.Mfrs[0]
ki1, kc1 = ki[0], kc[0]
alpha1 = umux.get_alpha_vec()[0]
a_in = np.sqrt(10 ** ((P_CHIP - 30) / 10))
phi_grid = np.linspace(-0.5, 2.5, 3001)
s21_grid = np.empty(len(phi_grid), complex)
wp = 2 * np.pi * F_PROBE
for i, phi in enumerate(phi_grid):
    ramp_line.I = phi * Phi0 / Mfr
    Er = 0.0
    for _ in range(30):
        a = np.sqrt(kc1 / 2) * a_in / (1j * (wp - res.get_omega_r(Er)) + kappa / 2)
        Er = abs(a) ** 2
    s21_grid[i] = 1 - np.exp(1j * alpha1) / np.cos(alpha1) * np.sqrt(kc1 / 2) * a / a_in
ramp_line.I = 0.0
np.savez(OUT + "static_grid_0926.npz", phi=phi_grid, s21=s21_grid, f_probe=F_PROBE, P_chip=P_CHIP)
log("static grid done")

# ----------------------------------------------------------------------------------------------
# E. Flux-line transfer function: notebook apply_tf (ramp form and pulse form) and H(f)
# ----------------------------------------------------------------------------------------------
poles = np.load("data/flux_line_poles_2026-10-02.npz")["poles_MHz"] * 1e6


def H(f):
    return np.prod(1 / (1 - 1j * np.asarray(f)[..., None] / poles), axis=-1)


def ideal_ramp(rate, amp=RAMP_AMP):
    return lambda t: amp * ((t * rate) % 1)


def apply_tf_ramp(source, rate, dt=0.25e-9):
    n = int(2 ** np.ceil(np.log2(1 / (rate * dt))))
    u = (np.arange(n) + 0.5) / n
    f = np.fft.rfftfreq(n, d=1 / (n * rate))
    coil = np.fft.irfft(np.fft.rfft(source(u / rate)) * H(f), n)
    u, coil = np.r_[u[-1] - 1, u, u[0] + 1], np.r_[coil[-1], coil, coil[0]]
    return lambda t: np.interp((t * rate) % 1, u, coil)


T_PULSE = 40e-6


def ideal_pulse(phi_low, phi_high, period=T_PULSE):
    return lambda t: np.where((t / period) % 1 < 0.5, phi_high, phi_low)


def apply_tf_pulse(source, period=T_PULSE, dt=0.25e-9):
    n = int(2 ** np.ceil(np.log2(period / dt)))
    u = (np.arange(n) + 0.5) / n
    f = np.fft.rfftfreq(n, d=period / n)
    coil = np.fft.irfft(np.fft.rfft(source(u * period)) * H(f), n)
    u, coil = np.r_[u[-1] - 1, u, u[0] + 1], np.r_[coil[-1], coil, coil[0]]
    return lambda t: np.interp((t / period) % 1, u, coil)


f_h = np.logspace(3, 8, 501)
t_r = np.linspace(-0.25, 0.75, 4000) / 100e3
coil_r = apply_tf_ramp(ideal_ramp(100e3), 100e3)(t_r)
t_r5 = np.linspace(-0.25, 0.75, 4000) / 500e3
coil_r5 = apply_tf_ramp(ideal_ramp(500e3), 500e3)(t_r5)
t_p = np.linspace(-0.5e-6, 3e-6, 4000)
coil_p = apply_tf_pulse(ideal_pulse(0.0, 0.25))(t_p)
coil_p_fall = apply_tf_pulse(ideal_pulse(0.0, 0.25))(t_p + T_PULSE / 2)
np.savez(OUT + "flux_line_tf.npz", poles_hz=poles, f=f_h, H=H(f_h), t_ramp100k=t_r, coil_ramp100k=coil_r,
         t_ramp500k=t_r5, coil_ramp500k=coil_r5, t_pulse=t_p, coil_pulse_rise=coil_p, coil_pulse_fall=coil_p_fall,
         ramp_amp=RAMP_AMP, T_pulse=T_PULSE)
log("flux-line TF done")

# AAA poles from the s2p, exactly as 44_ramp/transfer_function
from scipy.interpolate import AAA  # noqa: E402

s2p = DATA + "2026-10-02/fridge_bypass_m20dBm.s2p"
d = np.array([l.split() for l in open(s2p) if l.strip() and l[0] not in "!#"], float)
f2 = d[:, 0]
S11, S21, S12, S22 = (10 ** (d[:, i] / 20) * np.exp(1j * np.deg2rad(d[:, i + 1])) for i in (1, 3, 5, 7))
Sdd = (S11 - S12 - S21 + S22) / 2
Sdc = (S11 + S12 - S21 - S22) / 2
Zdd = 100 * (1 + Sdd) / (1 - Sdd)
R_l = 1983.0 - 12.0
m = (f2 >= 80e3) & (f2 <= 40e6)
s = np.r_[1j * f2[m], -1j * f2[m]] / 1e6
Zop = np.r_[Zdd[m], np.conj(Zdd[m])] + R_l
r = AAA(s, Zop, rtol=1e-2)
roots = r.roots()
poles_aaa = np.array(sorted([z for z in roots if z.real < 0 and z.imag > 0 and np.min(np.abs(roots - np.conj(z))) < 0.02 * abs(z)],
                            key=lambda z: z.imag))
np.savez(OUT + "flux_line_aaa.npz", f=f2, Sdd=Sdd, Sdc=Sdc, Zdd=Zdd, poles_MHz=poles_aaa, R_load=R_l,
         n_terms=len(r.poles()), max_rel_err=np.max(np.abs(r(s) - Zop) / np.abs(Zop)))
log(f"AAA poles (MHz): {poles_aaa}")

# ----------------------------------------------------------------------------------------------
# F. Ramp simulation (52_IQ simulate) at 100 kHz, 500 kHz, and 1 kHz (TF only)
# ----------------------------------------------------------------------------------------------
a_scale = np.sqrt(kc1 / 2) * a_in / (kappa / 2)


def simulate_ramp(rate, use_tf=True, phi_reset=0.0, dt_out=10e-9, settle=5e-6):
    T = 1 / rate
    n_out = max(2000, int(np.ceil(T / dt_out)))
    ramp = apply_tf_ramp(ideal_ramp(rate), rate) if use_tf else ideal_ramp(rate)
    flux = lambda t: phi_reset + ramp(t)  # noqa: E731
    delta, _, _ = umux.compile_static_parameters(2 * np.pi * F_PROBE, flux_ramp_current_func=lambda t: flux(t) * Phi0 / Mfr)
    t = np.arange(n_out) / n_out * T
    sol = Solver(np.r_[-settle, t], delta, ki, kc, a_in, alpha1)
    sol.a0 = np.array([np.sqrt(kc1 / 2) * a_in / (1j * delta[0](-settle, 0.0) + kappa / 2)])
    sol.run_solver(method="BDF", rtol=1e-6, atol=1e-6 * a_scale, max_step=min(20e-9, T / 500))
    return t, sol.get_a_out()[1:] / a_in


ramp_ref = {}
for rate in (100e3, 500e3):
    for use_tf in (False, True):
        t, s21 = simulate_ramp(rate, use_tf, PHI_RESET)
        ramp_ref[f"t_{int(rate/1e3)}k_{'tf' if use_tf else 'ideal'}"] = t
        ramp_ref[f"s21_{int(rate/1e3)}k_{'tf' if use_tf else 'ideal'}"] = s21
        log(f"ramp {rate/1e3:g} kHz tf={use_tf} done")
t1, s1 = simulate_ramp(1e3, True, PHI_RESET)
ramp_ref["t_1k_tf"] = t1
ramp_ref["s21_1k_tf"] = s1
ramp_ref["phi_reset"] = PHI_RESET
ramp_ref["f_probe"] = F_PROBE
ramp_ref["P_chip"] = P_CHIP
np.savez(OUT + "ramp_sims_0926.npz", **ramp_ref)
log("ramp sims done")

# ----------------------------------------------------------------------------------------------
# G. Pulse simulation (53_pulse simulate) on the 10/04 config, with source-level flux calibration
# ----------------------------------------------------------------------------------------------
SOURCE_DIR = DATA + "2026-10-04/good/bare_source_output/"
R_LOOP, R_SRC = 1983.0, 50.0
F_PROBE_P = 5515.030e6
P_CHIP_P = EXTRACT_VNA_DBM + ATTEN


def source_levels(amp):
    x = np.load(glob.glob(SOURCE_DIR + f"*0p5phi0offset_{str(amp).replace('.', 'p')}phi0amp*.npz")[0])["Q"].astype(float)
    mid = (x.max() + x.min()) / 2
    return np.median(x[x < mid]), np.median(x[x >= mid])


SRC_AMPS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
lo, hi = np.array([source_levels(a) for a in SRC_AMPS]).T
v_zero = np.median(np.load(glob.glob(SOURCE_DIR + "*_0p0phi0amp*.npz")[0])["Q"])
v_amp = np.polyfit(SRC_AMPS, hi - lo, 1)[0]
v_off = (lo.mean() - v_zero) / 0.5
Mfr_oct = loader_oct.Mfrs[0]
phi_per_V = Mfr_oct / loader_oct.Phi0 / (R_LOOP + R_SRC)
KA, KO, B0 = v_amp * phi_per_V, v_off * phi_per_V, v_zero * phi_per_V
log(f"source levels: v_amp {v_amp:.6f}, v_off {v_off:.6f}, v_zero {v_zero:.6f}; ka {KA:.6f} ko {KO:.6f} b {B0:.6f}")


def level_flux(off, amp):
    return KO * off + B0 + KA * amp


_, ki_o, kc_o = umux_oct.compile_static_parameters(0.0)
kappa_o = (ki_o + kc_o)[0]
alpha_o = umux_oct.get_alpha_vec()[0]
a_in_o = np.sqrt(10 ** ((P_CHIP_P - 30) / 10))
a_scale_o = np.sqrt(kc_o[0] / 2) * a_in_o / (kappa_o / 2)


def simulate_pulse(off, amp, use_tf=True, dt_out=5e-9, settle=5e-6):
    source = ideal_pulse(level_flux(off, 0.0), level_flux(off, amp))
    flux = apply_tf_pulse(source) if use_tf else source
    delta, _, _ = umux_oct.compile_static_parameters(2 * np.pi * F_PROBE_P, flux_ramp_current_func=lambda t: flux(t) * loader_oct.Phi0 / Mfr_oct)
    t = np.arange(int(round(T_PULSE / dt_out))) * dt_out
    sol = Solver(np.r_[-settle, t], delta, ki_o, kc_o, a_in_o, alpha_o)
    sol.a0 = np.array([np.sqrt(kc_o[0] / 2) * a_in_o / (1j * delta[0](-settle, 0.0) + kappa_o / 2)])
    sol.run_solver(method="BDF", rtol=1e-6, atol=1e-6 * a_scale_o, max_step=20e-9)
    return t, sol.get_a_out()[1:] / a_in_o


pulse_ref = dict(ka=KA, ko=KO, b=B0, v_amp=v_amp, v_off=v_off, v_zero=v_zero, lo=lo, hi=hi, src_amps=np.array(SRC_AMPS),
                 f_probe=F_PROBE_P, P_chip=P_CHIP_P, T=T_PULSE)
for (o, a) in [(0.5, 0.25), (1.0, 0.1)]:
    for use_tf in (False, True):
        t, s21 = simulate_pulse(o, a, use_tf)
        key = f"{str(o).replace('.', 'p')}_{str(a).replace('.', 'p')}_{'tf' if use_tf else 'ideal'}"
        pulse_ref["t_" + key] = t
        pulse_ref["s21_" + key] = s21
        log(f"pulse ({o}, {a}) tf={use_tf} done")
np.savez(OUT + "pulse_sims_1004.npz", **pulse_ref)

# static grid for the pulse notebook (10/04 config, probe 5515.030 MHz)
fl_o = res_oct.squids[0].flux_lines["FluxRamp"]
s21_grid_o = np.empty(len(phi_grid), complex)
wp_o = 2 * np.pi * F_PROBE_P
for i, phi in enumerate(phi_grid):
    fl_o.I = phi * loader_oct.Phi0 / Mfr_oct
    Er = 0.0
    for _ in range(30):
        a = np.sqrt(kc_o[0] / 2) * a_in_o / (1j * (wp_o - res_oct.get_omega_r(Er)) + kappa_o / 2)
        Er = abs(a) ** 2
    s21_grid_o[i] = 1 - np.exp(1j * alpha_o) / np.cos(alpha_o) * np.sqrt(kc_o[0] / 2) * a / a_in_o
fl_o.I = 0.0
np.savez(OUT + "static_grid_1004.npz", phi=phi_grid, s21=s21_grid_o, f_probe=F_PROBE_P, P_chip=P_CHIP_P)
log("pulse static grid done")

# ----------------------------------------------------------------------------------------------
# H. Fitting: fit_trace (res2_params) on 09/26 at k and on 10/04; fit_flux; refined_fit raw; comb
# ----------------------------------------------------------------------------------------------
Ls = 24.2e-12


def notch(f, fr, Qr, Qc, alpha):
    return 1 - np.exp(1j * alpha) * (Qr / Qc) / (1 + 2j * Qr * (f - fr) / fr)


def fit_trace(f, s):
    edge = (f < f[0] + 1.2e6) | (f > f[-1] - 1.2e6)
    f_dip = f[np.argmin(np.abs(s))]
    shape, tau = np.ones_like(s), np.nan
    for it in range(8):
        band = edge & (np.abs(f - f_dip) >= 1.5e6) if it == 0 else edge
        m, b = np.polyfit(f[band], np.unwrap(np.angle(s / shape))[band], 1)
        sc = s * np.exp(-1j * (m * f + b))
        w = np.abs(f - f[np.argmin(np.abs(sc))]) < 1.5e6
        _, _, fr, Qr, Qc, alpha, _ = rf.refined_fit(f[w], sc[w], tau=0, plot_mode='none')
        if np.isnan(fr) or abs(-m / (2 * np.pi) - tau) < 5e-12:
            break
        shape, tau = notch(f, fr, Qr, Qc, alpha), -m / (2 * np.pi)
    return fr, Qr, Qc, alpha


def fr_model(q, I):
    return lf.fr_of_I(I, *q, Ls=Ls)


import scipy.optimize as op  # noqa: E402


def fit_flux(I, fr):
    tries = []
    for seed in range(5):
        np.random.seed(seed)
        tries.append(lf.fit_lambda(I, fr, Ls=Ls))
    q0 = min(tries, key=lambda q: np.sum((fr - fr_model(q, I)) ** 2))
    scale = np.array([1e-6, 1e-12, 1e-12, 1e6, 1.0])
    r = op.least_squares(lambda x: (fr_model(x * scale, I) - fr) / 1e3, q0 / scale, x_scale='jac')
    cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun ** 2) / (len(fr) - 5)
    return r.x * scale, np.sqrt(np.diag(cov)) * scale, np.array(tries)


with np.load(DATA + "2026-09-26/vna/VNA_Flux_Power_Sweep_res_120260926_133948.npz") as dd:
    s21_data, power_data, flux_data, freq_data = dd['s21_data'], dd['vna_powers'], dd['flux_bias'], dd['frequencies']
k = int(np.argmin(np.abs(power_data - EXTRACT_VNA_DBM)))
circ = np.array([fit_trace(freq_data, s) for s in s21_data[k]])
I = flux_data * lf.Phi0 / Mfr_nominal
q, e, tries = fit_flux(I, circ[:, 0])
# raw refined_fit outputs on three delay-corrected windows (the inner call), for a direct notch test
raw = []
for s in s21_data[k, [0, 25, 50]]:
    edge = (freq_data < freq_data[0] + 1.2e6) | (freq_data > freq_data[-1] - 1.2e6)
    m, b = np.polyfit(freq_data[edge], np.unwrap(np.angle(s))[edge], 1)
    sc = s * np.exp(-1j * (m * freq_data + b))
    w = np.abs(freq_data - freq_data[np.argmin(np.abs(sc))]) < 1.5e6
    raw.append(rf.refined_fit(freq_data[w], sc[w], tau=0, plot_mode='none'))
np.savez(OUT + "fits_0926.npz", k=k, fr=circ[:, 0], Qr=circ[:, 1], Qc_abs=circ[:, 2], alpha=circ[:, 3], I=I,
         q=q, q_err=e, tries=tries, raw_refined=np.array(raw), raw_idx=np.array([0, 25, 50]),
         trapped_current=lf.relative_I_trapped(q[0], q[1], q[4]), I_bare=lf.I_bare(q[1], q[4]))
log(f"fits 09/26: Qr med {np.median(circ[:,1]):.3f}, q = {q}")

with np.load(DATA + "2026-10-04/VNA_Flux_Power_Sweep_umux2_res2_20261004_154516.npz") as dd:
    s21_oct, flux_oct, freq_oct = dd['s21_data'][0], dd['flux_bias'], dd['frequencies']
circ_o = np.array([fit_trace(freq_oct, s) for s in s21_oct])
I_oct = flux_oct * lf.Phi0 / Mfr_nominal
q_o, e_o, tries_o = fit_flux(I_oct, circ_o[:, 0])
np.savez(OUT + "fits_1004.npz", fr=circ_o[:, 0], Qr=circ_o[:, 1], Qc_abs=circ_o[:, 2], alpha=circ_o[:, 3], I=I_oct,
         q=q_o, q_err=e_o, tries=tries_o, trapped_current=lf.relative_I_trapped(q_o[0], q_o[1], q_o[4]))
log(f"fits 10/04: q = {q_o}")

# comb: fit_all_resonators (resonator_fitting version) at flux index 0 of the 09/26 full sweep
with np.load(DATA + "2026-09-26/vna/VNA_Flux_Power_Sweep_20260926_130343.npz") as dd:
    f_comb, S_comb = dd["frequencies"], dd["s21_data"][0]
tau_comb = rf.calculate_tau(f_comb, S_comb[0])
peaks = ar.find_all_peaks(f_comb, S_comb[0], dist=1e6, prominence=10.0)
comb = ar.fit_all_resonators(f_comb, S_comb[0], tau_comb, dist=1e6, width=3e6)
np.savez(OUT + "comb_0926.npz", tau=tau_comb, peaks=peaks, aas=comb[0], taus=comb[1], frs=comb[2], Qrs=comb[3], Qcs=comb[4],
         Qis=comb[5], bws=comb[6], alphas=comb[7], chi2s=comb[8])
log(f"comb: {len(peaks)} peaks, {len(comb[2])} fits")

# ----------------------------------------------------------------------------------------------
# I. IQ record folding (52_IQ cell 11) and noise floor (cell 25)
# ----------------------------------------------------------------------------------------------
IQ_DIR = DATA + "2026-09-26/iq/"


def fold(z, period, first=0.0, max_bins=2000):
    u = ((np.arange(len(z)) - first) / period) % 1
    nb = int(min(max_bins, period))
    kk = (u * nb).astype(int)
    n = np.bincount(kk, minlength=nb)
    return np.bincount(kk, u, nb) / n, (np.bincount(kk, z.real, nb) + 1j * np.bincount(kk, z.imag, nb)) / n


iq_ref = {}
for path in sorted(glob.glob(IQ_DIR + "*.npz")):
    dd = np.load(path)
    rate, fs = float(dd["f"]), float(dd["fs"])
    z = dd["I"] + 1j * dd["Q"]
    if rate == 1e3:
        n = int(fs / rate)
        jump = np.abs(np.diff(np.convolve(z, np.ones(8) / 8, "valid")))
        resets = [p * n + 4 + np.argmax(jump[p * n:(p + 1) * n]) for p in range(len(z) // n)]
        period, first = np.polyfit(np.arange(len(resets)), resets, 1)
        u, zf = fold(z, period, first)
        iq_ref["period_1k"] = period
        iq_ref["first_1k"] = first
    else:
        u, zf = fold(z, fs / rate)
    tag = f"{int(rate/1e3)}k"
    iq_ref["u_" + tag] = u
    iq_ref["z_" + tag] = zf
# noise floor for 10 kHz
dd = np.load(glob.glob(IQ_DIR + "*_f_10_kHz.npz")[0])
z, period = dd["I"] + 1j * dd["Q"], dd["fs"] / 10e3
i = np.arange(len(z))
nb = int(min(2000, period))
kk = ((i / period) % 1 * nb).astype(int)
even = (i // period) % 2 == 0
half = [(np.bincount(kk[mm], z.real[mm], nb) + 1j * np.bincount(kk[mm], z.imag[mm], nb)) / np.bincount(kk[mm], minlength=nb)
        for mm in (even, ~even)]
iq_ref["noise_10k_rms_half_diff"] = np.sqrt(np.mean(np.abs(half[0] - half[1]) ** 2)) / 2
np.savez(OUT + "iq_folds_0926.npz", **iq_ref)
log("IQ folds done")

# ----------------------------------------------------------------------------------------------
# J. Pulse folding and alignment (53_pulse cell 8) for three records
# ----------------------------------------------------------------------------------------------
PULSE_DIR = DATA + "2026-10-04/good/IQ_pulse_data/"
FS = 40e6
NBIN = int(round(T_PULSE * FS))
PLATEAU = 10e-6
N_PLAT = int(round(PLATEAU * FS))


def load_pulse(off, amp):
    name = f"IQ_{str(off).replace('.', 'p')}phi0_offset_{str(amp).replace('.', 'p')}phi0amp_*.npz"
    dd = np.load(glob.glob(PULSE_DIR + name)[0])
    return dd["I"].astype(float) + 1j * dd["Q"].astype(float)


def period_est(z, P0=NBIN, K=20):
    N = len(z) // 4
    w1 = np.exp(-2j * np.pi * np.arange(N) / P0)
    w, num, den = np.ones(N), 0.0, 0.0
    for kk in range(1, K + 1):
        w = w * w1
        X1, X2 = (z[:N] * w).sum(), (z[-N:] * w).sum()
        num += kk * abs(X1) ** 2 * np.angle(X2 / X1)
        den += (kk * abs(X1)) ** 2
    return 1 / (1 / P0 + num / den / (2 * np.pi * 3 * N))


def fold_p(z, P):
    kk = np.minimum(((np.arange(len(z)) / P) % 1 * NBIN).astype(int), NBIN - 1)
    return (np.bincount(kk, z.real, NBIN) + 1j * np.bincount(kk, z.imag, NBIN)) / np.bincount(kk, minlength=NBIN)


def departure(zf, kk):
    base = zf[(kk - 400 + np.arange(200)) % NBIN]
    p, sig = base.mean(), np.sqrt(np.mean(np.abs(base - base.mean()) ** 2))
    step = abs(zf[(kk + 500 + np.arange(200)) % NBIN].mean() - p)
    scan = (kk - 200 + np.arange(400)) % NBIN
    return (scan[np.argmax(np.abs(zf[scan] - p) > max(6 * sig, 0.005 * step))] - 1) % NBIN


def align(zf, z_static):
    h = NBIN // 2
    s = np.cumsum(np.r_[0, zf, zf])
    kk = np.argmax(np.abs(2 * (s[h:h + NBIN] - s[:NBIN]) - zf.sum()))
    ka, kb = departure(zf, kk), departure(zf, kk + h)
    ka = (ka + min((kb - ka) % NBIN - h, 0)) % NBIN
    za, zb = np.roll(zf, -ka), np.roll(zf, -(ka + h))
    far = lambda z: abs(z[h - N_PLAT:h].mean() - z_static)  # noqa: E731
    return za if far(za) >= far(zb) else zb


def plateaus(zf):
    return zf[NBIN // 2 - N_PLAT:NBIN // 2].mean(), zf[NBIN - N_PLAT:].mean()


pref = {}
static_05 = load_pulse(0.5, 0.0).mean()
static_10 = load_pulse(1.0, 0.0).mean()
pref["static_0p5"] = static_05
pref["static_1p0"] = static_10
for (o, a), zs in [((0.5, 0.25), static_05), ((1.0, 0.1), static_10)]:
    z = load_pulse(o, a)
    P = period_est(z)
    zf = align(fold_p(z, P), zs)
    key = f"{str(o).replace('.', 'p')}_{str(a).replace('.', 'p')}"
    pref["P_" + key] = P
    pref["z_" + key] = zf
    pref["plateaus_" + key] = np.array(plateaus(zf))
np.savez(OUT + "pulse_folds_1004.npz", **pref)
log("pulse folds done")
log("ALL DONE")
