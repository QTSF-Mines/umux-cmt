"""Whole-sweep fitting references from the ORIGINAL code, written to ``fitting_coverage.npz``.

``make_reference.py`` captured the fitting path at one power of the 09/26 sweep and at one flux point of
the comb. This script extends that to everything the fitting code does on the shipped data, and to its
failure paths, with the same original code (``resonator_fitting/include``) driven by verbatim copies of the
``notch``, ``fit_trace``, ``fit_flux`` and ``table_params`` cells of the paper's parameter notebook
(``umux_modeling/LW_uMUX_src/jupyter/final/43_characterization/res2_params.ipynb``; ``fit_flux`` here also
returns its five differential-evolution solutions):

* the circle fit of all 16 powers x 101 flux points of the 09/26 sweep (cell 3) and the flux-response fit at
  every power (cell 5, with the rms residual it prints), the fit of each flux quantum on its own at the
  extraction power (cell 7), and ``table_params`` for 09/26 and 10/04 (cell 11);
* ``calculate_tau``, ``find_all_peaks`` and ``fit_all_resonators`` at all 101 flux points of the comb sweep;
* the full ``refined_fit`` tuple on the twelve simulated traces of ``power_grid_0926.npz``;
* failure paths: four windows on which the original raises (net gain above 6 dB, a NaN sample, constant
  data, a window too narrow for the phase fit), recorded with the exception it raised, and the NaN results
  of ``refined_fit`` and ``fit_trace`` when the phase fit or the full fit does not converge (forced by
  making ``curve_fit`` raise ``RuntimeError`` for that stage).

Paths, environment and the bytecode precaution are those of ``make_reference.py``::

    PYTHONDONTWRITEBYTECODE=1 UMUX_REFERENCE_OUT=/tmp/ref python tests/reference/make_fitting_coverage.py

It reads the original code and data only, and checks that the overlapping values agree bit for bit with
``fits_0926.npz``, ``fits_1004.npz`` and ``comb_0926.npz``. About 1.5 minutes.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize as op

HERE = Path(__file__).resolve().parent
REPO = os.environ.get("UMUX_ORIGINAL_ROOT", str(HERE.parents[2]))
DATA = REPO + "/NIST_uMux_measurment_data/"
OUT = os.environ.get("UMUX_REFERENCE_OUT", str(HERE)) + "/"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

from resonator_fitting.include import resonance_fitting as rf, lambdafit as lf, all_resonances as ar  # noqa: E402

Ls = 24.2e-12
Mfr_nominal = 14.4e-12
EXTRACT_VNA_DBM = -26.0
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# The notebook cells (res2_params.ipynb cells 3, 5 and 11), verbatim
# ----------------------------------------------------------------------------------------------
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


def fit_flux(I, fr):
    tries = []
    for seed in range(5):
        np.random.seed(seed)
        tries.append(lf.fit_lambda(I, fr, Ls=Ls))
    q0 = min(tries, key=lambda q: np.sum((fr - fr_model(q, I))**2))
    scale = np.array([1e-6, 1e-12, 1e-12, 1e6, 1.0])
    r = op.least_squares(lambda x: (fr_model(x * scale, I) - fr) / 1e3, q0 / scale, x_scale='jac')
    cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun**2) / (len(fr) - 5)
    return r.x * scale, np.sqrt(np.diag(cov)) * scale, np.array(tries)


def table_params(Qr_arr, Qc_abs_arr, alpha_arr, flux_fit):
    alpha = np.median(alpha_arr)
    Qr = np.median(Qr_arr)
    Qc = np.median(Qc_abs_arr) / np.cos(alpha)     # DCM
    (I0, Mfr, Mc, f0, lamb), err = flux_fit
    return dict(Qr=Qr, Qc=Qc, Qi=1 / (1 / Qr - 1 / Qc), alpha=alpha, f0=f0, lamb=lamb, Mc=Mc, Mfr=Mfr, err=err,
                phi_tr=2 * np.pi * lf.relative_I_trapped(I0, Mfr, lamb) * Mfr / lf.Phi0)


TABLE_KEYS = ("Qr", "Qc", "Qi", "alpha", "f0", "lamb", "Mc", "Mfr", "err", "phi_tr")
ref = {}

# ----------------------------------------------------------------------------------------------
# A. 09/26: every power (cells 3, 5, 7, 11)
# ----------------------------------------------------------------------------------------------
with np.load(DATA + "2026-09-26/vna/VNA_Flux_Power_Sweep_res_120260926_133948.npz") as d:
    s21_data = d['s21_data']        # (power, flux, freq)
    power_data = d['vna_powers']
    flux_data = d['flux_bias']
    freq_data = d['frequencies']
I = flux_data * lf.Phi0 / Mfr_nominal

circ = np.array([[fit_trace(freq_data, s) for s in s21_p] for s21_p in s21_data])
fr_all, Qr_all, Qc_abs_all, alpha_all = np.moveaxis(circ, -1, 0)    # each (power, flux)
log(f"09/26 circle fits: {circ.shape[0] * circ.shape[1]} traces, {np.isnan(fr_all).sum()} NaN")

flux_fits = [fit_flux(I, fr) for fr in fr_all]
ref["q_all"] = np.array([ff[0] for ff in flux_fits])
ref["q_err_all"] = np.array([ff[1] for ff in flux_fits])
ref["tries_all"] = np.array([ff[2] for ff in flux_fits])
ref["rms_all"] = np.array([np.std(fr - fr_model(ff[0], I)) for fr, ff in zip(fr_all, flux_fits)])
log("09/26 flux fits at every power done")

k = np.argmin(np.abs(power_data - EXTRACT_VNA_DBM))
halves = [fit_flux(I[sel], fr_all[k, sel]) for sel in (flux_data <= 1, flux_data >= 1)]
ref["half_q"] = np.array([h[0] for h in halves])
ref["half_q_err"] = np.array([h[1] for h in halves])
ref["half_tries"] = np.array([h[2] for h in halves])

sep = table_params(Qr_all[k], Qc_abs_all[k], alpha_all[k], flux_fits[k][:2])
ref.update(k=k, power_data=power_data, fr_all=fr_all, Qr_all=Qr_all, Qc_abs_all=Qc_abs_all, alpha_all=alpha_all, I=I)
ref.update({f"table_0926_{key}": sep[key] for key in TABLE_KEYS})

# ----------------------------------------------------------------------------------------------
# B. 10/04 (cells 3, 5, 11)
# ----------------------------------------------------------------------------------------------
with np.load(DATA + "2026-10-04/VNA_Flux_Power_Sweep_umux2_res2_20261004_154516.npz") as d:
    s21_oct = d['s21_data'][0]      # (flux, freq), one power
    flux_oct = d['flux_bias']
    freq_oct = d['frequencies']
I_oct = flux_oct * lf.Phi0 / Mfr_nominal
fr_oct, Qr_oct, Qc_abs_oct, alpha_oct = np.array([fit_trace(freq_oct, s) for s in s21_oct]).T
flux_fit_oct = fit_flux(I_oct, fr_oct)
octo = table_params(Qr_oct, Qc_abs_oct, alpha_oct, flux_fit_oct[:2])
ref.update({f"table_1004_{key}": octo[key] for key in TABLE_KEYS})
log("10/04 done")

# the overlap with the first capture must agree bit for bit
r0926, r1004 = np.load(HERE / "fits_0926.npz"), np.load(HERE / "fits_1004.npz")
assert k == int(r0926["k"])
for mine, theirs in ((fr_all[k], "fr"), (Qr_all[k], "Qr"), (Qc_abs_all[k], "Qc_abs"), (alpha_all[k], "alpha"),
                     (flux_fits[k][0], "q"), (flux_fits[k][1], "q_err"), (flux_fits[k][2], "tries")):
    assert np.array_equal(mine, r0926[theirs]), theirs
for mine, theirs in ((fr_oct, "fr"), (Qr_oct, "Qr"), (Qc_abs_oct, "Qc_abs"), (alpha_oct, "alpha"),
                     (flux_fit_oct[0], "q"), (flux_fit_oct[1], "q_err"), (flux_fit_oct[2], "tries")):
    assert np.array_equal(mine, r1004[theirs]), theirs

# ----------------------------------------------------------------------------------------------
# C. The comb: every flux point (resonator_fitting.all_resonances, width 3 MHz, delay of each trace)
# ----------------------------------------------------------------------------------------------
with np.load(DATA + "2026-09-26/vna/VNA_Flux_Power_Sweep_20260926_130343.npz") as d:
    f_comb, S_comb = d["frequencies"], d["s21_data"][0]
n_flux = len(S_comb)
combs, taus, peak_lists = [], [], []
for j in range(n_flux):
    taus.append(rf.calculate_tau(f_comb, S_comb[j]))
    peak_lists.append(ar.find_all_peaks(f_comb, S_comb[j], dist=1e6, prominence=10.0))
    combs.append(ar.fit_all_resonators(f_comb, S_comb[j], taus[-1], dist=1e6, width=3e6))
n_peaks = np.array([len(p) for p in peak_lists])
n_max = n_peaks.max()
peaks_arr = np.full((n_flux, n_max), -1, dtype=np.int64)
fits_arr = np.full((9, n_flux, n_max), np.nan)
for j in range(n_flux):
    peaks_arr[j, :n_peaks[j]] = peak_lists[j]
    for i in range(9):
        fits_arr[i, j, :n_peaks[j]] = combs[j][i]
ref.update(comb_tau=np.array(taus), comb_n_peaks=n_peaks, comb_peaks=peaks_arr)
for i, name in enumerate(("aas", "taus", "frs", "Qrs", "Qcs", "Qis", "bws", "alphas", "chi2s")):
    ref["comb_" + name] = fits_arr[i]
r_comb = np.load(HERE / "comb_0926.npz")
assert taus[0] == float(r_comb["tau"]) and np.array_equal(peak_lists[0], r_comb["peaks"])
for i, name in enumerate(("aas", "taus", "frs", "Qrs", "Qcs", "Qis", "bws", "alphas", "chi2s")):
    assert np.array_equal(np.array(combs[0][i]), r_comb[name], equal_nan=True), name
log(f"comb: {n_flux} flux points, {n_peaks.sum()} windows, {np.isnan(fits_arr[2]).sum()} NaN fits")

# ----------------------------------------------------------------------------------------------
# D. refined_fit on the simulated power-grid traces (the full tuple; make_reference.py kept fr only)
# ----------------------------------------------------------------------------------------------
pg = np.load(HERE / "power_grid_0926.npz")
ref["power_grid_refined"] = np.array([[rf.refined_fit(pg["f"][j], s, tau=0) for s in pg["s21"][j]]
                                      for j in range(len(pg["f"]))])
assert np.array_equal(ref["power_grid_refined"][:, :, 2], pg["fr_fit"])

# ----------------------------------------------------------------------------------------------
# E. Failure paths
# ----------------------------------------------------------------------------------------------
f_w = np.linspace(5.514e9, 5.517e9, 301)
base = 0.9 * np.exp(0.3j) * notch(f_w, 5.5155e9, 1.0e4, 1.1e4, 0.05)
nan_trace = base.copy()
nan_trace[100] = np.nan
f_narrow = np.linspace(5.5155e9 - 1.5e3, 5.5155e9 + 1.5e3, 301)
cases = {"gain_x3": (f_w, 3 * base),                                   # baseline |S21| = 2.7 > 2
         "nan_sample": (f_w, nan_trace),
         "constant": (f_w, np.full(301, 0.9 + 0.1j)),
         "narrow_window": (f_narrow, 0.9 * np.exp(0.3j) * notch(f_narrow, 5.5155e9, 1.0e4, 1.1e4, 0.05))}
raised = []
for name, (ff, ss) in cases.items():
    try:
        rf.refined_fit(ff, ss, tau=0)
        raised.append(("none", ""))
    except Exception as e:
        raised.append((type(e).__name__, str(e)))
    log(f"failure case {name}: {raised[-1]}")
ref.update(fail_names=np.array(list(cases)), fail_f=np.array([c[0] for c in cases.values()]),
           fail_s21=np.array([c[1] for c in cases.values()]), fail_exception=np.array([r[0] for r in raised]),
           fail_message=np.array([r[1] for r in raised]))

# the NaN path: make one stage's curve_fit fail to converge (phase fit: maxfev=10000, full fit: maxfev=5000)
real_curve_fit = rf.curve_fit


def failing_curve_fit(maxfev_failing):
    def wrapper(*args, **kwargs):
        if kwargs.get("maxfev") == maxfev_failing:
            raise RuntimeError("Optimal parameters not found (forced)")
        return real_curve_fit(*args, **kwargs)
    return wrapper


for stage, maxfev in (("phase", 10000), ("full", 5000)):
    rf.curve_fit = failing_curve_fit(maxfev)
    try:
        ref[f"nan_refined_{stage}"] = np.array(rf.refined_fit(f_w, base, tau=0), float)
        ref[f"nan_trace_{stage}"] = np.array(fit_trace(freq_data, s21_data[k, 0]), float)
    finally:
        rf.curve_fit = real_curve_fit
    log(f"forced {stage}-fit failure: refined_fit {ref[f'nan_refined_{stage}']}, fit_trace {ref[f'nan_trace_{stage}']}")

np.savez_compressed(OUT + "fitting_coverage.npz", **ref)
log(f"wrote {OUT}fitting_coverage.npz")
