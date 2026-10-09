"""Bit-faithfulness of ``umux_cmt.fitting`` against the resonator-fitting code the paper used.

The reference values in ``tests/reference`` were captured from the original code with numpy 2.5.0 and
scipy 1.18.0, the versions these tests assume, by two scripts kept in that folder:

* ``make_reference.py`` wrote ``fits_0926.npz``, ``fits_1004.npz``, ``comb_0926.npz`` and
  ``power_grid_0926.npz`` (with the simulation references): the original ``resonator_fitting`` functions
  (``resonance_fitting``, ``lambdafit``, ``all_resonances``), driven by verbatim copies of the ``notch``,
  ``fit_trace`` and ``fit_flux`` cells of the paper's parameter notebook ``res2_params.ipynb``, at the
  extraction power of the 09/26 sweep, on the 10/04 sweep and at one flux point of the comb.
* ``make_fitting_coverage.py`` wrote ``fitting_coverage.npz``: the same code on all 16 powers of the 09/26
  sweep, both flux quanta on their own, the notebook's ``table_params``, every flux point of the comb, the
  full ``refined_fit`` tuple on the simulated power-grid traces, and the failure paths.

Everything is compared exactly (``rtol = atol = 0``): the notch fits, the seeded differential-evolution fits
and their least-squares polish, the trapped-current conversion, the table and configuration values, the
comb survey, and where the original raised or returned NaNs. Tests marked ``slow`` (about a minute) cover
the whole sweeps; ``pytest -m "not slow"`` skips them.
"""
import contextlib
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest

import umux_cmt as uc
from umux_cmt import fitting
from umux_cmt.fitting import notch

HERE = Path(__file__).parent
REF = HERE / "reference"
ROOT = HERE.parent
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"

exact = dict(rtol=0, atol=0)
M_NOMINAL = 14.4e-12          # the flux axis of the sweeps is in Phi0 at this mutual
LS = 24.2e-12                 # what the paper's fits and the configs assume
TABLE_KEYS = ("Qr", "Qc", "Qi", "alpha", "f0", "lamb", "Mc", "Mfr", "err", "phi_tr")
COMB_FIELDS = (("a", "aas"), ("tau", "taus"), ("fr", "frs"), ("Qr", "Qrs"), ("Qc_abs", "Qcs"), ("Qi_abs", "Qis"),
               ("bw", "bws"), ("alpha", "alphas"), ("chi2", "chi2s"))


def _load(name):
    return np.load(REF / name)


@contextlib.contextmanager
def preserved_global_rng():
    """The seeded fits reseed numpy's global RandomState (as the original did); put it back afterwards."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


@pytest.fixture(autouse=True)
def _restore_global_rng():
    with preserved_global_rng():
        yield


def _trace_fits(f, s21_flux):
    fits = [fitting.fit_resonance_trace(f, s) for s in s21_flux]
    return fits, np.array([t.as_tuple() for t in fits]).T


def _flux_fit(I, fr):
    with preserved_global_rng():
        return fitting.fit_flux_response(I, fr, Ls=LS)


@pytest.fixture(scope="module")
def sweep0926():
    return uc.load_vna_sweep(DATA / "vna" / "res2_flux_power_sweep_2026-09-26.npz")


@pytest.fixture(scope="module")
def sweep1004():
    return uc.load_vna_sweep(DATA / "vna" / "res2_flux_sweep_2026-10-04.npz")


@pytest.fixture(scope="module")
def comb_sweep():
    return uc.load_vna_sweep(DATA / "vna" / "full_comb_flux_sweep_2026-09-26.npz")


@pytest.fixture(scope="module")
def coverage():
    return _load("fitting_coverage.npz")


@pytest.fixture(scope="module")
def fits0926(sweep0926):
    r = _load("fits_0926.npz")
    return r, _trace_fits(sweep0926.f, sweep0926.s21[int(r["k"])])


@pytest.fixture(scope="module")
def fits1004(sweep1004):
    return _load("fits_1004.npz"), _trace_fits(sweep1004.f, sweep1004.s21[0])


@pytest.fixture(scope="module")
def flux_fits(fits0926, fits1004, sweep0926, sweep1004):
    """The flux-response fit of each day at its extraction power, computed once for the tests below."""
    out = {}
    for tag, (r, (_, (fr, *_))), sw in (("0926", fits0926, sweep0926), ("1004", fits1004, sweep1004)):
        out[tag] = _flux_fit(sw.bias_current(M_NOMINAL, Phi0=uc.Phi0), fr)
    return out


@pytest.fixture(scope="module")
def every_power_0926(sweep0926):
    """``fit_resonance_trace`` on all 16 x 101 traces, laid out as the notebook did, ``(power, flux)`` each."""
    par = np.array([[fitting.fit_resonance_trace(sweep0926.f, s).as_tuple() for s in row] for row in sweep0926.s21])
    return np.moveaxis(par, -1, 0)


# --------------------------------------------------------------------------- notch fits

def test_trace_fits_0926(fits0926):
    r, (fits, (fr, Qr, Qc_abs, alpha)) = fits0926
    np.testing.assert_allclose(fr, r["fr"], **exact)
    np.testing.assert_allclose(Qr, r["Qr"], **exact)
    np.testing.assert_allclose(Qc_abs, r["Qc_abs"], **exact)
    np.testing.assert_allclose(alpha, r["alpha"], **exact)
    assert all(t.ok for t in fits)


def test_trace_fits_1004(fits1004):
    r, (fits, (fr, Qr, Qc_abs, alpha)) = fits1004
    np.testing.assert_allclose(fr, r["fr"], **exact)
    np.testing.assert_allclose(Qr, r["Qr"], **exact)
    np.testing.assert_allclose(Qc_abs, r["Qc_abs"], **exact)
    np.testing.assert_allclose(alpha, r["alpha"], **exact)


@pytest.mark.slow
def test_trace_fits_every_power_0926(every_power_0926, coverage):
    """All 16 powers x 101 flux points of the 09/26 sweep."""
    for mine, name in zip(every_power_0926, ("fr_all", "Qr_all", "Qc_abs_all", "alpha_all")):
        np.testing.assert_allclose(mine, coverage[name], **exact, err_msg=name)


def test_refined_fit_tuple(sweep0926):
    """refined_fit on three traces that were delay-corrected with a plain edge-band line."""
    r = _load("fits_0926.npz")
    f, S = sweep0926.f, sweep0926.s21[int(r["k"])]
    edge = (f < f[0] + 1.2e6) | (f > f[-1] - 1.2e6)
    for row, idx in zip(r["raw_refined"], r["raw_idx"]):
        s = S[idx]
        m, b = np.polyfit(f[edge], np.unwrap(np.angle(s))[edge], 1)
        sc = s * np.exp(-1j * (m * f + b))
        w = np.abs(f - f[np.argmin(np.abs(sc))]) < 1.5e6
        out = np.array(fitting.refined_fit(f[w], sc[w], 0), float)
        np.testing.assert_allclose(out, row, **exact)
        # the same fit through the NotchFit wrapper, and its model reproduces the fitted baseline
        nf = fitting.fit_resonance(f[w], sc[w], 0.0)
        fields = (nf.a, nf.tau, nf.fr, nf.Qr, nf.Qc_abs, nf.alpha, nf.chi2)
        np.testing.assert_allclose(np.array(fields, float), row, **exact)
        assert nf.as_tuple() == (nf.fr, nf.Qr, nf.Qc_abs, nf.alpha)
        assert abs(np.abs(nf.model(f[w])[0]) - nf.a) < 1e-3


def test_initial_fit_returns_the_original_triple(sweep0926):
    """initial_fit returns (zc, z_t, r) as the original did; the refined circle only on request."""
    t = fitting.fit_resonance_trace(sweep0926.f, sweep0926.s21[7, 0])
    f, s = sweep0926.f[t.window], t.s21_corrected[t.window]
    zc, z_t, r = fitting.initial_fit(f, s, 0)
    zc2, z_t2, r2, circle_t = fitting.initial_fit(f, s, 0, full_output=True)
    assert zc == zc2 and r == r2 and np.array_equal(z_t, z_t2) and len(circle_t) == 3
    st = fitting.fit_resonance_stages(f, s)
    assert st.zc == zc and st.r == r and np.array_equal(st.z_t, z_t)
    assert st.fit.as_tuple() == t.as_tuple()


def test_trace_fit_holds_the_window_fit(fits0926, sweep0926):
    """The TraceFit carries the delay-corrected trace and the window the notch was fitted in."""
    r, (fits, _) = fits0926
    f = sweep0926.f
    t = fits[0]
    assert t.window.sum() > 100 and t.s21_corrected.shape == f.shape
    a, tau, fr, Qr, Qc, alpha, chi2 = fitting.refined_fit(f[t.window], t.s21_corrected[t.window], 0)
    assert (fr, Qr, Qc, alpha) == t.as_tuple()
    assert (a, chi2) == (t.a, t.chi2)
    assert t.converged and 1 <= t.n_iter <= 8
    assert 7e-8 < t.tau < 9e-8          # the ~83 ns cable delay of the setup


def test_power_grid_refined_fit():
    """refined_fit on the twelve simulated CMT traces of the power-grid reference: the full tuple."""
    pg, ref = _load("power_grid_0926.npz"), _load("fitting_coverage.npz")
    out = np.array([[fitting.refined_fit(pg["f"][j], s, tau=0) for s in pg["s21"][j]] for j in range(len(pg["f"]))])
    np.testing.assert_allclose(out, ref["power_grid_refined"], **exact)
    np.testing.assert_allclose(out[:, :, 2], pg["fr_fit"], **exact)


# --------------------------------------------------------------------------- failure paths

def test_errors_propagate_as_in_the_original(sweep0926):
    """Where the original refined_fit raised, the port raises the same exception (no NaNs)."""
    ref = _load("fitting_coverage.npz")
    for name, f, s, exc, msg in zip(ref["fail_names"], ref["fail_f"], ref["fail_s21"], ref["fail_exception"],
                                    ref["fail_message"]):
        assert exc == "ValueError", name
        for fit in (fitting.refined_fit, fitting.fit_resonance, fitting.fit_resonance_stages):
            with pytest.raises(ValueError) as info:
                fit(f, s, 0.0)
            assert type(info.value).__name__ == str(exc) and str(info.value) == str(msg), (name, fit.__name__)
    # the parameter notebook's fit_trace had no error handling: a trace with more than 6 dB of gain raises
    with pytest.raises(ValueError, match="outside of provided bounds"):
        fitting.fit_resonance_trace(sweep0926.f, 100 * sweep0926.s21[7, 0])


def _failing_curve_fit(maxfev_failing):
    real = notch.curve_fit

    def wrapper(*args, **kwargs):
        if kwargs.get("maxfev") == maxfev_failing:
            raise RuntimeError("Optimal parameters not found (forced)")
        return real(*args, **kwargs)
    return wrapper


@pytest.mark.parametrize("stage,maxfev", [("phase", 10000), ("full", 5000)])
def test_nan_path_as_in_the_original(stage, maxfev, monkeypatch, sweep0926):
    """A phase fit or full fit that does not converge gives NaNs, as the original's did."""
    ref = _load("fitting_coverage.npz")
    f, s = ref["fail_f"][0], ref["fail_s21"][0] / 3          # the 'gain_x3' window at unit gain
    monkeypatch.setattr(notch, "curve_fit", _failing_curve_fit(maxfev))
    with pytest.warns(RuntimeWarning, match="returning NaNs"):
        out = np.array(fitting.refined_fit(f, s, 0), float)
    np.testing.assert_array_equal(out, ref[f"nan_refined_{stage}"])
    assert np.isnan(out).all()
    with pytest.warns(RuntimeWarning, match="returning NaNs"):
        nf = fitting.fit_resonance(f, s)
    assert not nf.ok and np.isnan([nf.a, nf.tau, nf.Qr, nf.chi2, nf.bg_phase]).all()
    with pytest.raises(fitting.NotchFitError):
        fitting.fit_resonance_stages(f, s)
    with pytest.warns(RuntimeWarning, match="delay iteration 0"):
        t = fitting.fit_resonance_trace(sweep0926.f, sweep0926.s21[int(ref["k"]), 0])
    np.testing.assert_array_equal(np.array(t.as_tuple(), float), ref[f"nan_trace_{stage}"])
    assert not t.ok and t.n_iter == 1 and not t.converged


# --------------------------------------------------------------------------- the comb

def test_calculate_tau_comb(comb_sweep):
    r = _load("comb_0926.npz")
    assert fitting.calculate_tau(comb_sweep.f, comb_sweep.s21[0, 0]) == float(r["tau"])


def test_comb_fit(comb_sweep):
    r = _load("comb_0926.npz")
    f, s21 = comb_sweep.f, comb_sweep.s21[0, 0]
    tau = fitting.calculate_tau(f, s21)
    np.testing.assert_array_equal(fitting.find_resonances(f, s21, dist=1e6), r["peaks"])
    comb = fitting.fit_all_resonances(f, s21, tau, dist=1e6, width=3e6)
    np.testing.assert_array_equal(comb.peaks, r["peaks"])
    for mine, ref in COMB_FIELDS:
        np.testing.assert_allclose(getattr(comb, mine), r[ref], **exact, err_msg=mine)
    assert len(comb) == 32
    # Qi_abs is the survey's |Qc| definition; the DCM value is what NotchFit.Qi and TraceFit.Qi mean
    np.testing.assert_allclose(comb.Qi_dcm, 1 / (1 / comb.Qr - np.cos(comb.alpha) / comb.Qc_abs), rtol=1e-12)


@pytest.mark.slow
def test_comb_every_flux_point(comb_sweep, coverage):
    """calculate_tau, find_resonances and fit_all_resonances at all 101 flux points of the comb sweep."""
    f = comb_sweep.f
    for j, s21 in enumerate(comb_sweep.s21[0]):
        n = int(coverage["comb_n_peaks"][j])
        tau = fitting.calculate_tau(f, s21)
        assert tau == coverage["comb_tau"][j], j
        np.testing.assert_array_equal(fitting.find_resonances(f, s21, dist=1e6), coverage["comb_peaks"][j, :n])
        comb = fitting.fit_all_resonances(f, s21, tau, dist=1e6, width=3e6)
        np.testing.assert_array_equal(comb.peaks, coverage["comb_peaks"][j, :n])
        for mine, ref in COMB_FIELDS:
            np.testing.assert_allclose(getattr(comb, mine), coverage["comb_" + ref][j, :n], **exact,
                                       err_msg=f"{mine}, flux point {j}")


# --------------------------------------------------------------------------- flux-response fits

@pytest.mark.parametrize("tag", ["0926", "1004"])
def test_flux_response_fit(tag, fits0926, fits1004, sweep0926, sweep1004, flux_fits, coverage):
    r, _ = fits0926 if tag == "0926" else fits1004
    sw = sweep0926 if tag == "0926" else sweep1004
    I = sw.bias_current(M_NOMINAL, Phi0=uc.Phi0)
    np.testing.assert_allclose(I, r["I"], **exact)
    ff = flux_fits[tag]
    np.testing.assert_allclose(ff.tries, r["tries"], **exact)
    np.testing.assert_allclose(ff.params, r["q"], **exact)
    np.testing.assert_allclose(ff.errors, r["q_err"], **exact)
    assert ff.trapped_current == float(r["trapped_current"])
    assert np.isfinite(ff.rms) and ff.rms < 5e3
    if tag == "0926":
        assert ff.rms == coverage["rms_all"][int(r["k"])]       # the rms the notebook printed


@pytest.mark.slow
def test_flux_response_fit_every_power_0926(every_power_0926, sweep0926, coverage):
    """The flux-response fit at all 16 powers of the 09/26 sweep, including those where Mc rails."""
    I = sweep0926.bias_current(M_NOMINAL, Phi0=uc.Phi0)
    np.testing.assert_allclose(I, coverage["I"], **exact)
    for k, fr in enumerate(every_power_0926[0]):
        ff = _flux_fit(I, fr)
        np.testing.assert_allclose(ff.tries, coverage["tries_all"][k], **exact, err_msg=f"power {k}")
        np.testing.assert_allclose(ff.params, coverage["q_all"][k], **exact, err_msg=f"power {k}")
        np.testing.assert_allclose(ff.errors, coverage["q_err_all"][k], **exact, err_msg=f"power {k}")
        assert ff.rms == coverage["rms_all"][k], k


@pytest.mark.slow
def test_flux_response_fit_each_flux_quantum(fits0926, sweep0926, coverage):
    """Each flux quantum of the 09/26 sweep fitted on its own at the extraction power (notebook cell 7)."""
    _, (_, (fr, *_)) = fits0926
    I = sweep0926.bias_current(M_NOMINAL, Phi0=uc.Phi0)
    for h, sel in enumerate((sweep0926.flux <= 1, sweep0926.flux >= 1)):
        ff = _flux_fit(I[sel], fr[sel])
        np.testing.assert_allclose(ff.tries, coverage["half_tries"][h], **exact)
        np.testing.assert_allclose(ff.params, coverage["half_q"][h], **exact)
        np.testing.assert_allclose(ff.errors, coverage["half_q_err"][h], **exact)


def test_fit_lambda_seed_mechanism(fits0926):
    """fit_lambda(seed=s) is np.random.seed(s) followed by the unseeded call, and nothing else."""
    r, (_, (fr, *_)) = fits0926
    I = r["I"]
    np.random.seed(3)
    unseeded = fitting.fit_lambda(I, fr, Ls=LS)
    np.testing.assert_allclose(np.array(unseeded), r["tries"][3], **exact)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seeded = fitting.fit_lambda(I, fr, Ls=LS, seed=3)
    np.testing.assert_allclose(np.array(seeded), r["tries"][3], **exact)
    assert not [w for w in caught if "global RandomState" in str(w.message)]


def test_fit_lambda_warns_if_the_seed_has_no_effect(fits0926, monkeypatch):
    """If differential_evolution stops drawing from the global RandomState, the seed is useless: warn."""
    r, (_, (fr, *_)) = fits0926
    real = fitting.flux.op.differential_evolution
    monkeypatch.setattr(fitting.flux.op, "differential_evolution",
                        lambda *args, **kwargs: real(*args, rng=np.random.default_rng(0), **kwargs))
    with pytest.warns(RuntimeWarning, match="global RandomState"):
        fitting.fit_lambda(r["I"], fr, Ls=LS, seed=3)


def test_offset_conventions(fits0926):
    r, _ = fits0926
    I0, Mfr, Mc, f0, lamb = r["q"]
    assert fitting.I_bare(Mfr, lamb) == float(r["I_bare"])
    assert fitting.relative_I_trapped(I0, Mfr, lamb) == float(r["trapped_current"])
    Itr = fitting.relative_I_trapped(I0, Mfr, lamb)
    # the round trip lands on I0 up to a whole number of flux quanta
    d = (fitting.absolute_I0(Itr, Mfr, lamb) - I0) / (uc.Phi0 / Mfr)
    assert abs(d - round(d)) < 1e-9
    # at the phi = pi/2 point the SQUID does not load the resonator: fr = f0
    assert abs(fitting.fr_of_I(fitting.I_bare(Mfr, lamb) - I0, I0, Mfr, Mc, f0, lamb, Ls=LS) - f0) < 1e-3
    phi_tr = fitting.trapped_flux_phase(I0, Mfr, lamb)
    assert phi_tr == 2 * np.pi * Itr * Mfr / uc.Phi0


# --------------------------------------------------------------------------- Table I and the configuration files

@pytest.mark.parametrize("tag", ["0926", "1004"])
def test_summarize_channel_is_table_params(tag, fits0926, fits1004, flux_fits, coverage):
    """summarize_channel gives exactly what the notebook's table_params gave (Table I)."""
    _, (_, (fr, Qr, Qc_abs, alpha)) = fits0926 if tag == "0926" else fits1004
    for flux_fit in (flux_fits[tag], (flux_fits[tag].params, flux_fits[tag].errors)):
        table = fitting.summarize_channel(Qr, Qc_abs, alpha, flux_fit)
        for key in TABLE_KEYS:
            np.testing.assert_array_equal(table[key], coverage[f"table_{tag}_{key}"], err_msg=key)


@pytest.mark.parametrize("tag,cfg", [("0926", "nist_res2_2026-09-26.toml"), ("1004", "nist_res2_2026-10-04.toml")])
def test_config_values_reproduced(tag, cfg, fits0926, fits1004, flux_fits):
    _, (_, (fr, Qr, Qc_abs, alpha)) = fits0926 if tag == "0926" else fits1004
    table = fitting.summarize_channel(Qr, Qc_abs, alpha, flux_fits[tag])
    with open(CONFIGS / cfg, "rb") as fh:
        toml = tomllib.load(fh)
    res, sq = toml["resonators"], toml["squids"]
    assert table["Qr"] == res["Qr"]
    assert table["Qc_abs"] == res["Qc"]                       # the file holds |Qc|
    assert table["alpha"] == res["alpha"]
    assert table["f0"] == res["f0"]
    assert table["lamb"] == sq["lamb"]
    assert table["Mc"] == sq["Mc"]
    assert table["Mfr"] == sq["Mfr"]
    assert table["trapped_current"] == sq["trapped_current"]
    assert sq["Ls"] == LS
    # DCM conversion: the table's Qc and Qi are what the config loader builds the channel with
    dev = uc.load_config(CONFIGS / cfg)
    ch = dev.channels[0]
    assert abs(ch.Qc - table["Qc"]) < 1e-9 * table["Qc"]
    assert abs(ch.Qi - table["Qi"]) < 1e-9 * table["Qi"]
    assert abs(table["phi_tr"] - 2 * np.pi * sq["trapped_current"] * sq["Mfr"] / uc.Phi0) < 1e-15


def test_written_config_round_trip(tmp_path, fits0926, sweep0926, flux_fits):
    """A config written from the pipeline loads to a channel whose fr(I) matches the measured one."""
    _, (_, (fr, Qr, Qc_abs, alpha)) = fits0926
    I = sweep0926.bias_current(M_NOMINAL, Phi0=uc.Phi0)
    t = fitting.summarize_channel(Qr, Qc_abs, alpha, flux_fits["0926"])
    path = tmp_path / "res2.toml"
    uc.write_config(path, f0=t["f0"], Qr=t["Qr"], Qc_abs=t["Qc_abs"], alpha=t["alpha"], lamb=t["lamb"], Mc=t["Mc"],
                    Mfr=t["Mfr"], trapped_current=t["trapped_current"], Ls=LS)
    ch = uc.load_config(path).channels[0]
    fr_cfg = []
    for Ik in I:
        ch.line("FluxRamp").I = Ik
        fr_cfg.append(ch.get_omega_r(0.0) / (2 * np.pi))
    dev = np.array(fr_cfg) - fr
    assert np.sqrt(np.mean(dev ** 2)) < 5e3            # kHz-level residuals, as in the notebook's round trip


# --------------------------------------------------------------------------- packaging

def test_import_does_not_load_matplotlib():
    """The numerical package loads matplotlib only when the plots are used."""
    code = ("import sys, umux_cmt, umux_cmt.fitting\n"
            "assert 'matplotlib' not in sys.modules, 'import umux_cmt loaded matplotlib'\n"
            "from umux_cmt.fitting import plots\n"
            "assert plots.notch_panels and 'matplotlib.pyplot' in sys.modules\n")
    subprocess.run([sys.executable, "-c", code], check=True)
