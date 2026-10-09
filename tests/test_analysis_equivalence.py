"""Bit-faithfulness of ``umux_cmt.analysis`` against the final notebooks the paper was produced with.

The reference files were captured by executing the cells of ``52_IQ/IQ_simulation.ipynb`` and
``53_pulse/pulse_simulation.ipynb`` verbatim (the notebooks' ``record()`` replaced by an in-memory
stand-in), with the same numpy; each file carries its provenance:

``iq_folds_0926.npz``
    cell 11: the folds of all 28 ramp records, the 1 kHz reset period, the 10 kHz noise.
``iq_analysis_0926.npz``
    cells 15, 17, 21, 25: ``A``, ``B`` and the residual of the calibration against the full 1 kHz
    model, the noise floor at all 27 rates, the quasi-static alignment at all 27 rates, and the
    periodic interpolant and alignment of the 100 and 500 kHz reference simulations.
``pulse_folds_1004.npz``
    cell 8 step by step for two records.
``pulse_analysis_1004.npz``
    cell 8 for all 31 records and the noise of each (cell 21), the source levels and flux scales
    (cell 12), the 36-level calibration with its held-out residuals (cell 13), ``total_residual``,
    ``fit_shift`` and the edge rms (cell 21) on the two records of ``pulse_sims_1004.npz``, and the
    same on the static models of all 31 records (cell 24).

Inputs that need the original simulator (static grids, simulations, device constants) come from the
other reference files. Every function is checked for exact equality (``rtol=0, atol=0``). The raw
archive is not shipped; the tests that need it are skipped without it, and the shipped reduced files
are checked against the same references instead. The end-to-end checks also reproduce the numbers the
notebooks recorded for the paper (``values.json``) at full double precision.
"""
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import umux_cmt as uc
from umux_cmt.analysis import ramps, pulses as pz
from umux_cmt.io import load_record

HERE = Path(__file__).parent
REF = HERE / "reference"
ROOT = HERE.parent
DATA = ROOT / "data"
RAW = Path(os.environ.get("UMUX_RAW_DATA", ROOT.parent / "NIST_uMux_measurment_data"))
PULSE_DIR = RAW / "2026-10-04/good/IQ_pulse_data"
SOURCE_DIR = RAW / "2026-10-04/good/bare_source_output"
FOLDED = DATA / "pulses_2026-10-04/pulses_folded.npz"
LEVELS = DATA / "pulses_2026-10-04/source_levels.npz"

exact = dict(rtol=0, atol=0)

# fixed inputs of the final notebooks (values.json of the paper)
RAMP_AMP = 0.945856249388244          # Phi0 per ramp period
PHI_RESET = 0.1845                    # ramp-line flux at the reset, from the scan below
MFR_1004 = 1.4400671023359391e-11     # H, flux-ramp mutual of the 10/04 configuration
TWO = [(0.5, 0.25, "0p5_0p25"), (1.0, 0.1, "1p0_0p1")]     # the pulse records with reference simulations
STATIC_SHIFTS = (-0.3e-6, 1.0e-6, 5e-9)                    # the shift grid of 53_pulse cell 24

needs_ramps = pytest.mark.skipif(not (DATA / "flux_ramps_2026-09-26").exists(), reason="demo data not present")
needs_raw_pulses = pytest.mark.skipif(not PULSE_DIR.exists(), reason="raw pulse archive not present")
needs_raw_source = pytest.mark.skipif(not SOURCE_DIR.exists(), reason="raw source archive not present")
needs_folded = pytest.mark.skipif(not (FOLDED.exists() and LEVELS.exists()), reason="prepared pulse data not present")


def _load(name):
    return np.load(REF / name)


def _ramp(tag):
    """Record ``ramp_<tag>Hz.npz`` and its rate (``tag`` like ``'10k'``)."""
    return load_record(DATA / f"flux_ramps_2026-09-26/ramp_{tag}Hz.npz"), int(tag[:-1]) * 1e3


def _tag(rate):
    return f"{int(rate / 1e3)}k"


def _static_model(tag):
    """The quasi-static S21(phi) of the reference, as the notebooks' ``s21_static`` interpolant."""
    r = _load(f"static_grid_{tag}.npz")
    phi, s21 = r["phi"], r["s21"]
    return lambda q: np.interp(q, phi, s21.real) + 1j * np.interp(q, phi, s21.imag)


def _same(a, b):
    """Exact equality, dtype included."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


IQ_RATES = [float(x) for x in _load("iq_analysis_0926.npz")["rates"]]


# ----------------------------------------------------------------------------- flux ramps

@needs_ramps
@pytest.mark.parametrize("tag", [k[2:] for k in _load("iq_folds_0926.npz").files if k.startswith("u_")])
def test_fold_every_ramp_record(tag):
    r = _load("iq_folds_0926.npz")
    rec, rate = _ramp(tag)
    assert rec.rate == rate
    u, z = ramps.fold_record(rec.z, rec.fs, rec.rate, from_resets=(rate == 1e3))
    np.testing.assert_allclose(u, r[f"u_{tag}"], **exact)
    np.testing.assert_allclose(z, r[f"z_{tag}"], **exact)


@needs_ramps
def test_reset_period_of_the_1khz_ramp():
    r = _load("iq_folds_0926.npz")
    rec, _ = _ramp("1k")
    period, first = ramps.reset_period(rec.z, rec.fs, rec.rate)
    assert period == r["period_1k"] and first == r["first_1k"]
    assert abs(rec.fs / period - 999.2) < 0.1          # the ramp runs slightly slow


@needs_ramps
def test_even_odd_noise_of_the_10khz_record():
    r = _load("iq_folds_0926.npz")
    rec, _ = _ramp("10k")
    assert ramps.even_odd_noise(rec.z, rec.fs / rec.rate) == r["noise_10k_rms_half_diff"]


@needs_ramps
@pytest.mark.parametrize("rate", IQ_RATES, ids=_tag)
def test_noise_floor_at_every_rate(rate):
    """52_IQ cell 25 ``noise_floor``: the even/odd noise in units of the 1 kHz excursion."""
    ia = _load("iq_analysis_0926.npz")
    excursion = ramps.rms_excursion(_load("iq_folds_0926.npz")["z_1k"])
    assert excursion == ia["excursion"]
    rec, _ = _ramp(_tag(rate))
    assert ramps.even_odd_noise(rec.z, rec.fs / rec.rate) / excursion == ia["noise_floor"][IQ_RATES.index(rate)]


@needs_ramps
def test_fold_ramp_records_matches_single_folds():
    r = _load("iq_folds_0926.npz")
    folds = ramps.fold_ramp_records(sorted((DATA / "flux_ramps_2026-09-26").glob("ramp_*kHz.npz")))
    assert len(folds) == sum(k.startswith("u_") for k in r.files)
    for rate, (u, z) in folds.items():
        tag = _tag(rate)
        np.testing.assert_allclose(u, r[f"u_{tag}"], **exact)
        np.testing.assert_allclose(z, r[f"z_{tag}"], **exact)


def test_calibration_scan_recovers_the_reset_flux_of_the_paper():
    """52_IQ cell 15: scanning Phi_reset with the quasi-static model, fitting A and B at each trial."""
    s21_static = _static_model("0926")
    r = _load("iq_folds_0926.npz")
    u_1k, z_1k = r["u_1k"], r["z_1k"]
    excursion = ramps.rms_excursion(z_1k)
    assert excursion == np.sqrt(np.mean(np.abs(z_1k - z_1k.mean()) ** 2))

    def fit_AB(phi_reset):                           # the notebook's function, verbatim
        M = np.c_[s21_static(phi_reset + RAMP_AMP * u_1k), np.ones(len(u_1k))]
        (A, B), *_ = np.linalg.lstsq(M, z_1k, rcond=None)
        return A, B, np.sqrt(np.mean(np.abs(z_1k - M @ [A, B]) ** 2)) / excursion

    scan = np.arange(0, 1, 1 / 4000)
    errs = []
    for phi in scan:
        A, B, z_fit = ramps.fit_gain_offset(s21_static(phi + RAMP_AMP * u_1k), z_1k, return_fit=True)
        A0, B0, err0 = fit_AB(phi)
        assert A == A0 and B == B0
        errs.append(ramps.rms_distance(z_1k, z_fit, excursion))
        assert errs[-1] == err0
    assert scan[np.argmin(errs)] == PHI_RESET == _load("iq_analysis_0926.npz")["phi_reset"]


def test_calibration_against_the_full_1khz_model():
    """52_IQ cell 15: A and B fitted against the full model at the reset flux reproduce the paper's IQResCal."""
    ia, rs, r = _load("iq_analysis_0926.npz"), _load("ramp_sims_0926.npz"), _load("iq_folds_0926.npz")
    u_1k, z_1k = r["u_1k"], r["z_1k"]
    model = ramps.periodic_phase_interpolant(rs["t_1k_tf"] * 1e3, rs["s21_1k_tf"])
    A, B, z_fit = ramps.fit_gain_offset(model(u_1k), z_1k, return_fit=True)
    assert A == ia["A"] and B == ia["B"]
    err = ramps.rms_distance(z_1k, z_fit, ramps.rms_excursion(z_1k))
    assert err == ia["err_1k"] == 0.037765363750435446                          # IQResCal


@pytest.mark.parametrize("rate", IQ_RATES, ids=_tag)
def test_quasi_static_alignment_at_every_rate(rate):
    """52_IQ cells 21 and 25: ``align(rate, quasi_static)`` and the residual there, at every rate."""
    ia, r = _load("iq_analysis_0926.npz"), _load("iq_folds_0926.npz")
    s21_static = _static_model("0926")
    quasi_static = lambda u: s21_static(PHI_RESET + RAMP_AMP * (np.asarray(u) % 1))   # noqa: E731  cell 25 model
    i = IQ_RATES.index(rate)
    u_m, z_m = r[f"u_{_tag(rate)}"], r[f"z_{_tag(rate)}"]
    shift, err = ramps.best_phase_shift(z_m, u_m, quasi_static, ia["A"], ia["B"], scale=ia["excursion"])
    assert shift == ia["qs_shift"][i] and err == ia["qs_err"][i]


@pytest.mark.parametrize("tag", ["100k", "500k"])
@pytest.mark.parametrize("model_kind", ["ideal", "tf"])
def test_periodic_model_and_alignment_of_the_reference_simulations(tag, model_kind):
    """52_IQ cells 17 and 21: ``periodic(t * rate, s21)`` and ``align`` with the reference simulations."""
    ia, rs, r = _load("iq_analysis_0926.npz"), _load("ramp_sims_0926.npz"), _load("iq_folds_0926.npz")
    rate = int(tag[:-1]) * 1e3
    model = ramps.periodic_phase_interpolant(rs[f"t_{tag}_{model_kind}"] * rate, rs[f"s21_{tag}_{model_kind}"])
    assert _same(model(ia["probe"]), ia[f"periodic_{tag}_{model_kind}"])
    shift, err = ramps.best_phase_shift(r[f"z_{tag}"], r[f"u_{tag}"], model, ia["A"], ia["B"], scale=ia["excursion"])
    assert shift == ia[f"shift_{tag}_{model_kind}"] and err == ia[f"err_{tag}_{model_kind}"]


def test_best_phase_shift_requires_the_scale_and_the_trace_first():
    u = np.linspace(0, 1, 50, endpoint=False)
    model = ramps.periodic_phase_interpolant(u, np.exp(2j * np.pi * u))
    z = 2 * model(u - 0.25) + 0.1
    with pytest.raises(TypeError):
        ramps.best_phase_shift(z, u, model, 2.0, 0.1)                       # no scale
    with pytest.raises(TypeError, match=r"\(z_meas, u_meas"):
        ramps.best_phase_shift(u, z, model, 2.0, 0.1, scale=1.0)            # arguments in fold order
    shift, err = ramps.best_phase_shift(z, u, model, 2.0, 0.1, scale=1.0, n=100)
    assert shift == 0.25 and err < 1e-12


# ----------------------------------------------------------------------------- square pulses, raw records

@pytest.fixture(scope="module")
def raw_folds():
    return pz.fold_pulse_records(PULSE_DIR)


@needs_raw_pulses
@pytest.mark.parametrize("offset,amp,key", TWO)
def test_fold_and_align_raw_pulse_record(offset, amp, key):
    r = _load("pulse_folds_1004.npz")
    static = load_record(pz._find(PULSE_DIR, pz.pulse_record_name(offset, 0.0))).z.mean()
    assert static == r[f"static_{str(offset).replace('.', 'p')}"]
    z = load_record(pz._find(PULSE_DIR, pz.pulse_record_name(offset, amp))).z
    P = pz.estimate_period(z)
    assert P == r[f"P_{key}"]
    zf = pz.align_to_rising_edge(pz.fold_pulses(z, P), static)
    np.testing.assert_allclose(zf, r[f"z_{key}"], **exact)
    np.testing.assert_allclose(np.array(pz.plateau_levels(zf)), r[f"plateaus_{key}"], **exact)


@needs_raw_pulses
def test_fold_pulse_records_reproduces_every_record(raw_folds):
    """53_pulse cell 8 on all 40 raw records: 31 folds, their periods and noise, 5 static levels."""
    pa = _load("pulse_analysis_1004.npz")
    for key, ref in [("offset", "offset"), ("amp", "amp"), ("period_samples", "P"), ("noise", "noise"),
                     ("static_offset", "static_offset"), ("static_z", "static_z")]:
        assert _same(raw_folds[key], pa[ref]), key
    for i, (o, a) in enumerate(zip(pa["offset"], pa["amp"])):
        assert _same(raw_folds["z"][i], pa["z"][i]), (o, a)


@needs_raw_source
def test_source_levels_and_flux_scales_from_raw_records():
    s, pa = _load("pulse_sims_1004.npz"), _load("pulse_analysis_1004.npz")
    lv = pz.source_record_levels(SOURCE_DIR)
    np.testing.assert_allclose(lv["amps"], s["src_amps"], **exact)
    for k in ("lo", "hi", "v_zero", "v_amp", "v_off"):
        assert _same(lv[k], pa[k]) and _same(lv[k], s[k]), k
    assert lv["v_zero"].dtype == np.float32
    assert lv["excerpt"].shape == (6, 3 * pz.NBIN) and lv["excerpt"].dtype == np.float32
    assert lv["excerpt_zero"].shape == (3 * pz.NBIN,)
    cal = pz.LevelCalibration.from_source(lv, MFR_1004)
    assert (cal.ka, cal.ko, cal.b) == (pa["KA"], pa["KO"], pa["B0"]) == (s["ka"], s["ko"], s["b"])
    # the baseline record, found as 53_pulse cell 12 found it
    assert pz._find(SOURCE_DIR, pz.source_baseline_name()) == Path(glob.glob(str(SOURCE_DIR) + "/*_0p0phi0amp*.npz")[0])
    # the individual pieces, verbatim from 53_pulse cell 12
    x = np.load(pz._find(SOURCE_DIR, pz.source_record_name(0.25)))["Q"].astype(float)
    mid = (x.max() + x.min()) / 2
    assert pz.source_levels(x) == (np.median(x[x < mid]), np.median(x[x >= mid]))
    assert pz.source_scales(lv["lo"], lv["hi"], lv["v_zero"], pz.SRC_AMPS) == (np.polyfit(pz.SRC_AMPS, lv["hi"] - lv["lo"], 1)[0],
                                                                                (lv["lo"].mean() - lv["v_zero"]) / 0.5)


# ----------------------------------------------------------------------------- record lookup

def _touch(path, **arrays):
    np.savez(path, **(arrays or dict(Q=np.zeros(4, np.float32), fs=40e6)))
    return path


def test_record_lookup_follows_glob(tmp_path):
    real = _touch(tmp_path / "Q_only_src_0p5phi0offset_0p25phi0amp20261004_170035.npz")
    (tmp_path / "._Q_only_src_0p5phi0offset_0p25phi0amp20261004_170035.npz").write_bytes(b"\0" * 64)   # macOS AppleDouble
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pz._find(tmp_path, pz.source_record_name(0.25)) == real          # the hidden file is not a match
    with pytest.raises(FileNotFoundError):
        pz._find(tmp_path, pz.source_record_name(0.5))
    _touch(tmp_path / "Q_only_src_0p5phi0offset_0p25phi0amp20261004_180000.npz")
    with pytest.warns(UserWarning, match="2 files match"):
        found = pz._find(tmp_path, pz.source_record_name(0.25))
    assert found == Path(glob.glob(str(tmp_path) + "/" + pz.source_record_name(0.25))[0])   # glob's first, as the notebooks


def test_source_baseline_is_found_at_any_offset(tmp_path):
    """53_pulse cell 12 found the amplitude-0 source record by its amplitude alone."""
    rng = np.random.default_rng(1)
    for a in pz.SRC_AMPS:
        q = np.where(np.arange(4000) % 1000 < 500, 0.3 * a, 0.0) + 0.14 + 1e-3 * rng.standard_normal(4000)
        _touch(tmp_path / f"Q_only_src_0p5phi0offset_{str(a).replace('.', 'p')}phi0amp2026.npz", Q=q.astype(np.float32), fs=40e6)
    q0 = (1e-3 * rng.standard_normal(4000) - 0.02).astype(np.float32)
    _touch(tmp_path / "Q_only_src_0p0phi0offset_0p0phi0amp2026.npz", Q=q0, fs=40e6)
    lv = pz.source_record_levels(tmp_path, n_excerpt=10)
    assert lv["v_zero"] == np.median(q0) and lv["v_zero"].dtype == np.float32
    assert _same(lv["excerpt_zero"], q0[:10])


# ----------------------------------------------------------------------------- square pulses, prepared demo data

def test_plateau_noise_equals_the_notebook_mask_form():
    """53_pulse cell 21 builds the plateau windows from bin times; the slices of ``plateau_noise`` select the same bins."""
    r = _load("pulse_folds_1004.npz")
    T, PLATEAU, NBIN = pz.PERIOD, pz.PLATEAU, pz.NBIN
    tb = (np.arange(NBIN) + 0.5) * T / NBIN
    in_plat = ((tb >= T / 2 - PLATEAU) & (tb < T / 2)) | (tb >= T - PLATEAU)
    np.testing.assert_array_equal(in_plat, pz.plateau_mask())
    for _, _, key in TWO:
        z = r[f"z_{key}"]
        on, off = r[f"plateaus_{key}"]
        noise = np.sqrt(np.mean(np.abs(np.r_[z[in_plat & (tb < T / 2)] - on, z[in_plat & (tb >= T / 2)] - off]) ** 2))
        assert pz.plateau_noise(z) == noise


def test_plateau_levels_and_noise_of_every_record():
    """53_pulse cell 13 ``plateaus`` and cell 21 ``noise`` of all 31 records."""
    pa = _load("pulse_analysis_1004.npz")
    for i in range(len(pa["z"])):
        assert _same(np.array(pz.plateau_levels(pa["z"][i])), pa["plateaus"][i])
        assert pz.plateau_noise(pa["z"][i]) == pa["noise"][i]
    assert np.median(pa["noise"]) * 1e3 == 0.027761976243859048                # PulseNoise (mV)


def test_bin_masks_and_model_binning():
    T, NBIN = pz.PERIOD, pz.NBIN
    tb = (np.arange(NBIN) + 0.5) * T / NBIN
    np.testing.assert_allclose(pz.bin_centres(), tb, **exact)
    np.testing.assert_array_equal(pz.edge_mask(), (tb < 3e-6) | ((tb >= T / 2) & (tb < T / 2 + 3e-6)))
    assert pz.edge_mask().sum() == 240 and pz.plateau_mask().sum() == 2 * pz.N_PLAT
    model = lambda q: np.exp(2j * np.pi * q / T)       # noqa: E731
    shift = -13.25e-9
    q = (np.arange(NBIN)[:, None] + (np.arange(5) + 0.5) / 5) * (T / NBIN) - shift
    np.testing.assert_allclose(pz.binned_model(model, shift), model(q).mean(axis=1), **exact)


@needs_folded
def test_prepared_pulse_folds_match_the_reference():
    """The shipped ``pulses_folded.npz`` and ``source_levels.npz`` hold the original's output for every record."""
    pa, r = _load("pulse_analysis_1004.npz"), _load("pulse_folds_1004.npz")
    with np.load(FOLDED) as f:
        assert f["z"].shape == (31, pz.NBIN) and int(f["nbin"]) == pz.NBIN and int(f["n_plat"]) == pz.N_PLAT
        assert FOLDED.stat().st_size < 1.2e6 and LEVELS.stat().st_size < 1e6
        np.testing.assert_allclose(f["t"], (np.arange(pz.NBIN) + 0.5) / pz.FS, **exact)
        for key, ref in [("offset", "offset"), ("amp", "amp"), ("period_samples", "P"), ("noise", "noise"), ("z", "z"),
                         ("static_offset", "static_offset"), ("static_z", "static_z")]:
            assert _same(f[key], pa[ref]), key
        for offset, amp, key in TWO:
            i = pz.record_index(f, offset, amp)
            assert f["period_samples"][i] == r[f"P_{key}"]
            np.testing.assert_allclose(f["z"][i], r[f"z_{key}"], **exact)
        for (o, a) in pz.DUPES:
            with pytest.raises(KeyError):
                pz.record_index(f, o, a)
    with np.load(LEVELS) as lv:
        for k in ("lo", "hi", "v_zero", "v_amp", "v_off"):
            assert _same(lv[k], pa[k]), k
        cal = pz.LevelCalibration.from_source(lv, MFR_1004)
    assert (cal.ka, cal.ko, cal.b) == (pa["KA"], pa["KO"], pa["B0"])


def test_flux_scales_are_double_precision():
    """53_pulse cell 12: ``phi_per_V``, ``KA``, ``KO``, ``B0`` and ``level_flux`` were float64, also for float32 input."""
    pa, s = _load("pulse_analysis_1004.npz"), _load("pulse_sims_1004.npz")
    assert pa["Mfr"] == MFR_1004 and pa["Phi0"] == uc.Phi0
    for Mfr in (MFR_1004, pa["Mfr"][()]):                                     # a Python float or numpy.float64
        phi_per_V = pz.flux_per_volt(Mfr)
        assert type(phi_per_V) is np.float64 and phi_per_V == pa["phi_per_V"]
        b = s["v_zero"][()] * phi_per_V                                         # float32 * phi_per_V, as cell 12
        assert type(b) is np.float64 and b == pa["B0"] == -0.08284077479415483        # PulseBase
        cal = pz.LevelCalibration.from_source(s, Mfr)
        assert all(type(v) is np.float64 for v in (cal.ka, cal.ko, cal.b))
        assert (cal.ka, cal.ko, cal.b) == (pa["KA"], pa["KO"], pa["B0"])
    assert pa["KA"] == 1.0265406594641897 and pa["KO"] == 0.9350293129762964          # PulseKa, PulseKo
    assert _same(cal.flux(pa["cal_off"], pa["cal_amp"]), pa["cal_phi"])
    x = pa["cal_off"].astype(np.float32)
    assert _same(cal.flux(x, 0.0), pa["KO"] * x + pa["B0"] + pa["KA"] * 0.0)            # float64, as level_flux
    plain = pz.LevelCalibration(ka=float(pa["KA"]), ko=float(pa["KO"]), b=float(pa["B0"]))
    assert all(type(v) is np.float64 for v in (plain.ka, plain.ko, plain.b)) and plain.flux(x, 0.0).dtype == np.float64


def test_plateau_calibration_reproduces_the_paper():
    """53_pulse cell 13: the fit of A, B to the 36 plateau levels, also with each offset held out.

    Run on the captured folds (equal to the shipped ones, above); the arrays match the original's to
    the last bit and the summary numbers are the ones recorded in values.json (PulseNLevels, PulseCalRms,
    PulseCalRel, PulseCalHeld, PulseLevelScatter).
    """
    pa = _load("pulse_analysis_1004.npz")
    s21_static = _static_model("1004")
    cal = pz.LevelCalibration.from_source(pa, MFR_1004)
    off, amp, z, off_levels = pz.calibration_levels(pa)
    assert _same(off, pa["cal_off"]) and _same(amp, pa["cal_amp"]) and _same(z, pa["cal_z"])
    fit = pz.fit_levels(s21_static(cal.flux(off, amp)), z, off)
    assert fit.A == pa["A"] and fit.B == pa["B"]
    assert _same(fit.residual, pa["cal_res"]) and _same(fit.held_out, pa["held"])
    assert _same(fit.held_out_by_level[fit.held_out_index], fit.held_out)
    assert sorted(fit.held_out_index) == list(range(36))
    diam = pz.iq_circle_diameter(fit.A, pa["kc"], pa["kappa"], pa["alpha"])
    assert diam == pa["diam"]
    ch = uc.load_config(ROOT / "configs/nist_res2_2026-10-04.toml").channels[0]
    assert pz.iq_circle_diameter(fit.A, ch.get_kappa_c(), ch.kappa, ch.alpha) == diam
    scatter = pz.off_level_scatter(off_levels)
    assert scatter == pa["scatter"]

    assert len(z) == 36                                                     # PulseNLevels
    assert pz.rms(fit.residual) * 1e3 == 0.6229623846445                    # PulseCalRms (mV)
    assert pz.rms(fit.residual) / diam * 100 == 0.9855097657605404          # PulseCalRel (% of the circle diameter)
    assert pz.rms(fit.held_out) * 1e3 == 0.7323081753769349                 # PulseCalHeld (mV)
    assert scatter * 1e3 == 0.07126921220086925                             # PulseLevelScatter (mV)


@needs_folded
def test_plateau_calibration_on_the_shipped_folds():
    """The same calibration from the shipped files, as the example does it."""
    pa = _load("pulse_analysis_1004.npz")
    with np.load(FOLDED) as f, np.load(LEVELS) as lv:
        folds, cal = dict(f), pz.LevelCalibration.from_source(lv, MFR_1004)
    off, amp, z, _ = pz.calibration_levels(folds)
    fit = pz.fit_levels(_static_model("1004")(cal.flux(off, amp)), z, off)
    assert fit.A == pa["A"] and fit.B == pa["B"] and _same(fit.held_out, pa["held"])


def test_edge_shift_fit_with_the_reference_simulations():
    """53_pulse cell 21 ``fit_shift``, ``total_residual`` and the edge rms, on the two simulated records."""
    pa, ps = _load("pulse_analysis_1004.npz"), _load("pulse_sims_1004.npz")
    A, B = pa["A"][()], pa["B"][()]
    zs = [pa["z"][pz.record_index(pa, o, a)] for o, a, _ in TWO]
    for kind in ("ideal", "tf"):
        models = [uc.periodic_interpolant(ps[f"t_{key}_{kind}"], ps[f"s21_{key}_{kind}"], pz.PERIOD) for _, _, key in TWO]
        shift = pz.fit_edge_shift(zs, models, A, B)
        assert shift == pa[f"two_shift_{kind}"]
        assert pz.edge_residual_sum(zs, models, A, B, shift) == pa[f"two_total_{kind}"]
        curve = np.array([pz.edge_residual_sum(zs, models, A, B, s) for s in np.arange(*pz.SHIFT_COARSE)])
        assert _same(curve, pa[f"two_curve_{kind}"])
        assert _same(np.array([pz.edge_rms(z, m, A, B, shift) for z, m in zip(zs, models)]), pa[f"two_edge_rms_{kind}"])
    with pytest.raises(ValueError):
        pz.edge_residual_sum(zs, models[:1], A, B, 0.0)


def test_edge_residuals_of_every_record():
    """53_pulse cells 21 and 24: the summed edge residual over all 31 records (summed in record order), the shift
    scan without refinement, and the edge rms of each record, with the static model of each record."""
    pa = _load("pulse_analysis_1004.npz")
    s21_static = _static_model("1004")
    cal = pz.LevelCalibration(ka=pa["KA"], ko=pa["KO"], b=pa["B0"])
    T = pz.PERIOD
    models = [lambda q, lo=cal.flux(o, 0.0), hi=cal.flux(o, a): s21_static(np.where((q / T) % 1 < 0.5, hi, lo))   # cell 24
              for o, a in zip(pa["offset"], pa["amp"])]
    A, B, zs = pa["A"][()], pa["B"][()], list(pa["z"])
    curve = np.array([pz.edge_residual_sum(zs, models, A, B, s) for s in np.arange(*STATIC_SHIFTS)])
    assert _same(curve, pa["static_curve"])
    shift = pz.fit_edge_shift(zs, models, A, B, coarse=STATIC_SHIFTS, fine=None)
    assert shift == pa["static_shift"]
    assert _same(np.array([pz.edge_rms(z, m, A, B, shift) for z, m in zip(zs, models)]), pa["static_edge_rms"])
