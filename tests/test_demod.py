"""The demodulator recovers the detector flux from a modulated signal period by period."""
import numpy as np
import pytest

from umux_cmt.demod import demodulate, phase_to_flux, probe_phase


def _record(f_ramp=100e3, fs=50e6, n_periods=20, phi_in=None):
    """A readout signal periodic in the total flux: ramp (rising, 1 Phi0 per period) plus the detector flux."""
    T = 1 / f_ramp
    t = np.arange(0, n_periods * T, 1 / fs)
    tau = t - np.floor(t / T) * T
    phi = 0.1 if phi_in is None else phi_in(t)
    return t, np.cos(2 * np.pi * (f_ramp * tau + phi)), T


@pytest.mark.parametrize("method", ["sum", "lstsq"])
def test_recovers_known_detector_flux(method):
    T = 1 / 100e3
    t, y, _ = _record(phi_in=lambda t: np.where(t < 10 * T, 0.1, 0.3))
    t_p, dtheta = demodulate(y, t, 100e3, 1.0, method=method)
    assert len(t_p) == 20
    expected = np.where(t_p < 10 * T, 0.1, 0.3)
    np.testing.assert_allclose(phase_to_flux(dtheta), expected, atol=1e-6)


def test_sum_and_lstsq_agree_over_full_periods():
    t, y, _ = _record(phi_in=lambda t: 0.05 + 0.2 * np.sin(2 * np.pi * 5e3 * t))
    _, a = demodulate(y, t, 100e3, 1.0, method="sum")
    _, b = demodulate(y, t, 100e3, 1.0, method="lstsq")
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_discard_drops_the_start_of_each_period_without_bias():
    t, y, T = _record(n_periods=5, phi_in=lambda t: 0.4 / (2 * np.pi))
    tau = t - np.floor(t / T) * T
    y = y + np.where(tau < 0.2 * T - 1e-9, 10.0, 0.0)          # a transient after every reset
    _, bad = demodulate(y, t, 100e3, 1.0)
    _, good = demodulate(y, t, 100e3, 1.0, discard=0.2 * T)
    assert np.max(np.abs(phase_to_flux(good) - 0.4 / (2 * np.pi))) < 1e-7
    assert np.max(np.abs(phase_to_flux(bad) - 0.4 / (2 * np.pi))) > 0.01
    _, biased = demodulate(y, t, 100e3, 1.0, discard=0.2 * T, method="sum")
    assert np.max(np.abs(phase_to_flux(biased) - 0.4 / (2 * np.pi))) > 1e-4   # the raw sums are biased on a partial period


def test_harmonics_in_the_basis_remove_their_leakage():
    T = 1 / 100e3
    t, y, _ = _record(n_periods=8, phi_in=lambda t: 0.05)
    tau = t - np.floor(t / T) * T
    y = y + 0.5 * np.cos(2 * (2 * np.pi * (1e5 * tau + 0.05)))     # a strong second harmonic, as a SQUID response has
    _, one = demodulate(y, t, 100e3, 1.0, discard=0.3 * T, n_harmonics=1)
    _, two = demodulate(y, t, 100e3, 1.0, discard=0.3 * T, n_harmonics=2)
    assert np.max(np.abs(phase_to_flux(two) - 0.05)) < 1e-6
    assert np.max(np.abs(phase_to_flux(one) - 0.05)) > 1e-3


def test_probe_phase_about_the_circle_centre():
    phi = np.linspace(0, 1, 1000, endpoint=False)
    s21 = 0.5 + 0.4 * np.exp(2j * np.pi * phi)
    theta = probe_phase(s21, center=0.5)
    assert abs((theta[-1] - theta[0]) - 2 * np.pi * (1 - 1 / 1000)) < 1e-9
