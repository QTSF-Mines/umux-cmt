"""umux_cmt: time-domain simulation of microwave SQUID multiplexers with coupled-mode theory.

The package follows the paper "Time-Domain Simulation of Microwave SQUID
Multiplexers using Coupled-Mode Theory" (Wanner, Mates, Van De Pontseele).

Typical use::

    import numpy as np
    import umux_cmt as uc

    device = uc.load_config("configs/nist_res2_2026-09-26.toml")   # the virtual chip
    s21 = uc.steady_state_sweep(device, f_probe, P_chip_dBm=-89)   # virtual VNA

    tf = uc.FluxLineTF.from_file("data/flux_line_2026-10-02/flux_line_poles.npz")
    ramp = tf.apply_periodic(uc.signals.sawtooth(100e3, amplitude=0.95), 1 / 100e3)
    I_coil = uc.signals.flux_to_current(ramp, M=device[0].line("FluxRamp").M, Phi0=uc.Phi0)
    res = uc.simulate(device, 5515.8e6, -89, t_out=np.arange(2000) / 2000 / 100e3, flux_ramp_current_func=I_coil)
    res.s21   # the transmitted probe tone over one ramp period

Modules
-------
squid, channel, device
    The virtual device: SQUID screening model, lumped resonator, feedline.
config
    TOML configuration files <-> devices.
solver, simulate
    The equation of motion and the three high-level simulations
    (steady-state sweep, quasi-static response, time domain).
flux_line, signals
    The bias-line transfer function and ideal flux drives.
fitting
    Resonator circle fits and SQUID flux-response fits that produce the
    configuration parameters from VNA data.
analysis
    Folding, alignment and calibration of measured IQ records.
io, plotting
    File readers and a plotting style. ``plotting`` (and with it matplotlib)
    is imported on first use, so that the numerical package does not load
    matplotlib.
"""
from .constants import Phi0, hbar
from .squid import rfSQUID, FluxLine, screening_weights, screening_factor, solve_kepler, screening_exact
from .channel import Channel
from .device import Multiplexer
from .config import Config, load_config, write_config, write_multichannel_config
from .solver import Solver, steady_state_field
from .simulate import (steady_state_sweep, static_response, simulate, periodic_interpolant,
                       StaticResponse, TimeDomainResult)
from .flux_line import FluxLineTF, poles_from_s2p, poles_from_impedance, differential_impedance
from .io import load_vna_sweep, load_record, read_s2p, dbm_to_watt, watt_to_dbm, dbm_to_field
from . import signals, io, fitting, analysis, demod

__version__ = "0.1.0"


def __getattr__(name):
    """Import :mod:`umux_cmt.plotting` (and matplotlib) on first use rather than with the package."""
    if name == "plotting":
        import importlib
        return importlib.import_module(".plotting", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Phi0", "hbar",
    "rfSQUID", "FluxLine", "screening_weights", "screening_factor", "solve_kepler", "screening_exact",
    "Channel", "Multiplexer",
    "Config", "load_config", "write_config", "write_multichannel_config",
    "Solver", "steady_state_field",
    "steady_state_sweep", "static_response", "simulate", "periodic_interpolant", "StaticResponse", "TimeDomainResult",
    "FluxLineTF", "poles_from_s2p", "poles_from_impedance", "differential_impedance",
    "load_vna_sweep", "load_record", "read_s2p", "dbm_to_watt", "watt_to_dbm", "dbm_to_field",
    "signals", "io", "fitting", "analysis", "plotting", "demod",
]
