"""Reduction of measured IQ records to the quantities compared with the simulations.

The simulations in :mod:`umux_cmt.simulate` produce ``S21(t)`` over one
period of a drive. A measurement produces a long, untriggered digitiser
record in volts. This subpackage closes the gap, with no fitting beyond a
complex gain and offset of the IQ plane and a shift in time:

:mod:`~umux_cmt.analysis.ramps`
    Flux-ramp records (Sec. V B of the paper): fold the periods into one,
    measure the noise of the average, fit the gain and offset ``z = A*S21 + B``,
    and align a periodic model in phase.
:mod:`~umux_cmt.analysis.pulses`
    Square-pulse records (Sec. V C): measure the true period, fold, align on
    the rising edge, read the plateau levels and noise, calibrate the flux
    of every level from the bare-source records, fit the gain and offset on
    the plateau levels (also with each offset held out), and score models on
    the edges with one common time shift.

Both modules are transcriptions of the analysis the paper was produced with,
kept numerically identical to it.
"""
from . import ramps, pulses
from .ramps import (fold_periods, reset_period, fold_record, fold_ramp_records, even_odd_noise, rms_excursion,
                    rms_distance, fit_gain_offset, periodic_phase_interpolant, best_phase_shift)
from .pulses import (estimate_period, fold_pulses, departure_bin, align_to_rising_edge, plateau_levels, plateau_noise,
                     bin_centres, plateau_mask, edge_mask, binned_model, source_levels, source_scales, flux_per_volt,
                     LevelCalibration, rms, calibration_levels, LevelFit, fit_levels, off_level_scatter,
                     iq_circle_diameter, edge_rms, edge_residual_sum, fit_edge_shift,
                     fold_pulse_records, record_index, source_record_levels)

__all__ = [
    "ramps", "pulses",
    "fold_periods", "reset_period", "fold_record", "fold_ramp_records", "even_odd_noise", "rms_excursion",
    "rms_distance", "fit_gain_offset", "periodic_phase_interpolant", "best_phase_shift",
    "estimate_period", "fold_pulses", "departure_bin", "align_to_rising_edge", "plateau_levels", "plateau_noise",
    "bin_centres", "plateau_mask", "edge_mask", "binned_model", "source_levels", "source_scales", "flux_per_volt",
    "LevelCalibration", "rms", "calibration_levels", "LevelFit", "fit_levels", "off_level_scatter",
    "iq_circle_diameter", "edge_rms", "edge_residual_sum", "fit_edge_shift",
    "fold_pulse_records", "record_index", "source_record_levels",
]
