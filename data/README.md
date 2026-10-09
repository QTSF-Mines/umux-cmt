# Demo measurement data

Measurements of one channel ("resonator 2", about 5515.3-5515.7 MHz depending on the day) of a 32-channel
NIST microwave SQUID multiplexer, taken at the Colorado School of Mines in September and October 2026.
They are the data behind the figures of the paper and are included, with NIST's approval, so that the
example notebooks can be rerun end to end. Each folder is one measurement day. Only data from the same
day should be compared with each other: the resonator's state (its bare frequency, trapped flux)
changed between cooldowns and even between days.

| Folder | Contents | Used in |
|---|---|---|
| `vna/res2_flux_power_sweep_2026-09-26.npz` | Keysight P5004B sweep of resonator 2: 16 VNA powers (-40 to -10 dBm) x 101 flux-bias points over 2 Phi0 x 1001 frequencies (5511.6-5520 MHz). Taken 2 h before the flux-ramp records | `01`, `04`, `05` (Table I, Figs. 3, 6, 7) |
| `vna/res2_flux_sweep_2026-10-04.npz` | Same resonator, one power (-25 dBm), 101 flux points over 1.5 Phi0, 5513.0-5516.8 MHz. Taken 80 min before the pulse records | `01`, `07` (Table I) |
| `vna/full_comb_flux_sweep_2026-09-26.npz` | All 32 channels, -29 dBm, 101 flux points over 2 Phi0, 5.47-5.75 GHz at 28 kHz resolution | `02` (Fig. 4 style distributions) |
| `flux_ramps_2026-09-26/ramp_<rate>kHz.npz` | Homodyne IQ records of resonator 2 while a sawtooth flux ramp of 1 nominal Phi0 ran at 28 rates from 1 kHz to 2 MHz. 40 MS/s, 20 ramp periods each, probe 5515.8 MHz at -26.33 dBm at the fridge input. `I`, `Q` in V | `06` (Figs. 8-10, Table III) |
| `flux_ramp_source_2026-09-29/source_<rate>kHz.npz` | The flux-ramp source voltage measured directly at the fridge-input BNC (digitizer at 1 MOhm), 10 periods per rate. Gives the ramp amplitude in volts; 200 kHz-2 MHz resolve the reset | `03` (ramp amplitude) |
| `flux_line_2026-10-02/flux_line_two_port.s2p` | Two-port S-parameters of the flux-ramp bias line, fridge cold, 9 kHz-100 MHz (log, 1601 points), one port on each leg of the twisted pair with the two 1 kOhm series resistors bypassed | `03` (Fig. 5, Table II) |
| `flux_line_2026-10-02/flux_line_poles.npz` | The poles of the line's transfer function extracted from that file (cached result of `03`) | `00`, `06`, `07`, `08` |
| `pulses_2026-10-04/pulses_folded.npz` | Square flux pulses (25 kHz, 20 us on / 20 us off) at 5 nominal offsets x 8 nominal amplitudes, read out at 5515.030 MHz. Each raw record (0.2 s, 5000 pulses, 40 MS/s) is folded into one 40 us period of 1600 bins and aligned to its rising edge; 31 pulsed records plus the static IQ point of each offset | `07` (Fig. 11, Table III) |
| `pulses_2026-10-04/source_levels.npz` | Levels of the pulse source measured directly on the digitizer (offset 0.5, amplitudes 0-1): the low and high medians of each record, the no-pulse baseline, and short excerpts for plotting | `07` (flux calibration) |

## Conventions

* Flux bias axes of the VNA sweeps are nominal, in Phi0 at the design mutual inductance 14.4 pH of the
  flux-ramp coil; the bias current is `flux_bias * Phi0 / 14.4e-12`. The fitted mutual turns it into flux.
* Powers: the VNA output and fridge-input powers reach the chip through -61 dB of discrete attenuators plus
  cable loss. The effective attenuation `A = -62.9 dB` used everywhere is fitted in example `05` to the measured
  power dependence of the resonance; it also absorbs the numerical prefactor of the RF-flux conversion
  (Sec. II A of the paper), so it is a power scale rather than a pure cable loss.
* Digitizer records are not triggered on the drive; the examples find the drive phase in the data.
* The pulse records carry wrong metadata (`f = 50000`, `n_periods = 10000`); the true period is 40 us.

## Reduced files

The raw pulse records (40 x 27 MB) and bare-source records (7 x 9 MB) are too large to ship. They were
reduced to the two files above with `scripts/prepare_demo_data.py`, using the folding and alignment
functions of `umux_cmt.analysis.pulses` that example `07` documents. The raw files are available from the
corresponding author on request, and the script rebuilds this folder from them.

All other files are unmodified copies of the lab files, renamed; the original names are
`VNA_Flux_Power_Sweep_res_120260926_133948.npz`, `VNA_Flux_Power_Sweep_umux2_res2_20261004_154516.npz`,
`VNA_Flux_Power_Sweep_20260926_130343.npz`, `IQ_20260926_<time>_f_<rate>_kHz.npz`,
`I_only_20260929_<time>_f_<rate>_kHz.npz` and `fridge_bypass_m20dBm.s2p`.
