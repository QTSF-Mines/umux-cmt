# From the paper to the code

Where each equation and procedure of *Time-Domain Simulation of Microwave SQUID Multiplexers using
Coupled-Mode Theory* lives in `umux_cmt`.

## Section II A: the effective lumped-element model

| Paper | Code |
|---|---|
| Eq. (1), `Leff(phi)` in the low-power limit | `squid.screening_exact` (reference only); the simulation uses the series below at `Phi_rf = 0` |
| Eq. (2), `fr(phi)` of a channel | `fitting.flux.fr_of_I` (the fit model), `Channel.get_fr` (the simulation model) |
| Eq. (7), `omega_r = 1/sqrt(Ceff (Lres + Leff))`, `Ceff = 4/(pi omega_0 Z0)` | `Channel._calibrate`, `Channel.get_omega_r` |
| Eq. (8), Kepler's equation for the total flux | `squid.solve_kepler`; `fitting.flux.phi_of_phie` |
| Eq. (9), the RF flux from the stored energy, `Phi_rf = Mc sqrt(16 f0 Er / Z0)` | `Channel.get_Phi_rf`, used inside `Channel.get_omega_r` |
| Eq. (10), the Wegner series (36 terms, `lamb < 0.6`) | `squid.WEGNER_A/B/C`, `squid.screening_weights`, `squid.screening_factor`, `rfSQUID.get_screening_factor` |
| `omega_r ~ omega_0` in the RF-flux conversion | `Channel.get_Phi_rf` uses `f0` |

## Section II B: the equation of motion

| Paper | Code |
|---|---|
| Eq. (23), `da/dt = -i Delta a - kappa/2 a + sqrt(kappa_c/2) a_in` | `solver.Solver.dadt` |
| Eq. (24), `a_out = a_in prod_n [1 - e^{i alpha}/cos(alpha) sqrt(kappa_c/2) a_n/a_in]` | `solver.Solver.get_a_out` |
| Eq. (25), `S21 = a_out/a_in`; Eq. (26), `Pd = |a_in|^2` | `solver.Solver.get_s21`, `io.dbm_to_field` |
| `kappa_c = omega_0 cos(alpha)/|Qc_hat| = omega_0/Qc` (DCM) | `config.Config` converts `|Qc_hat| -> Qc`; `Channel.get_kappa_c` |
| Eq. (27), the steady-state notch | `solver.steady_state_field`; `fitting.notch.notch_model` |
| `Delta(t) = omega_d - omega_r(t)` with the flux lines evaluated at `t` | `Channel.get_Delta_func`, `Multiplexer.compile_static_parameters` |

## Section III: the framework

| Paper | Code |
|---|---|
| Configuration file, virtual device | `config.Config`, `config.load_config`, `device.Multiplexer` |
| Flux lines with their mutual inductances (`Mfr`, `Min`) | `squid.FluxLine`, `rfSQUID.add_flux_line` |
| BDF integration with user-set tolerances | `solver.Solver.run_solver` (`scipy.integrate.solve_ivp`) |
| Steady-state point: `20/kappa` from rest, `8/kappa` from the previous point | `simulate.steady_state_sweep` |
| One ramp period after 5 us of settling; `max_step = min(20 ns, T/500)` | `simulate.simulate` (the notebooks pass these values) |

## Section IV: measurements

| Paper | Code |
|---|---|
| Circle fit of `S21` (Probst et al. 2015, Khalil et al. 2012) | `fitting.circle.fit_circle`, `fitting.notch.refined_fit`, `fitting.notch.fit_resonance` |
| Cable delay from the edge bands, iterated with the resonance divided out | `fitting.notch.fit_resonance_trace` |
| Flux-response fit of Eq. (2), `Ls` fixed | `fitting.flux.fit_lambda` (differential evolution), `fitting.flux.fit_flux_response` (best of five seeds, least-squares polish) |
| `phi_tr` measured from the `phi = pi/2` point | `fitting.flux.relative_I_trapped`, `fitting.flux.trapped_flux_phase` |
| Table I | `fitting.flux.summarize_channel`, `config.write_config` |
| Fig. 4, all 32 channels | `fitting.comb.find_resonances`, `fitting.comb.fit_all_resonances` |
| Eq. (28), `Sdd`, `Zdd` of the balanced pair | `flux_line.differential_impedance` |
| Eq. (29), zeros of `R + Zdd(s)` by the AAA algorithm | `flux_line.poles_from_impedance`, `flux_line.poles_from_s2p` |
| Eq. (30), `H(s) = prod (1 - s/p_k)^-1`; Table II | `flux_line.FluxLineTF.H`, `FluxLineTF.mode_table` |
| Filtering the source current into the coil current | `FluxLineTF.apply_periodic` (FFT over one period), `FluxLineTF.apply` (one-shot) |
| Ramp amplitude from the source records | notebook `03` |

## Section V: results

| Paper | Code |
|---|---|
| Fig. 6, steady-state `S21` vs measurement | notebook `04` |
| Fig. 7, power dependence, fitted attenuation `A` | notebook `05` |
| Eq. (31), `z(u) = A S21(Phi_reset + n u) + B`; Table III (ramps) | `analysis.ramps.fit_gain_offset`, `best_phase_shift`; notebook `06` |
| Folding of the IQ records, noise of the average | `analysis.ramps.fold_periods`, `fold_record`, `even_odd_noise` |
| Static (quasi-static) model | `simulate.static_response` |
| Pulse folding and alignment on the departure from the plateau | `analysis.pulses.estimate_period`, `fold_pulses`, `align_to_rising_edge` |
| Source levels to flux, `Phi = ko o + b + ka a`; Table III (pulses) | `analysis.pulses.source_levels`, `source_scales`, `flux_per_volt`; notebook `07` |
| Fig. 11, pulse edges | notebook `07` |

## Introduction

| Paper | Code |
|---|---|
| Eq. (3), `fm = n_Phi0 f_fr`; Eq. (5), demodulation per ramp cycle | `demod.demodulate` (notebook `08`) |
| Eq. (4), `theta(t) = arctan(Q/I)` | `demod.probe_phase` |
| Eq. (6), `tau_r = 2 Qr/omega_r` | `Channel.ring_up_time` |
