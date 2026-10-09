# Device configurations

A configuration file is the complete description of a virtual multiplexer chip: one TOML file,
loaded with `umux_cmt.load_config(path)`. The format is documented in `umux_cmt/config.py`.

| File | Device | Extracted from | Used in |
|---|---|---|---|
| `nist_res2_2026-09-26.toml` | Resonator 2 of the NIST 32-channel chip, 26 Sep 2026 state | `data/vna/res2_flux_power_sweep_2026-09-26.npz` at -26 dBm VNA output (-89 dBm at the chip), by example `01` | Paper Table I (Sep. 26); examples `00`, `04`, `05`, `06`, `08` |
| `nist_res2_2026-10-04.toml` | The same resonator, 4 Oct 2026 state (bare frequency 383 kHz lower, otherwise unchanged) | `data/vna/res2_flux_sweep_2026-10-04.npz` at -25 dBm (-88 dBm at the chip), by example `01` | Paper Table I (Oct. 4); example `07` |

Example `01` rewrites both files into `examples/results/configs/` from the demo data and compares them with
these; example `02` writes a 32-channel configuration of the whole chip there.

Notes on the parameters:

* `Qc` in the file is the fitted magnitude `|Qc_hat|` of the complex coupling; the loader converts it to
  the real-part coupling Q of the diameter correction method, `Qc = |Qc_hat| / cos(alpha)`, and recomputes
  `Qi` from `Qr` and that `Qc`.
* `lamb` and `Mc` were fitted at the extraction power and therefore already contain the RF dressing at that
  power; the simulation dresses them again from the stored energy (Sec. V A of the paper discusses the
  size of this effect, about 1% of the linewidth).
* `trapped_current` is the current on the `TrappedFlux` line (mutual `Mtr = Mfr`) that reproduces the
  SQUID's flux offset, measured from the point `phi = pi/2 + lamb` where the SQUID does not load the
  resonator and the resonance sits at `f0`.
* `Min = 246 pH` is the design value of the detector-input mutual inductance; no detector was connected
  in these measurements.
