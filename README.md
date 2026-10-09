# umux_cmt

**Time-domain simulation of microwave SQUID multiplexers with coupled-mode theory.**

`umux_cmt` simulates the transmitted probe tone of a microwave SQUID multiplexer (uMux) channel as a
function of time, for any flux drive on the SQUID: slow sweeps, fast flux ramps with their resets, square
pulses, detector signals. Each resonator is a single mode whose field obeys the input-output (coupled-mode)
equation of motion, and whose resonant frequency follows the flux- and readout-power-dependent inductance of
the rf-SQUID at every instant. The flux drive can be passed through the measured transfer function of the
bias line, which is what makes the transient after a ramp reset look the way it does in a real system.

The package is the code behind

> L. A. Wanner, M. Kasch, B. Mates, D. Bennett and W. Van De Pontseele, *Time-Domain Simulation of Microwave
> SQUID Multiplexers using Coupled-Mode Theory* (2026).

Every result of the paper can be regenerated from the measurements shipped in `data/` with the example
notebooks in `examples/`, and the numerics are kept identical to the code the paper was produced with
(`tests/`).

## What it does

* **Virtual device from a configuration file.** A TOML file holds the resonator parameters (`f0`, `Qr`, `Qc`,
  asymmetry angle `alpha`) and the SQUID parameters (`lamb`, `Mc`, `Mfr`, `Min`, trapped flux) of every
  channel; `uc.load_config` builds the device. The SQUID screening uses the series of Wegner, Enss and Kempf
  (2022), so the readout-power dependence is built in.
* **Three simulations.** `steady_state_sweep` (a virtual VNA, each point a short time-domain solve run to its
  steady state), `static_response` (the quasi-static `S21` versus flux at one probe tone) and `simulate` (the
  full time-domain response to any flux drive, started from the static steady state).
* **The bias line.** `FluxLineTF` extracts the poles of the flux-ramp line from a VNA two-port measurement
  (AAA rational fit of the differential impedance) and filters any periodic drive into the current the SQUID
  actually sees.
* **Parameter extraction.** `umux_cmt.fitting` turns VNA flux sweeps into configuration files: cable-delay
  removal, circle/notch fits at every flux point, flux-response fits of the SQUID, whole-chip surveys.
* **Measurement reduction.** `umux_cmt.analysis` folds and aligns digitizer records of flux ramps and pulses,
  calibrates the IQ plane with one complex gain and offset, and scores the simulation against the data.
* **Demodulation.** `umux_cmt.demod` recovers the detector flux from a flux-ramp-modulated record, with or
  without discarding the samples after each reset.

Validated on one channel of a 32-channel NIST multiplexer: steady-state `S21` to 1.5% of the resonance-circle
diameter, the power dependence with a single fitted attenuation, flux-ramp IQ traces from 1 kHz to 500 kHz to
a median of 6.5% of the signal excursion after one calibration, and square flux steps to 1.6% of the circle
diameter.

## Installation

Python 3.11 or newer, with numpy, scipy (1.15 or newer, for the AAA algorithm) and matplotlib.

```bash
git clone <this repository>
cd umux-cmt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[examples,dev]"      # the package, Jupyter, and pytest
pytest                                 # about 2.5 minutes: checks the numerics against the paper's code
```

`pytest -m "not slow"` leaves out the whole-sweep fitting checks and takes about a minute less. To get the
exact library versions the reference values were captured with, add the `paper` extra
(`pip install -e ".[examples,dev,paper]"`, numpy 2.5.0 and scipy 1.18.0).

## A first simulation

```python
import numpy as np
import umux_cmt as uc

device = uc.load_config("configs/nist_res2_2026-09-26.toml")      # one channel of the NIST chip
ch = device[0]
print(device.summary())

# virtual VNA: steady-state S21 around the resonance at -89 dBm at the chip
f = np.linspace(ch.get_fr() - 3e6, ch.get_fr() + 3e6, 601)
s21 = uc.steady_state_sweep(device, f, P_chip_dBm=-89)

# a 100 kHz flux ramp of 0.95 Phi0 through the measured bias line, read out at 5515.8 MHz
tf = uc.FluxLineTF.from_file("data/flux_line_2026-10-02/flux_line_poles.npz")
ramp = tf.apply_periodic(uc.signals.sawtooth(100e3, amplitude=0.95, offset=0.18), period=1 / 100e3)
I_coil = uc.signals.flux_to_current(ramp, M=ch.line("FluxRamp").M, Phi0=uc.Phi0)
res = uc.simulate(device, 5515.8e6, -89, t_out=np.arange(2000) / 2000 / 100e3, flux_ramp_current_func=I_coil)
res.s21        # transmitted tone over one ramp period, reset at t = 0
```

`examples/00_quickstart.ipynb` walks through this in more detail.

## Repository layout

```
src/umux_cmt/        the package
  squid.py           rf-SQUID screening response (Wegner series), flux lines, Kepler's equation
  channel.py         lumped resonator model: loss rates, instantaneous resonance, detuning
  device.py          the multiplexer: channels on one feedline
  config.py          TOML configuration files <-> devices
  solver.py          the coupled-mode equation of motion (scipy BDF)
  simulate.py        steady-state sweep, quasi-static response, time-domain simulation
  flux_line.py       bias-line transfer function from a two-port measurement
  signals.py         sawtooth, square and exponential drives
  demod.py           flux-ramp demodulation
  fitting/           parameter extraction from VNA data
  analysis/          folding, alignment and calibration of measured records
  io.py, plotting.py file readers, a value store for the notebooks, a plotting style
configs/             device configurations of the paper's channel (two days)
data/                the measurements of the paper (see data/README.md)
examples/            notebooks that reproduce the paper, section by section
tests/               bit-faithfulness tests against the original code (tests/reference)
scripts/             run the examples; rebuild the data folder from the raw archive
docs/                how the equations of the paper map onto the code
```

## The examples

| Notebook | Paper | Data | Run time |
|---|---|---|---|
| `00_quickstart` | the model in five minutes | none | < 1 min |
| `01_resonator_parameters` | Sec. IV B, Table I, Fig. 3: circle and flux-response fits, configuration files | VNA flux sweeps | 1 min |
| `02_all_channels` | Fig. 4: all 32 channels from one sweep; a multi-channel simulation | full-comb sweep | 1.5 min |
| `03_flux_line_transfer_function` | Sec. IV C, Fig. 5, Table II: poles of the bias line; the ramp amplitude | two-port, source records | < 1 min |
| `04_steady_state_s21` | Sec. V A, Fig. 6 | VNA sweep | < 1 min |
| `05_power_dependence` | Sec. V A, Fig. 7: the fitted attenuation | VNA power sweep | 1 min |
| `06_flux_ramp_response` | Sec. V B, Figs. 8-10, Table III | 28 IQ records | 1 min |
| `07_square_pulse_response` | Sec. V C, Fig. 11, Table III | folded pulse records | < 1 min |
| `08_demodulation_extension` | beyond the paper: a detector pulse through demodulation | none | 1 min |

Run them in Jupyter from the `examples/` folder, or all at once with `python scripts/run_examples.py`
(about six minutes for all nine on one core). `docs/known_issues.md` lists what to know before building on the code.
A few numbers flow from one notebook to the next through `examples/results/values.json` (the line
attenuation from `05`, the ramp amplitude from `03`); every notebook also runs on its own with the paper's
values as defaults.

## Reproducibility

The simulation core, the fitting functions and the record analysis reproduce the numbers of the paper's
original notebooks bit for bit (`tests/test_*_equivalence.py` compare against values captured from that code
with numpy 2.5.0 and scipy 1.18.0; the scripts that captured them are in `tests/reference`). The dependency
ranges in `pyproject.toml` are wider than that: with other library versions, differences at the level of
floating-point rounding can appear in the time-domain solver and in the stochastic flux-response fit, which
seeds numpy's legacy global random state before each differential-evolution run. That seeding relies on
SciPy's transitional behaviour (SPEC 7), in which `differential_evolution` without an `rng` argument draws
from the global state; should a SciPy version drop it, `fitting.fit_lambda` warns that its seed no longer
has an effect, and the equivalence tests fail.

## Citing

Please cite the paper above (see `CITATION.cff`). The device was designed and fabricated at NIST Boulder;
the measurements in `data/` were taken at the Colorado School of Mines and are distributed with NIST's
approval.

## License

MIT, see `LICENSE`.
