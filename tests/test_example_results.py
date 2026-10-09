"""The headline numbers the example notebooks record must match the paper.

Runs only after the examples have been executed (``python scripts/run_examples.py``),
which writes ``examples/results/values.json``. Each entry is checked against the
value printed in the paper, within the tolerance of its printed precision.
"""
import json
from pathlib import Path

import pytest

VALUES = Path(__file__).parents[1] / "examples" / "results" / "values.json"

# name: (paper value, tolerance); the notebook that records it is in the comment
PAPER = {
    # 01_resonator_parameters, Table I
    "lamb0926": (0.318, 0.001), "lamb1004": (0.316, 0.001),
    "Mc0926": (3.69, 0.01), "Mc1004": (3.70, 0.01),
    "Mfr0926": (14.41, 0.01), "Mfr1004": (14.40, 0.01),
    "f0_0926": (5515.689, 0.001), "f0_1004": (5515.305, 0.001),
    "Qr0926": (9515, 1), "Qr1004": (9527, 1),
    "QcDcm0926": (10475, 2), "QcDcm1004": (10464, 2),
    "phitr0926": (-2.791, 0.002), "phitr1004": (-2.651, 0.002),
    # 03_flux_line_transfer_function, Table II and the ramp amplitude
    "PoleOneF": (4.53, 0.005), "PoleOneQ": (5.6, 0.05), "ResBW": (580, 1), "HatResBW": (1.02, 0.005),
    "RampAmp": (269.2, 0.1), "RampAmpPhi": (0.946, 0.001),
    # 04_steady_state_s21
    "SResRel": (1.5, 0.05), "SResRelMax": (2.8, 0.05), "SFitRel": (1.1, 0.05), "SNoiseRel": (0.38, 0.01),
    "SfrOff": (-6.7, 0.1), "frSim": (5515.955, 0.001),
    # 05_power_dependence
    "Atten": (-62.9, 0.05), "PowerRms": (5.7, 0.1), "PowerNone": (41, 1), "PowerUnexplained": (14, 1),
    "PowerHeld": (5.8, 0.1), "PowerSpan": (170, 1),
    # 06_flux_ramp_response, Table III and the residuals
    "IQPhiReset": (0.1845, 0.001), "IQGainMag": (68.96, 0.05), "IQGainArg": (1.220, 0.005), "IQResCal": (0.038, 0.002),
    "IQResMed": (6.5, 0.1), "IQResMax": (10.1, 0.1), "IQNoiseMed": (1.6, 0.1), "IQQsMed": (47, 1), "IQUnexplainedMed": (16, 1),
    # 07_square_pulse_response
    "PulseKa": (1.027, 0.001), "PulseKo": (0.935, 0.001), "PulseBase": (-0.083, 0.001),
    "PulseGainMag": (69.15, 0.05), "PulseCalRms": (0.62, 0.02), "PulseShift": (-13, 1),
    "PulseResRel": (1.6, 0.1), "PulseResRelMax": (2.4, 0.1), "PulseStaticRel": (14, 1), "PulseUnexplainedMed": (11, 1),
    "PulseRatio": (105, 3),
}


@pytest.mark.skipif(not VALUES.exists(), reason="examples not executed yet")
@pytest.mark.parametrize("name", sorted(PAPER))
def test_recorded_value_matches_paper(name):
    d = json.loads(VALUES.read_text())
    if name not in d:
        pytest.skip(f"{name} not recorded (its notebook has not run)")
    expected, tol = PAPER[name]
    assert abs(d[name]["value"] - expected) <= tol, (name, d[name]["value"], expected)
