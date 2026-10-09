"""Build ``data/`` (the demo measurements shipped with the examples) from the raw lab archive.

The raw archive is the Colorado School of Mines / NIST measurement folder
(``NIST_uMux_measurment_data``). Most files are copied and renamed. The
square-pulse records (40 x 27 MB) and the bare-source records (7 x 9 MB) are
too large to ship, so they are reduced here to what the examples need: the
folded, edge-aligned average of every pulse record and the source levels of
every bare-source record. The reduction uses the same functions the pulse
example documents (:mod:`umux_cmt.analysis.pulses`), so a user with the raw
records can redo it.

    python scripts/prepare_demo_data.py --raw /path/to/NIST_uMux_measurment_data --out data

Only reads the raw archive; never modifies it.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np

RAW_DEFAULT = Path(__file__).resolve().parents[2] / "NIST_uMux_measurment_data"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "data"


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"  missing: {src}")
        return
    shutil.copy2(src, dst)
    print(f"  {src.name} -> {dst.relative_to(dst.parents[1])}")


def rate_khz(name):
    return int(re.search(r"_f_(\d+)_kHz", name).group(1))


def main(raw, out):
    raw, out = Path(raw), Path(out)
    print("VNA sweeps")
    copy(raw / "2026-09-26/vna/VNA_Flux_Power_Sweep_res_120260926_133948.npz", out / "vna/res2_flux_power_sweep_2026-09-26.npz")
    copy(raw / "2026-10-04/VNA_Flux_Power_Sweep_umux2_res2_20261004_154516.npz", out / "vna/res2_flux_sweep_2026-10-04.npz")
    copy(raw / "2026-09-26/vna/VNA_Flux_Power_Sweep_20260926_130343.npz", out / "vna/full_comb_flux_sweep_2026-09-26.npz")

    print("flux-ramp IQ records (2026-09-26)")
    for src in sorted((raw / "2026-09-26/iq").glob("IQ_*_f_*_kHz.npz")):
        copy(src, out / f"flux_ramps_2026-09-26/ramp_{rate_khz(src.name)}kHz.npz")

    print("flux-ramp source records (2026-09-29)")
    for src in sorted((raw / "2026-09-29/flux_ramp_direct").glob("I_only_*_f_*_kHz.npz")):
        copy(src, out / f"flux_ramp_source_2026-09-29/source_{rate_khz(src.name)}kHz.npz")

    print("flux line two-port (2026-10-02)")
    copy(raw / "2026-10-02/fridge_bypass_m20dBm.s2p", out / "flux_line_2026-10-02/flux_line_two_port.s2p")

    print("square pulses (2026-10-04): folding the raw records")
    from umux_cmt.analysis import pulses as pz   # noqa: E402  (needs the package installed)
    pulse_dir = raw / "2026-10-04/good/IQ_pulse_data"
    source_dir = raw / "2026-10-04/good/bare_source_output"
    if pulse_dir.exists():
        folded = pz.fold_pulse_records(pulse_dir)
        (out / "pulses_2026-10-04").mkdir(parents=True, exist_ok=True)
        np.savez(out / "pulses_2026-10-04/pulses_folded.npz", **folded)
        print(f"  {len(folded['offset'])} folded records -> pulses_2026-10-04/pulses_folded.npz")
    else:
        print(f"  missing: {pulse_dir}")
    if source_dir.exists():
        levels = pz.source_record_levels(source_dir)
        np.savez(out / "pulses_2026-10-04/source_levels.npz", **levels)
        print("  source levels -> pulses_2026-10-04/source_levels.npz")
    else:
        print(f"  missing: {source_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW_DEFAULT), help="raw measurement archive")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output data folder")
    args = ap.parse_args()
    main(args.raw, args.out)
