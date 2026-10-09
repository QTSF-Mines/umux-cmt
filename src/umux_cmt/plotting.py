"""A compact plotting style and palette for the example notebooks.

::

    from umux_cmt.plotting import use_style, SINGLE, DOUBLE, TEAL, ORANGE, save
    use_style()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    ...
    save(fig, "my_figure", "results/figures")

The six colours are a desaturated ColorBrewer Dark2 (Harrower & Brewer 2003)
plus one blue. Under normal vision the full cycle keeps a CIEDE2000 distance
above 22; under deuteranopia VIOLET/BLUE and TEAL/ROSE come close, so avoid
using only those two in one figure. ``ORDERED_3`` rises monotonically in
lightness and is meant for a swept quantity (power, flux, temperature) plotted
from low to high.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SINGLE = 3.4
"""Width of a single-column figure in inches (APS/AIP two-column layout)."""
DOUBLE = 7.0
"""Width of a double-column figure in inches."""

VIOLET = "#5b4b8a"
BLUE = "#1f6f9c"
TEAL = "#1f9c8a"
ORANGE = "#c9782e"
ROSE = "#b1527a"
BROWN = "#84523d"
GRAY = "#6b6b6b"

MEASURED = "black"
"""Colour of measured data."""
ZERO = "#9a9a9a"
"""Colour of dashed zero/reference lines."""

COLORS = [VIOLET, BLUE, TEAL, ORANGE, ROSE, BROWN]
ORDERED_3 = [VIOLET, BLUE, TEAL]

MODEL_TF = TEAL
"""Colour used throughout the examples for the model driven through the flux-line transfer function."""
MODEL_IDEAL = ORANGE
"""Colour used throughout the examples for the model driven with the ideal (unfiltered) waveform."""

STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "axes.linewidth": 0.8,
    "axes.prop_cycle": mpl.cycler(color=COLORS),
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "figure.figsize": (SINGLE, 2.5),
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}


def use_style():
    """Apply the style to every figure made after this call."""
    mpl.rcParams.update(STYLE)


def grid(ax):
    """The light dashed grid used in the examples."""
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)


def panel_labels(axes, x=-0.18, y=1.0):
    """Write (a), (b), (c), ... in the top-left corner of each axis."""
    for ax, letter in zip(axes, "abcdefghijklmnopqrstuvwxyz"):
        ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold", va="top")


def save(fig, name, directory="figures", formats=("pdf", "png")):
    """Save ``fig`` as ``<directory>/<name>.<ext>`` for every format, creating the directory."""
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(path.with_name(f"{path.name}.{ext}"))
    return path


__all__ = ["use_style", "grid", "panel_labels", "save", "SINGLE", "DOUBLE", "VIOLET", "BLUE", "TEAL", "ORANGE",
           "ROSE", "BROWN", "GRAY", "MEASURED", "ZERO", "COLORS", "ORDERED_3", "MODEL_TF", "MODEL_IDEAL", "plt"]
