"""Shared figure style: serif type, a muted palette, thin spines, light gridlines."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#2c1f2c"
MUTED = "#8c8c8c"
GRID = "#e3e3e3"
SPINE = "#6b6b6b"
TEAL = "#3f6f6c"
BRICK = "#b8573e"
OCHRE = "#8c7a45"
SAGE = "#6b8f71"
PLUM = "#5b3a5e"
# Model order of MODEL_LABELS (GPT-5, Qwen3-VL, Grok 4 Fast, Claude Sonnet 4.5), then the ensemble.
SERIES_COLORS = [TEAL, BRICK, OCHRE, SAGE, INK]
GRADE_COLORS = [TEAL, BRICK, OCHRE, SAGE, PLUM]
FILLS = ["#cfe0dc", "#f6cdb2", "#f7e9cc", "#dedfd0", "#d8e3ea"]
POINT = "#9aaeab"
SEQUENTIAL = LinearSegmentedColormap.from_list("cream_teal", ["#f7e9cc", TEAL])

RC = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def styled_subplots(*args, **kwargs) -> tuple[plt.Figure, object]:
    """``plt.subplots`` with the shared style applied."""
    mpl.rcParams.update(RC)
    return plt.subplots(*args, **kwargs)
