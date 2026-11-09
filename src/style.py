# styles.py -- shared IEEE figure style for every script
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,              # TrueType, not Type 3
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

W1, W2 = 3.5, 7.16                                       
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "black": "#000000"}  # Okabe-Ito

def save(fig, out, name):
    """Save fig as PDF (for LaTeX) and PNG (preview) in folder `out`."""
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)

# Credits: IEEE