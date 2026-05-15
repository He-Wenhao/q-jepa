"""
Plot Figure 4: fine-tuning convergence curves (equal simulation budget).
Run compute_fig4.py first.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 300,
})

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES  = os.path.join(ROOT, "results")
OUT  = os.path.dirname(os.path.abspath(__file__))

COL_EP = "#888888"
COL_TR = "#2166AC"


def main():
    r  = np.load(os.path.join(RES, "fig4c_conv.npy"), allow_pickle=True).item()
    ep = r["epochs"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.plot(ep, r["ep_mean"], color=COL_EP, linestyle="--", label="Endpoint-only")
    ax.fill_between(ep, r["ep_mean"] - r["ep_std"], r["ep_mean"] + r["ep_std"],
                    color=COL_EP, alpha=0.15)

    ax.plot(ep, r["tr_mean"], color=COL_TR, linestyle="-", label="Trajectory SSL")
    ax.fill_between(ep, r["tr_mean"] - r["tr_std"], r["tr_mean"] + r["tr_std"],
                    color=COL_TR, alpha=0.15)

    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Fine-tuning convergence\n"
                 r"($N=20$ labeled examples, equal simulation budget)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig4_convergence.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
