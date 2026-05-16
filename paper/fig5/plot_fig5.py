"""
Figure 3 (in paper): Two-panel mechanistic analysis.

Panel (a): Algorithm robustness — imaginary-time vs power-iteration SSL.
           Both converge to the ground state via different algorithms;
           similar performance confirms the method is not specific to
           imaginary-time evolution.

Panel (b): Mechanism — shuffled vs ordered trajectory SSL.
           Shuffled uses the same gamma frames but breaks temporal order.
           Gap between shuffled and ordered quantifies the contribution
           of directional dynamics (vs. just seeing more RDM examples).
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth":  0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.dpi":      150,
    "savefig.bbox":    "tight",
    "savefig.dpi":     300,
})

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES  = os.path.join(ROOT, "results")
OUT  = os.path.dirname(os.path.abspath(__file__))

N_TRAIN_LIST = [5, 10, 20, 50, 100, 200]

COL = {"endpoint":   "#888888",
       "imag_time":  "#2166AC",
       "shuffled":   "#E87722",
       "power_iter": "#1B7837"}
MRK = {"endpoint": "^", "imag_time": "o", "shuffled": "s", "power_iter": "D"}
LS  = {"endpoint": "--", "imag_time": "-", "shuffled": ":", "power_iter": "-."}


def draw_curves(ax, r, methods, labels, title):
    x = np.array(N_TRAIN_LIST)
    for m, lbl in zip(methods, labels):
        means = np.array([np.mean(r[m][n]) for n in N_TRAIN_LIST])
        stds  = np.array([np.std(r[m][n])  for n in N_TRAIN_LIST])
        ax.errorbar(x, means, yerr=stds,
                    color=COL[m], marker=MRK[m], linestyle=LS[m],
                    capsize=3, label=lbl, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"Number of simulations $N$")
    ax.set_ylabel(r"MAE of $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9, loc="upper right")
    ax.set_ylim(bottom=0)


def main():
    r = np.load(os.path.join(RES, "exp5_results.npy"), allow_pickle=True).item()

    fig = plt.figure(figsize=(10, 4.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel (a): algorithm robustness
    draw_curves(ax1, r,
                methods=["endpoint", "imag_time", "power_iter"],
                labels=["Endpoint-only",
                        "Imaginary-time SSL",
                        "Power-iteration SSL"],
                title="(a) Algorithm robustness\n"
                      "Different convergent dynamics, similar gain")

    # Panel (b): mechanism — does temporal ordering matter?
    draw_curves(ax2, r,
                methods=["endpoint", "shuffled", "imag_time"],
                labels=["Endpoint-only",
                        "Shuffled SSL\n(same frames, random pairing)",
                        "Imaginary-time SSL\n(ordered)"],
                title="(b) Mechanism: does ordering matter?\n"
                      "Ordered > shuffled > endpoint")
    ax2.set_ylabel("")   # share y-label with panel (a)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig5_two_panel.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
