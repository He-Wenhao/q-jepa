"""
Figure 5: Does trajectory structure matter?
Four methods at equal simulation budget.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth":  0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
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
LBL = {"endpoint":   "Endpoint-only",
       "imag_time":  "Imaginary-time SSL (ordered)",
       "shuffled":   "Shuffled SSL (same frames, random pairing)",
       "power_iter": "Power-iteration SSL"}
LS  = {"endpoint": "--", "imag_time": "-", "shuffled": ":", "power_iter": "-."}


def main():
    r = np.load(os.path.join(RES, "exp5_results.npy"), allow_pickle=True).item()
    x = np.array(N_TRAIN_LIST)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["endpoint", "imag_time", "shuffled", "power_iter"]:
        means = np.array([np.mean(r[m][n]) for n in N_TRAIN_LIST])
        stds  = np.array([np.std(r[m][n])  for n in N_TRAIN_LIST])
        ax.errorbar(x, means, yerr=stds,
                    color=COL[m], marker=MRK[m], linestyle=LS[m],
                    capsize=4, label=LBL[m], zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"Number of simulations $N$")
    ax.set_ylabel(r"MAE of predicted $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title("Does trajectory structure matter?\n(equal simulation budget)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9, loc="upper right")
    ax.set_ylim(bottom=0)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig5_structure.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)

    print("\nN    endpoint  imag_time  shuffled  power_iter")
    for n in N_TRAIN_LIST:
        print(f"{n:4d}  "
              f"{np.mean(r['endpoint'][n]):.4f}    "
              f"{np.mean(r['imag_time'][n]):.4f}     "
              f"{np.mean(r['shuffled'][n]):.4f}    "
              f"{np.mean(r['power_iter'][n]):.4f}")


if __name__ == "__main__":
    main()
