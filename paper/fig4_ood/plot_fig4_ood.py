"""
Figure 4: OOD generalization at equal simulation budget.
Train/pretrain on U∈[0,6]; test on U∈[6,10] (Mott insulator regime).
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
    "legend.fontsize": 11,
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
COL = {"endpoint": "#888888", "trajectory": "#2166AC"}
MRK = {"endpoint": "^", "trajectory": "o"}
LBL = {"endpoint":   "Endpoint-only",
       "trajectory": "Trajectory SSL"}


def main():
    r = np.load(os.path.join(RES, "exp_ood_results.npy"), allow_pickle=True).item()
    x = np.array(N_TRAIN_LIST)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in ["endpoint", "trajectory"]:
        means = np.array([np.mean(r[m][n]) for n in N_TRAIN_LIST])
        stds  = np.array([np.std(r[m][n])  for n in N_TRAIN_LIST])
        ls    = "-" if m == "trajectory" else "--"
        ax.errorbar(x, means, yerr=stds,
                    color=COL[m], marker=MRK[m], linestyle=ls,
                    capsize=4, label=LBL[m], zorder=3)

    # Annotate speedup at N=10 and N=50
    for n in [10, 50]:
        ep = np.mean(r["endpoint"][n])
        tr = np.mean(r["trajectory"][n])
        ax.annotate(f"×{ep/tr:.1f}",
                    xy=(n, tr), xytext=(n * 1.15, tr * 1.4),
                    fontsize=9, color=COL["trajectory"],
                    arrowprops=dict(arrowstyle="-", color=COL["trajectory"],
                                    lw=0.8, alpha=0.6))

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"Number of simulations $N$")
    ax.set_ylabel(r"MAE of predicted $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title("OOD generalization (equal budget)\n"
                 r"Train $U\in[0,6]$ → Test $U\in[6,10]$ (Mott regime)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig4_ood.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)

    print("\nN    endpoint  trajectory  speedup")
    for n in N_TRAIN_LIST:
        ep = np.mean(r["endpoint"][n])
        tr = np.mean(r["trajectory"][n])
        print(f"{n:4d}  {ep:.4f}    {tr:.4f}      ×{ep/tr:.2f}")


if __name__ == "__main__":
    main()
