"""
Figure 3: OOD generalization — trajectory SSL vs endpoint-only on unseen U range.

SSL pretrained on U∈[0,6]; fine-tuned on N labeled examples from U∈[0,6];
tested on U∈[6,10] (Mott insulator regime, never seen during SSL).
Three methods: no pretrain, random-pair pretrain, trajectory pretrain.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

COL = {
    "no_pretrain":   "#888888",
    "rand_pretrain": "#E87722",
    "traj_pretrain": "#2166AC",
}
MRK = {"no_pretrain": "^", "rand_pretrain": "s", "traj_pretrain": "o"}
LBL = {
    "no_pretrain":   "No pretrain",
    "rand_pretrain": "Random-pair pretrain",
    "traj_pretrain": "Trajectory SSL",
}
N_LABEL_LIST = [5, 10, 20, 50]


def main():
    r = np.load(os.path.join(RES, "finetune_extrap_results.npy"),
                allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.array(N_LABEL_LIST)

    for m in ["no_pretrain", "rand_pretrain", "traj_pretrain"]:
        means = np.array([np.mean(r[m][n]) for n in N_LABEL_LIST])
        stds  = np.array([np.std(r[m][n])  for n in N_LABEL_LIST])
        ls = "-" if m == "traj_pretrain" else ("--" if m == "rand_pretrain" else ":")
        ax.errorbar(x, means, yerr=stds,
                    color=COL[m], marker=MRK[m], linestyle=ls,
                    capsize=3, label=LBL[m], zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"$N_\mathrm{labels}$ (fine-tuning examples)")
    ax.set_ylabel(r"MAE of $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title("OOD generalization\n"
                 r"SSL/train $U\in[0,6]$ → test $U\in[6,10]$ (Mott regime)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig3_extrap.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
