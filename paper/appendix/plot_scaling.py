"""
Appendix: SSL data volume scaling.
Shows MAE vs N_ssl (number of unlabeled SSL Hamiltonians) at fixed N_labels.
Dotted reference lines = no-pretrain baseline.
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

N_SSL_LIST   = [50, 100, 200, 500]
N_LABEL_LIST = [5, 10, 20, 50]


def main():
    r   = np.load(os.path.join(RES, "scaling_results.npy"), allow_pickle=True).item()
    ref = np.load(os.path.join(RES, "finetune_results.npy"), allow_pickle=True).item()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    cmap   = plt.get_cmap("Blues")
    colors = [cmap(0.4 + 0.15 * i) for i in range(len(N_LABEL_LIST))]
    x = np.array(N_SSL_LIST)

    for i, n_labels in enumerate(N_LABEL_LIST):
        means = [np.mean(r[n_ssl][n_labels]) for n_ssl in N_SSL_LIST]
        stds  = [np.std(r[n_ssl][n_labels])  for n_ssl in N_SSL_LIST]
        ax.errorbar(x, means, yerr=stds,
                    color=colors[i], marker="o", linestyle="-",
                    capsize=3, label=f"$N_\\mathrm{{labels}}={n_labels}$", zorder=3)
        ref_val = np.mean(ref["no_pretrain"][n_labels])
        ax.axhline(ref_val, color=colors[i], linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"$N_\mathrm{ssl}$ (unlabeled SSL Hamiltonians)")
    ax.set_ylabel(r"MAE of $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title("Appendix: SSL data volume scaling\n(dotted = no-pretrain baseline)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9, loc="upper right")
    ax.set_ylim(bottom=0)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig_scaling.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
