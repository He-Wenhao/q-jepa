"""
Generate all paper figures from saved results.

Figures:
  fig1_main.{pdf,png}     — Exp 1: interpolation, MAE vs N_labels
  fig2_extrap.{pdf,png}   — Exp 2: extrapolation (U∈[6,10] OOD)
  fig3_scaling.{pdf,png}  — Exp 3: SSL data volume scaling
  fig_combined.{pdf,png}  — 3-panel combined figure
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

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES   = os.path.join(ROOT, "results")
OUT   = os.path.dirname(os.path.abspath(__file__))

# ── Palette ───────────────────────────────────────────────────────────────────
COL = {
    "no_pretrain":   "#888888",
    "rand_pretrain": "#E87722",
    "traj_pretrain": "#2166AC",
}
MRK = {"no_pretrain": "^", "rand_pretrain": "s", "traj_pretrain": "o"}
LBL = {
    "no_pretrain":   "A: no pretrain",
    "rand_pretrain": "B: random-pair pretrain",
    "traj_pretrain": "C: trajectory pretrain",
}
N_LABEL_LIST = [5, 10, 20, 50]
N_SSL_LIST   = [50, 100, 200, 500]


# ── Helper ────────────────────────────────────────────────────────────────────

def load_main(fname="finetune_results.npy"):
    r = np.load(os.path.join(RES, fname), allow_pickle=True).item()
    methods = list(r.keys())
    means = {m: [np.mean(r[m][n]) for n in N_LABEL_LIST] for m in methods}
    stds  = {m: [np.std(r[m][n])  for n in N_LABEL_LIST] for m in methods}
    return means, stds


def draw_main_ax(ax, means, stds, title, show_ylabel=True):
    x = np.array(N_LABEL_LIST)
    for m in ["no_pretrain", "rand_pretrain", "traj_pretrain"]:
        ls = "-" if m == "traj_pretrain" else ("--" if m == "rand_pretrain" else ":")
        ax.errorbar(x, means[m], yerr=stds[m],
                    color=COL[m], marker=MRK[m], linestyle=ls,
                    capsize=3, label=LBL[m], zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"$N_\mathrm{labels}$ (fine-tuning examples)")
    if show_ylabel:
        ax.set_ylabel(r"MAE of $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=0)


def draw_scaling_ax(ax):
    r = np.load(os.path.join(RES, "scaling_results.npy"), allow_pickle=True).item()
    x = np.array(N_SSL_LIST)
    n_ssl_max = max(N_SSL_LIST)

    # Reference: no-pretrain at each N_labels (horizontal dashed lines, light)
    ref = np.load(os.path.join(RES, "finetune_results.npy"), allow_pickle=True).item()

    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.4 + 0.15 * i) for i in range(len(N_LABEL_LIST))]

    for i, n_labels in enumerate(N_LABEL_LIST):
        means = [np.mean(r[n_ssl][n_labels]) for n_ssl in N_SSL_LIST]
        stds  = [np.std(r[n_ssl][n_labels])  for n_ssl in N_SSL_LIST]
        ax.errorbar(x, means, yerr=stds,
                    color=colors[i], marker="o", linestyle="-",
                    capsize=3, label=f"$N_\\mathrm{{labels}}={n_labels}$", zorder=3)
        # Reference line (no pretrain)
        ref_val = np.mean(ref["no_pretrain"][n_labels])
        ax.axhline(ref_val, color=colors[i], linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel(r"$N_\mathrm{ssl}$ (SSL Hamiltonians)")
    ax.set_ylabel(r"MAE of $\hat{\gamma}_\mathrm{GS}$")
    ax.set_title("(c) SSL data volume scaling\n(dotted = no-pretrain baseline)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(framealpha=0.9, loc="upper right")
    ax.set_ylim(bottom=0)


# ── Individual figures ────────────────────────────────────────────────────────

def fig_main():
    means, stds = load_main("finetune_results.npy")
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_main_ax(ax, means, stds, "(a) In-distribution\n(train/test U∈[0,8])")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_main.{ext}"))
    plt.close(fig)
    print("Saved fig1_main.{pdf,png}")


def fig_extrap():
    means, stds = load_main("finetune_extrap_results.npy")
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_main_ax(ax, means, stds,
                 "(b) Extrapolation\n(SSL/train U∈[0,6], test U∈[6,10])",
                 show_ylabel=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_extrap.{ext}"))
    plt.close(fig)
    print("Saved fig2_extrap.{pdf,png}")


def fig_scaling():
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_scaling_ax(ax)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_scaling.{ext}"))
    plt.close(fig)
    print("Saved fig3_scaling.{pdf,png}")


def fig_combined():
    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    means1, stds1 = load_main("finetune_results.npy")
    means2, stds2 = load_main("finetune_extrap_results.npy")

    draw_main_ax(ax1, means1, stds1,
                 "(a) In-distribution\nU∈[0,8] train & test",
                 show_ylabel=True)
    draw_main_ax(ax2, means2, stds2,
                 "(b) Extrapolation\nSSL/fine-tune U∈[0,6] → test U∈[6,10]",
                 show_ylabel=False)
    draw_scaling_ax(ax3)

    # Remove duplicate legends from (b)
    ax2.get_legend().remove()
    ax2.legend(framealpha=0.9)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_combined.{ext}"))
    plt.close(fig)
    print("Saved fig_combined.{pdf,png}")


if __name__ == "__main__":
    fig_main()
    fig_extrap()
    fig_scaling()
    fig_combined()
