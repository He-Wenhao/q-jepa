"""
Plot Figure 4 from precomputed results.
Run compute_fig4.py first.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
COL_EX = "#B2182B"


def panel_a(ax):
    r = np.load(os.path.join(RES, "fig4a_iter.npy"), allow_pickle=True).item()
    k = r["k_axis"]
    t = r["t_axis"]

    ax.plot(k, r["model_mae"], color=COL_TR, linestyle="-",
            label="Model: $f^k(\\gamma_0, H)$")
    # True trajectory: rescale t axis to match scale (each step = δτ=0.1)
    # Plot on same x-axis as steps
    ax.plot(t, r["true_mae"], color=COL_EX, linestyle="--",
            label="True imaginary-time\ntrajectory $\\gamma_t$")

    # Mark where true traj ends
    ax.axvline(len(t)-1, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(len(t)-1+0.5, r["model_mae"][len(t)-1]*1.1, "end of\ntrajectory",
            fontsize=8, color="gray")

    ax.set_xlabel("Step $k$ (or $t$)")
    ax.set_ylabel(r"MAE$(\hat{\gamma},\, \gamma_\mathrm{GS})$")
    ax.set_title("(a) Iterated prediction\n$f^k(\\gamma_0, H)$ vs true dynamics")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)


def panel_b(ax):
    r = np.load(os.path.join(RES, "fig4b_obs.npy"), allow_pickle=True).item()
    U       = r["U"]
    true_ek = r["true_ekin"]
    ep_ek   = r["ep_ekin"]
    tr_ek   = r["tr_ekin"]

    # Bin by U and compute mean ± std
    bins = np.linspace(0, 8, 9)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def binned(vals):
        means, stds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (U >= lo) & (U < hi)
            if mask.sum() > 0:
                means.append(vals[mask].mean())
                stds.append(vals[mask].std())
            else:
                means.append(np.nan); stds.append(np.nan)
        return np.array(means), np.array(stds)

    true_m, true_s = binned(true_ek)
    ep_m,   ep_s   = binned(ep_ek)
    tr_m,   tr_s   = binned(tr_ek)

    ax.errorbar(centers, true_m, yerr=true_s, color=COL_EX,
                marker="o", linestyle="-", capsize=3, label="Exact ED", zorder=4)
    ax.errorbar(centers, ep_m,   yerr=ep_s,   color=COL_EP,
                marker="^", linestyle="--", capsize=3, label="Endpoint-only", zorder=3)
    ax.errorbar(centers, tr_m,   yerr=tr_s,   color=COL_TR,
                marker="s", linestyle="-",  capsize=3, label="Trajectory SSL", zorder=3)

    ax.set_xlabel("Interaction $U/t$")
    ax.set_ylabel(r"Kinetic energy $E_\mathrm{kin}$")
    ax.set_title("(b) Mott physics: $E_\\mathrm{kin}$ vs $U$\n($N=20$ labeled examples)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def panel_c(ax):
    r = np.load(os.path.join(RES, "fig4c_conv.npy"), allow_pickle=True).item()
    ep = r["epochs"]

    ax.plot(ep, r["ep_mean"], color=COL_EP, linestyle="--", label="Endpoint-only")
    ax.fill_between(ep,
                    r["ep_mean"] - r["ep_std"],
                    r["ep_mean"] + r["ep_std"],
                    color=COL_EP, alpha=0.15)

    ax.plot(ep, r["tr_mean"], color=COL_TR, linestyle="-",  label="Trajectory SSL")
    ax.fill_between(ep,
                    r["tr_mean"] - r["tr_std"],
                    r["tr_mean"] + r["tr_std"],
                    color=COL_TR, alpha=0.15)

    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel(r"Validation MAE")
    ax.set_title("(c) Fine-tuning convergence\n($N=20$ labeled examples)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)


def main():
    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    panel_a(ax1)
    panel_b(ax2)
    panel_c(ax3)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig4_combined.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
