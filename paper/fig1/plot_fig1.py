"""
Figure 1: Schematic overview of trajectory SSL for quantum many-body systems.

Panel (a): 1D Hubbard model + 1-RDM definition
Panel (b): Imaginary-time trajectory converging to ground state
Panel (c): Two-stage SSL pipeline (pretrain → fine-tune)
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 300,
})

OUT = os.path.dirname(os.path.abspath(__file__))

# Load real data for panels (b) and (c)
ROOT      = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_PATH = os.path.join(ROOT, "data", "exp1_combined.npz")

COL_TR = "#2166AC"
COL_EP = "#888888"
COL_GS = "#B2182B"


# ── Panel (a): 1D Hubbard lattice + γ heatmap ────────────────────────────────

def panel_a(ax):
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-1.5, 2.5); ax.axis("off")
    L = 6

    # Draw lattice bonds
    xs = np.arange(L) * 1.4
    for i in range(L):
        j = (i + 1) % L
        x1, x2 = xs[i], xs[j] if j > 0 else xs[0] + L * 1.4
        if j > i:
            ax.plot([x1, x2], [1.0, 1.0], "k-", lw=1.5, zorder=1)
        else:  # PBC arc
            ax.annotate("", xy=(xs[0], 0.7), xytext=(xs[L-1], 0.7),
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        connectionstyle="arc3,rad=0.4",
                                        lw=1.2, linestyle="dashed"))

    # Draw sites
    colors_up = ["#4575b4"] * L
    for i, x in enumerate(xs):
        c = plt.Circle((x, 1.0), 0.28, color="#DDDDDD", ec="k", lw=1.2, zorder=2)
        ax.add_patch(c)
        # spin up arrow
        ax.annotate("", xy=(x, 1.38), xytext=(x, 1.10),
                    arrowprops=dict(arrowstyle="-|>", color="#2166AC", lw=1.0))
        # spin down arrow (some sites)
        if i % 2 == 0:
            ax.annotate("", xy=(x, 0.62), xytext=(x, 0.90),
                        arrowprops=dict(arrowstyle="-|>", color="#D73027", lw=1.0))

    # Labels
    ax.text(xs[2], 1.85, r"$t_{ij}$", ha="center", fontsize=10, color="#333333")
    ax.annotate("", xy=(xs[2]+0.35, 1.0), xytext=(xs[1]+0.35, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
    ax.text(xs[1] - 0.1, 0.55, r"$U$", ha="center", fontsize=10, color="#8B0000")
    ax.text(-0.3, -0.7,
            r"$\gamma_{ij} = \langle c^\dagger_i c^{\ }_j \rangle$",
            fontsize=10.5)
    ax.set_title(r"(a) Model: 1D Hubbard + 1-RDM $\gamma$", pad=6)


# ── Panel (b): imaginary-time trajectory ────────────────────────────────────

def panel_b(ax):
    try:
        d = np.load(DATA_PATH)
        gamma_traj = d["gamma_traj"].astype(np.float32)   # (M, 1, T+1, 12, 12)
        gamma_gs   = d["gamma_gs"].astype(np.float32)
        # Pick a representative Hamiltonian (pool, not test)
        is_test = d["is_test"].astype(bool)
        pool_idx = np.where(~is_test)[0]
        idx = pool_idx[42]
        traj = gamma_traj[idx, 0]   # (T+1, 12, 12)
        gs   = gamma_gs[idx]
        T    = len(traj)
        tau  = np.arange(T) * 0.1
        mae  = np.array([np.abs(traj[t] - gs).mean() for t in range(T)])
    except Exception:
        # Fallback: synthetic curve
        tau = np.linspace(0, 3, 31)
        mae = 0.25 * np.exp(-1.2 * tau) + 0.02

    ax.plot(tau, mae, color=COL_TR, lw=2.0)
    ax.axhline(mae[-1], color=COL_GS, lw=1.2, linestyle="--", alpha=0.7,
               label=r"$\gamma_\mathrm{GS}$")

    # Annotate γ_0 and γ_GS
    ax.annotate(r"$\gamma_0$", xy=(tau[0], mae[0]),
                xytext=(tau[0]+0.15, mae[0]+0.02),
                fontsize=9.5, color=COL_TR,
                arrowprops=dict(arrowstyle="->", color=COL_TR, lw=0.8))
    ax.annotate(r"$\gamma_\mathrm{GS}$", xy=(tau[-1], mae[-1]),
                xytext=(tau[-1]-0.8, mae[-1]+0.03),
                fontsize=9.5, color=COL_GS,
                arrowprops=dict(arrowstyle="->", color=COL_GS, lw=0.8))

    # Mark a few intermediate steps
    for t in [5, 15]:
        ax.plot(tau[t], mae[t], "o", color=COL_TR, ms=6, zorder=5)
        ax.text(tau[t]+0.05, mae[t]+0.015,
                rf"$\gamma_{{{t}\delta\tau}}$", fontsize=8, color="#555555")

    ax.set_xlabel(r"Imaginary time $\tau$")
    ax.set_ylabel(r"MAE$(\gamma_\tau, \gamma_\mathrm{GS})$")
    ax.set_title(r"(b) Imaginary-time trajectory $\gamma_\tau \to \gamma_\mathrm{GS}$", pad=6)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)


# ── Panel (c): SSL pipeline ──────────────────────────────────────────────────

def panel_c(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

    def box(x, y, w, h, color, label, sublabel=""):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor="white",
                               linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.13 if sublabel else 0),
                label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=8,
                    color="white", alpha=0.9, zorder=4)

    def arrow(x1, x2, y, label="", color="#555555"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
                    zorder=5)
        if label:
            ax.text((x1+x2)/2, y+0.2, label, ha="center",
                    fontsize=8, color=color)

    # Stage 1: Pretraining
    ax.text(2.1, 3.65, "Stage 1 — Self-supervised pretraining",
            ha="center", fontsize=9, color="#444444", style="italic")
    box(0.1, 2.6, 1.9, 0.85, "#4575B4",
        r"Trajectory data", r"$(\gamma_t, \gamma_{t+1}, H)$")
    arrow(2.0, 2.7, 3.025, color="#4575B4")
    box(2.7, 2.6, 1.9, 0.85, "#4575B4",
        r"Pretrain $f_\theta$", r"$f(\gamma_t,H)\!\to\!\gamma_{t+1}$")
    arrow(4.6, 5.3, 3.025, label="init", color="#4575B4")

    # Stage 2: Fine-tuning
    ax.text(7.3, 3.65, "Stage 2 — Fine-tuning",
            ha="center", fontsize=9, color="#444444", style="italic")
    box(5.3, 2.6, 1.9, 0.85, "#1B7837",
        r"$N$ labeled pairs", r"$(H,\, \gamma_\mathrm{GS})$")
    arrow(7.2, 7.9, 3.025, color="#1B7837")
    box(7.9, 2.6, 1.9, 0.85, "#1B7837",
        r"Fine-tune $f_\theta$", r"$f(\gamma_0,H)\!\to\!\gamma_\mathrm{GS}$")

    # Baseline arrow (endpoint only)
    ax.text(4.75, 1.9, "Endpoint-only (baseline):",
            ha="center", fontsize=8.5, color=COL_EP, style="italic")
    box(5.3, 0.9, 1.9, 0.85, "#888888",
        r"$N$ labeled pairs", r"$(H,\, \gamma_\mathrm{GS})$")
    arrow(7.2, 7.9, 1.325, color="#888888")
    box(7.9, 0.9, 1.9, 0.85, "#888888",
        r"Train from scratch", r"$f(\gamma_0,H)\!\to\!\gamma_\mathrm{GS}$")

    # Brace / bracket indicating "same N simulations"
    ax.annotate("", xy=(5.25, 0.85), xytext=(5.25, 2.55),
                arrowprops=dict(arrowstyle="-[", color="#555555",
                                lw=1.0, mutation_scale=8))
    ax.text(4.95, 1.7, "same\n$N$ sims", ha="center", fontsize=7.5, color="#555555")

    ax.set_title("(c) Two-stage SSL pipeline", pad=6)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(14, 3.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                             width_ratios=[1.1, 1.0, 1.4])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    panel_a(ax1)
    panel_b(ax2)
    panel_c(ax3)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig1_schematic.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
