"""
Figure 1: Core idea — standard workflow vs trajectory SSL.

Left panel:  standard approach discards intermediate solver states
Right panel: trajectory SSL uses all intermediate states as free training data
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 300,
})

OUT = os.path.dirname(os.path.abspath(__file__))

# ── colors ────────────────────────────────────────────────────────────────────
C_SOLVER  = "#4A4A8A"   # dark blue-purple for solver box
C_TRAJ    = "#2166AC"   # blue for active trajectory states
C_GS      = "#B2182B"   # red for ground state
C_DISCARD = "#CCCCCC"   # gray for discarded states
C_SSL     = "#4575B4"   # blue for SSL stage
C_FT      = "#1B7837"   # green for fine-tune
C_EP      = "#777777"   # gray for endpoint-only


def rounded_box(ax, x, y, w, h, color, text, subtext="", text_color="white",
                fontsize=9.5, alpha=1.0, lw=1.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.07",
                          facecolor=color, edgecolor="white",
                          linewidth=lw, zorder=3, alpha=alpha)
    ax.add_patch(rect)
    ty = y + h / 2 + (0.12 if subtext else 0)
    ax.text(x + w / 2, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=text_color, zorder=4)
    if subtext:
        ax.text(x + w / 2, y + h / 2 - 0.18, subtext,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=text_color, alpha=0.9, zorder=4)


def down_arrow(ax, x, y_top, y_bot, color="#555555", lw=1.4, label="", ls="-"):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls),
                zorder=5)
    if label:
        ax.text(x + 0.1, (y_top + y_bot) / 2, label,
                fontsize=8, color=color, va="center")


def right_arrow(ax, x_left, x_right, y, color="#555555", lw=1.4):
    ax.annotate("", xy=(x_right, y), xytext=(x_left, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=5)


# ── draw a trajectory row ─────────────────────────────────────────────────────
def draw_trajectory(ax, x0, y, n_frames=6, active_mask=None, radius=0.22):
    """Draw γ₀ → γ₁ → … → γ_GS circles with arrows."""
    if active_mask is None:
        active_mask = [True] * n_frames
    spacing = 0.85
    xs = [x0 + i * spacing for i in range(n_frames)]

    for i, (cx, active) in enumerate(zip(xs, active_mask)):
        color = C_TRAJ if (active and i < n_frames - 1) else (C_GS if i == n_frames - 1 else C_DISCARD)
        ec = color if active else C_DISCARD
        circle = plt.Circle((cx, y), radius,
                             facecolor=color if active else "#EEEEEE",
                             edgecolor=ec, linewidth=1.4, zorder=3, alpha=1.0)
        ax.add_patch(circle)
        # label
        if i == 0:
            lbl = r"$\gamma_0$"
        elif i == n_frames - 1:
            lbl = r"$\gamma_\mathrm{GS}$"
        else:
            lbl = rf"$\gamma_{i}$"
        ax.text(cx, y, lbl, ha="center", va="center",
                fontsize=7.5 if i < n_frames - 1 else 8,
                color="white" if active else "#AAAAAA",
                fontweight="bold", zorder=4)
        # ✗ cross for inactive
        if not active and i < n_frames - 1:
            ax.text(cx, y + radius + 0.08, r"$\times$", ha="center", va="bottom",
                    fontsize=10, color="#CC4444", zorder=5)
        # arrows between circles
        if i < n_frames - 1:
            ax.annotate("", xy=(xs[i + 1] - radius, y),
                        xytext=(cx + radius, y),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=C_TRAJ if active_mask[i + 1] else C_DISCARD,
                                        lw=1.2),
                        zorder=4)
    return xs  # return x positions for wiring


# ── Main figure ───────────────────────────────────────────────────────────────
def main():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.8))

    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 6.2)
        ax.set_ylim(-0.2, 5.8)
        ax.axis("off")

    N_FRAMES  = 6   # γ₀ γ₁ γ₂ γ₃ γ₄ γ_GS

    # ── LEFT PANEL: standard approach ─────────────────────────────────────────
    ax_l.set_title("(a)  Standard approach", fontsize=13,
                   fontweight="bold", pad=8, color="#222222")

    # Solver box
    rounded_box(ax_l, 1.35, 4.5, 3.1, 0.75, C_SOLVER,
                "Quantum solver", r"(e.g. imaginary-time, DMRG, QMC)",
                fontsize=9)
    ax_l.text(3.0, 5.42, f"Run on  N  Hamiltonians",
              ha="center", fontsize=9, color="#444444", style="italic")
    down_arrow(ax_l, 3.0, 4.5, 3.95, color=C_SOLVER)

    # Trajectory label
    ax_l.text(0.08, 3.72, "Solver output:", fontsize=8.5,
              color="#444444", va="center")

    # Draw trajectory — only γ_GS active, rest discarded
    active = [False, False, False, False, False, True]
    xs_l = draw_trajectory(ax_l, 0.25, 3.55, n_frames=N_FRAMES, active_mask=active)

    # "discarded" label over the crossed states
    ax_l.text(xs_l[2], 3.55 - 0.45,
              "intermediate states  ——  discarded",
              ha="center", fontsize=8, color="#AA5555",
              style="italic")

    # Only γ_GS arrow down
    down_arrow(ax_l, xs_l[-1], 3.55 - 0.22, 2.65, color=C_GS)
    ax_l.text(xs_l[-1] + 0.12, 3.1, r"$N$ labels", fontsize=8, color=C_GS)

    # Train box
    rounded_box(ax_l, 1.4, 1.7, 3.0, 0.80, C_EP,
                "Supervised training from scratch",
                r"$f(\gamma_0, H) \to \gamma_\mathrm{GS}$",
                fontsize=9)

    # Down to result
    down_arrow(ax_l, 3.0, 1.7, 1.05, color=C_EP)
    rounded_box(ax_l, 1.4, 0.3, 3.0, 0.65, C_EP,
                "Trained predictor",
                r"MAE on test set — baseline",
                fontsize=9, alpha=0.75)

    # Budget annotation
    ax_l.text(3.0, -0.05, r"Cost: $N$ simulations", ha="center",
              fontsize=9.5, color="#333333",
              bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#BBBBBB", lw=1))

    # ── RIGHT PANEL: trajectory SSL ───────────────────────────────────────────
    ax_r.set_title("(b)  Trajectory SSL  (ours)", fontsize=13,
                   fontweight="bold", pad=8, color="#1B4F8A")

    # Solver box (identical)
    rounded_box(ax_r, 1.35, 4.5, 3.1, 0.75, C_SOLVER,
                "Quantum solver", r"(e.g. imaginary-time, DMRG, QMC)",
                fontsize=9)
    ax_r.text(3.0, 5.42, f"Run on  N  Hamiltonians  (same cost)",
              ha="center", fontsize=9, color="#444444", style="italic")
    down_arrow(ax_r, 3.0, 4.5, 3.95, color=C_SOLVER)

    ax_r.text(0.08, 3.72, "Solver output:", fontsize=8.5,
              color="#444444", va="center")

    # Draw trajectory — ALL states active
    active_all = [True] * N_FRAMES
    xs_r = draw_trajectory(ax_r, 0.25, 3.55, n_frames=N_FRAMES, active_mask=active_all)

    # "free!" label
    ax_r.text(xs_r[2], 3.55 - 0.45,
              "intermediate states  ——  free self-supervised signal!",
              ha="center", fontsize=8, color="#1B7837", style="italic",
              fontweight="bold")

    # All intermediate states arrow to Stage 1
    mid_x = (xs_r[0] + xs_r[-2]) / 2
    ax_r.annotate("", xy=(1.85, 2.72), xytext=(mid_x, 3.55 - 0.22),
                  arrowprops=dict(arrowstyle="-|>", color=C_SSL, lw=1.3,
                                  connectionstyle="arc3,rad=0.15"),
                  zorder=4)
    ax_r.text(0.62, 3.05, r"$NT$ pairs", fontsize=8, color=C_SSL)

    # γ_GS arrow to Stage 2
    ax_r.annotate("", xy=(4.2, 2.72), xytext=(xs_r[-1], 3.55 - 0.22),
                  arrowprops=dict(arrowstyle="-|>", color=C_GS, lw=1.3,
                                  connectionstyle="arc3,rad=-0.15"),
                  zorder=4)
    ax_r.text(xs_r[-1] + 0.08, 3.05, r"$N$ labels", fontsize=8, color=C_GS)

    # Stage 1 box
    rounded_box(ax_r, 0.25, 1.85, 2.75, 0.75, C_SSL,
                "Stage 1 — SSL pretrain",
                r"$f(\gamma_t, H) \to \gamma_{t+1}$  (no labels)",
                fontsize=8.5)

    # Stage 2 box
    rounded_box(ax_r, 3.2, 1.85, 2.75, 0.75, C_FT,
                "Stage 2 — Fine-tune",
                r"$f(\gamma_0, H) \to \gamma_\mathrm{GS}$",
                fontsize=8.5)

    # Arrow stage1 → stage2 (init weights)
    right_arrow(ax_r, 3.0, 3.2, 2.225, color=C_SSL)
    ax_r.text(3.1, 2.34, "init", fontsize=7.5, color=C_SSL, ha="center")

    # Down to result
    down_arrow(ax_r, 4.575, 1.85, 1.05, color=C_FT)
    rounded_box(ax_r, 1.4, 0.3, 3.0, 0.65, C_FT,
                "Trained predictor",
                r"MAE on test set — $\mathbf{\sim\!1.7\times}$ lower error",
                fontsize=9, alpha=0.9)

    # Budget annotation
    ax_r.text(3.0, -0.05, r"Cost: $N$ simulations  (identical!)",
              ha="center", fontsize=9.5, color="#1B4F8A",
              bbox=dict(boxstyle="round,pad=0.3", fc="#EEF4FF", ec="#4575B4", lw=1.2))

    plt.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig1_schematic.{ext}")
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
