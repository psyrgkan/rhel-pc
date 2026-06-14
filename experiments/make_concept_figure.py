"""Concept schematic (paper Figure 1) for Hamiltonian Predictive Coding.

Draws the mechanism, no training involved:
  (left)  the PC hierarchy with top-down predictions W f(z) and bottom-up
          prediction errors e; given momenta, it is one Hamiltonian unit.
  (right) the three-pass RHEL learning cycle: one free forward inference
          pass, then two momentum-flipped echo passes nudged by +/-eps; the
          per-synapse update is the local error x activity contrast across
          the echoes -- same weights forward and backward, forward-only.

Run:  python experiments/make_concept_figure.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def box(ax, xy, w, h, text, fc, fontsize=11):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="black", lw=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def arrow(ax, a, b, color="black", style="-|>", lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def left_panel(ax):
    ax.set_title("Predictive coding hierarchy = one Hamiltonian unit",
                 fontsize=12)
    levels = [("u  (input)", "#dfe6ef"), ("z(1)", "#cfe8cf"),
              ("z(2)", "#cfe8cf"), ("z(3)", "#cfe8cf")]
    ys = [0.5, 2.0, 3.5, 5.0]
    for (label, fc), y in zip(levels, ys):
        box(ax, (1.6, y), 1.8, 0.9, label, fc)
    # top-down predictions (W f(z)) and bottom-up errors (e)
    for i in range(3):
        y_lo, y_hi = ys[i] + 0.9, ys[i + 1]
        arrow(ax, (2.1, y_hi), (2.1, y_lo), color="#c0392b", rad=-0.35)
        arrow(ax, (2.9, y_lo), (2.9, y_hi), color="#2471a3", rad=-0.35)
    ax.text(1.05, 3.3, "top-down\nprediction\n$W_\\ell f(z_{\\ell+1})$",
            color="#c0392b", ha="center", fontsize=9)
    ax.text(3.95, 3.3, "bottom-up\nerror\n$e_\\ell$", color="#2471a3",
            ha="center", fontsize=9)
    ax.text(2.5, 6.3,
            r"$H=\frac{1}{2}\|\pi\|^2 + F,\quad F=\sum_\ell \frac{1}{2}\|e_\ell\|^2$",
            ha="center", fontsize=12)
    ax.text(2.5, 5.95, "(kinetic + free energy)", ha="center", fontsize=8,
            color="#555555")
    ax.text(2.5, -0.15, "momenta $\\pi$ make inference conservative\n"
                        "(symplectic, time-reversible)",
            ha="center", fontsize=9, color="#555555")
    ax.set_xlim(0, 5); ax.set_ylim(-0.7, 6.7); ax.axis("off")


def right_panel(ax):
    ax.set_title("RHEL learning: 3 forward passes, local update", fontsize=12)
    box(ax, (0.3, 4.6), 3.2, 1.0, "1. free forward pass\n(inference, leapfrog)", "#dfe6ef")
    box(ax, (0.3, 2.8), 3.2, 1.0, "2. flip momenta,\nreplay inputs backward", "#fdebd0")
    box(ax, (0.3, 1.0), 3.2, 1.0, "   nudge by $\\pm\\epsilon\\,\\nabla\\ell$\n   (two echo passes)", "#fdebd0")
    arrow(ax, (1.9, 4.6), (1.9, 3.8))
    arrow(ax, (1.9, 2.8), (1.9, 2.0))

    box(ax, (4.4, 2.5), 4.2, 2.0,
        r"$\Delta W_\ell=\frac{\delta}{2\epsilon}\,[\,e_\ell^{+}f(z)^{T}-e_\ell^{-}f(z)^{T}\,]$"
        "\n\nlocal: error $\\times$ activity\nsame $W$ forward & backward\n(no weight transport)",
        "#cfe8cf", fontsize=11)
    arrow(ax, (3.5, 1.5), (4.4, 2.9), color="#27ae60", rad=0.2)
    ax.text(2.0, 0.35, "= BPTT gradient as $\\epsilon\\to0$ (Theorem 3.2)",
            ha="center", fontsize=10, color="#333333")
    ax.set_xlim(0, 9); ax.set_ylim(0, 6); ax.axis("off")


def main():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5),
                            gridspec_kw={"width_ratios": [1, 1.5]})
    left_panel(axs[0])
    right_panel(axs[1])
    fig.suptitle("Hamiltonian Predictive Coding: inference and learning are "
                 "one conservative physical process", fontsize=13)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/concept_schematic.png", dpi=160)
    print("saved figures/concept_schematic.png")


if __name__ == "__main__":
    main()
