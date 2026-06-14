"""Experiment 2 — Fig. 4 replication: static RHEL vs BPTT gradient comparison.

Six-block linear and nonlinear HSSMs at the paper's SCP1 scale (sequence
length 896, 6 input channels, hidden dim 64, state dim 256, 6 blocks,
Table 4). The paper samples (u, y) from SelfRegulationSCP1; the dataset
host is unreachable from this environment, so we use a synthetic sample
with similar statistics (smooth random walks, binary one-hot target) —
Theorems 3.2/3.3 are data-independent, so this substitution is innocuous.

Panel A (float64, eps = 1e-6): cosine similarity and norm ratio between
RHEL and BPTT gradients, per parameter and per block. Replicates Fig. 4.

Panel B (float32, eps = 0.1): cosine similarity as a function of the loss
rescaling factor gamma (loss is multiplied by gamma during the echo phase,
gradients divided by gamma afterwards). Replicates the numerical underflow
analysis of App. A.5.4 (Figs. 5-6).

Run:  python experiments/exp2_gradient_comparison.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhel.models import HSSM, cross_entropy_loss, init_hssm

SEQ_LEN, DATA_DIM, LABEL_DIM = 896, 6, 2
HIDDEN, SSM_DIM, BLOCKS = 64, 256, 6
PARAM_NAMES = {
    "linear": ["A_raw", "B", "delta_raw"],
    "nonlinear": ["W_diag", "B", "b", "c_raw", "alpha"],
}
PARAM_LABELS = {
    "A_raw": "$A$", "B": "$B$", "delta_raw": r"$\delta$",
    "W_diag": "$A$", "b": "$b$", "c_raw": r"$\delta$", "alpha": r"$\alpha$",
}


def synthetic_sample(key, dtype):
    ku, ky = jr.split(key)
    u = jnp.cumsum(jr.normal(ku, (SEQ_LEN, DATA_DIM)), axis=0)
    u = (u - u.mean(0)) / (u.std(0) + 1e-8)
    y = jax.nn.one_hot(jr.randint(ky, (), 0, LABEL_DIM), LABEL_DIM)
    return u.astype(dtype), y.astype(dtype)


def compute_grads(model_type, alg, dtype, eps, gamma, key):
    params = init_hssm(key, model_type, DATA_DIM, LABEL_DIM, HIDDEN, SSM_DIM, BLOCKS)
    params = jax.tree.map(lambda x: x.astype(dtype), params)
    u, y = synthetic_sample(jr.fold_in(key, 1), dtype)
    model = HSSM(model_type, SSM_DIM, learning_algorithm=alg, epsilon=eps,
                 nonlinear_step=0.05)
    loss = lambda p: cross_entropy_loss(model, p, u[None], y[None], grad_scaler=gamma)
    grads = jax.grad(loss)(params)
    return jax.tree.map(lambda g: g / gamma, grads)


def cos_and_ratio(ga, gb):
    a, b = jnp.ravel(ga), jnp.ravel(gb)
    cos = jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30)
    ratio = jnp.linalg.norm(a) / (jnp.linalg.norm(b) + 1e-30)
    return float(cos), float(ratio)


def per_block_metrics(g_rhel, g_bptt, model_type):
    out = {}
    for name in PARAM_NAMES[model_type]:
        out[name] = [
            cos_and_ratio(g_rhel["blocks"][i]["hru"]["ssm"][name],
                          g_bptt["blocks"][i]["hru"]["ssm"][name])
            for i in range(BLOCKS)
        ]
    return out


def main():
    key = jr.PRNGKey(42)
    results = {}

    # ---- Panel A: float64 exactness (Fig. 4) ------------------------------
    for mt in ("linear", "nonlinear"):
        g_b = compute_grads(mt, "BPTT", jnp.float64, 1e-6, 1.0, key)
        g_r = compute_grads(mt, "RHEL", jnp.float64, 1e-6, 1.0, key)
        results[mt] = per_block_metrics(g_r, g_b, mt)
        print(f"[{mt}] per-parameter (cos, ratio) across blocks:")
        for name, vals in results[mt].items():
            print(f"  {name:10s} cos in [{min(v[0] for v in vals):.8f}, "
                  f"{max(v[0] for v in vals):.8f}]  "
                  f"ratio in [{min(v[1] for v in vals):.6f}, {max(v[1] for v in vals):.6f}]")

    # ---- Panel B: float32 underflow vs gamma (App. A.5.4) -----------------
    gammas = [1e0, 1e2, 1e4, 1e6]
    underflow = {}
    for mt in ("linear", "nonlinear"):
        g_b = compute_grads(mt, "BPTT", jnp.float32, 0.1, 1.0, key)
        cos_per_gamma = []
        for gam in gammas:
            g_r = compute_grads(mt, "RHEL", jnp.float32, 0.1, gam, key)
            m = per_block_metrics(g_r, g_b, mt)
            cos_per_gamma.append(
                float(np.mean([v[0] for vals in m.values() for v in vals]))
            )
        underflow[mt] = cos_per_gamma
        print(f"[{mt}] float32 mean cosine vs gamma {gammas}: {cos_per_gamma}")

    # ---- Plot --------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, BLOCKS))
    for col, mt in enumerate(("linear", "nonlinear")):
        names = PARAM_NAMES[mt]
        for row, metric in enumerate(("cosine similarity", "norm ratio")):
            ax = axs[row, col]
            for p_idx, name in enumerate(names):
                for l_idx in range(BLOCKS):
                    val = results[mt][name][l_idx][row]
                    pos = p_idx + (l_idx - BLOCKS / 2 + 0.5) * (0.8 / BLOCKS)
                    ax.bar(pos, val, width=0.8 / BLOCKS, color=colors[l_idx],
                           edgecolor="black", linewidth=0.4,
                           label=f"block {l_idx}" if p_idx == 0 and col == 0 and row == 0 else "")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([PARAM_LABELS[n] for n in names])
            ax.set_ylim(0.0, 1.15)
            ax.axhline(1.0, color="red", ls="--", lw=0.8)
            ax.grid(True, ls="--", alpha=0.5)
            if col == 0:
                ax.set_ylabel(metric)
            if row == 0:
                ax.set_title(f"{mt} HSSM (float64, $\\epsilon=10^{{-6}}$)")
    axs[0, 0].legend(fontsize=7, ncol=2)

    ax = axs[0, 2]
    for mt, marker in (("linear", "o"), ("nonlinear", "s")):
        ax.semilogx(gammas, underflow[mt], marker=marker, label=f"{mt} HSSM")
    ax.set_xlabel(r"loss rescaling $\gamma$")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(r"float32, $\epsilon=0.1$ (App. A.5.4)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=8)
    axs[1, 2].axis("off")
    fig.suptitle("RHEL vs BPTT parameter gradients, 6-block HSSMs (Fig. 4 replication)")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp2_gradient_comparison.png", dpi=160)

    os.makedirs("results", exist_ok=True)
    with open("results/exp2_metrics.json", "w") as f:
        json.dump({"fig4_float64": results,
                   "underflow_float32": {"gammas": gammas, **underflow}}, f, indent=2)
    print("saved figures/exp2_gradient_comparison.png")


if __name__ == "__main__":
    main()
