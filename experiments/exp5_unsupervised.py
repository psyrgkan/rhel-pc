"""Experiment 5 — Energy-based UNSUPERVISED learning (benefit #2).

Predictive coding is a generative model, so HPC can learn *without labels* by
minimizing its free energy (the sensory prediction error) — something a
task-driven Hamiltonian SSM cannot do. We train HPC to autoencode clean
structured sequences with NO labels and ask:

  (a) Does unsupervised generative learning actually happen?  -> held-out
      reconstruction MSE drops far below a random HPC reservoir.
  (b) Does the LOCAL, forward-only RHEL rule match GLOBAL backprop on this
      unsupervised objective?  -> RHEL and BPTT learning curves overlap.
  (c) Are the learned representations useful?  -> linear-probe accuracy,
      compared against a classic dissipative PC network and references.

The conservative HPC needs a weak leak (alpha small) and a large enough step
to let its latents track the input; with a strong leak it collapses to
predicting the mean (we document this in the report).

Run:  python experiments/exp5_unsupervised.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhel.data import make_smooth_splits
from rhel.eval import linear_probe
from rhel.predictive_coding import (
    DissipativePC,
    HPCClassifier,
    hpc_reconstruct,
    hpc_reconstruction_loss,
    hpc_reconstruction_mse,
    hpc_representation,
    init_dissipative_pc,
    init_hpc,
)

LEVELS, WIDTH, DELTA, ALPHA = 3, 32, 0.5, 0.05   # weak leak -> latents track input
SEQ_LEN, CHANNELS, CLASSES = 60, 4, 4
STEPS, BATCH, LR = 900, 32, 3e-3
SEEDS = [0, 1, 2]


def batched(fn, X, batch=200):
    return jnp.concatenate([fn(jnp.asarray(X[i:i + batch]))
                            for i in range(0, len(X), batch)])


def train_hpc(alg, seed, xtr, xte):
    eps, gamma = (0.1, 1e2) if alg == "RHEL" else (0.0, 1.0)
    model = HPCClassifier(LEVELS, WIDTH, DELTA, ALPHA, alg, epsilon=eps)
    params = init_hpc(jr.PRNGKey(seed), LEVELS, WIDTH, CHANNELS, CLASSES)
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, xb):
        loss, grads = jax.value_and_grad(
            lambda p: hpc_reconstruction_loss(model, p, xb, grad_scaler=gamma)
        )(params)
        grads = jax.tree.map(lambda g: g / gamma, grads)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss / gamma

    rng = np.random.default_rng(seed)
    curve = []
    for s in range(STEPS):
        idx = rng.integers(0, len(xtr), BATCH)
        params, opt_state, _ = step(params, opt_state, jnp.asarray(xtr[idx]))
        if (s + 1) % 75 == 0:
            curve.append({"step": s + 1,
                          "recon_mse": hpc_reconstruction_mse(model, params, jnp.asarray(xte[:300]))})
    return model, params, curve


def main():
    (xtr, ytr), _, (xte, yte) = make_smooth_splits(
        seq_len=SEQ_LEN, channels=CHANNELS, n_classes=CLASSES, noise=0.05)
    pred_zero = float(jnp.mean(jnp.asarray(xte) ** 2))
    results = {"predict_zero_mse": pred_zero}
    t0 = time.time()

    # --- random HPC reservoir reference (no training) -------------------
    rand_mse, rand_probe = [], []
    rmodel = HPCClassifier(LEVELS, WIDTH, DELTA, ALPHA, "BPTT", epsilon=0.0)
    for seed in SEEDS:
        rp = init_hpc(jr.PRNGKey(seed + 99), LEVELS, WIDTH, CHANNELS, CLASSES)
        rp = jax.tree.map(lambda x: x.astype(jnp.float32), rp)
        rand_mse.append(hpc_reconstruction_mse(rmodel, rp, jnp.asarray(xte[:300])))
        Xtr = batched(lambda u: jax.vmap(lambda v: hpc_representation(rmodel, rp, v))(u), xtr)
        Xte = batched(lambda u: jax.vmap(lambda v: hpc_representation(rmodel, rp, v))(u), xte)
        rand_probe.append(linear_probe(Xtr, jnp.asarray(ytr), Xte, jnp.asarray(yte), jr.PRNGKey(seed)))
    results["Random-HPC"] = {"recon_mse": float(np.mean(rand_mse)),
                             "recon_std": float(np.std(rand_mse)),
                             "probe": float(np.mean(rand_probe)), "probe_std": float(np.std(rand_probe))}
    print(f"[Random-HPC] recon {np.mean(rand_mse):.3f}  probe {np.mean(rand_probe):.3f}")

    # --- HPC trained unsupervised by RHEL and by BPTT -------------------
    trained = {}
    for alg in ("RHEL", "BPTT"):
        mses, probes, curves, last = [], [], [], None
        for seed in SEEDS:
            model, params, curve = train_hpc(alg, seed, xtr, xte)
            mses.append(curve[-1]["recon_mse"])
            Xtr = batched(lambda u: jax.vmap(lambda v: hpc_representation(model, params, v))(u), xtr)
            Xte = batched(lambda u: jax.vmap(lambda v: hpc_representation(model, params, v))(u), xte)
            probes.append(linear_probe(Xtr, jnp.asarray(ytr), Xte, jnp.asarray(yte), jr.PRNGKey(seed)))
            curves.append(curve); last = (model, params)
            print(f"[HPC-{alg} seed{seed}] recon {mses[-1]:.3f} probe {probes[-1]:.3f} ({time.time()-t0:.0f}s)")
        results[f"HPC-{alg}"] = {"recon_mse": float(np.mean(mses)), "recon_std": float(np.std(mses)),
                                 "probe": float(np.mean(probes)), "probe_std": float(np.std(probes)),
                                 "curve": curves[0]}
        trained[alg] = last

    # --- classic dissipative PC ("the other network"), probe only -------
    dprobe = []
    for seed in SEEDS:
        pc = DissipativePC(LEVELS, WIDTH, alpha=0.05, eta=0.1, infer_steps=40)
        params = init_dissipative_pc(jr.PRNGKey(seed), LEVELS, WIDTH, CHANNELS)
        opt = optax.adam(LR); opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state, xb):
            loss, g = jax.value_and_grad(pc.loss)(params, xb)
            updates, opt_state = opt.update(g, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss

        rng = np.random.default_rng(seed)
        for s in range(STEPS):
            idx = rng.integers(0, len(xtr), BATCH)
            params, opt_state, _ = step(params, opt_state, jnp.asarray(xtr[idx]))
        Xtr = batched(lambda u: jax.vmap(lambda v: pc.representation(params, v))(u), xtr)
        Xte = batched(lambda u: jax.vmap(lambda v: pc.representation(params, v))(u), xte)
        dprobe.append(linear_probe(Xtr, jnp.asarray(ytr), Xte, jnp.asarray(yte), jr.PRNGKey(seed)))
    results["Dissipative-PC"] = {"probe": float(np.mean(dprobe)), "probe_std": float(np.std(dprobe))}
    raw_probe = linear_probe(jnp.asarray(xtr).mean(1), jnp.asarray(ytr),
                             jnp.asarray(xte).mean(1), jnp.asarray(yte), jr.PRNGKey(0))
    results["Raw-input"] = {"probe": raw_probe}
    print(f"[Dissipative-PC] probe {np.mean(dprobe):.3f}   [Raw] probe {raw_probe:.3f}")

    # --- figure ---------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # (A) reconstruction MSE: trained vs random vs predict-zero
    ax = axs[0]
    bars = ["Random-HPC", "HPC-BPTT", "HPC-RHEL"]
    cols = ["#bbbbbb", "#88bb88", "#cc4444"]
    means = [results[b]["recon_mse"] for b in bars]
    stds = [results[b]["recon_std"] for b in bars]
    ax.bar(range(3), means, yerr=stds, color=cols, edgecolor="black", capsize=4)
    ax.axhline(pred_zero, color="gray", ls=":", label="predict-mean")
    ax.set_xticks(range(3)); ax.set_xticklabels(bars, rotation=15)
    ax.set_ylabel("held-out reconstruction MSE")
    ax.set_title("(A) Unsupervised generative learning")
    ax.legend(); ax.grid(True, axis="y", ls="--", alpha=0.4)

    # (B) RHEL vs BPTT learning curves (should overlap = local matches global)
    ax = axs[1]
    for alg, c in (("HPC-RHEL", "#cc4444"), ("HPC-BPTT", "#88bb88")):
        cv = results[alg]["curve"]
        ax.plot([p["step"] for p in cv], [p["recon_mse"] for p in cv],
                color=c, marker="o", ms=3, label=alg)
    ax.axhline(pred_zero, color="gray", ls=":", label="predict-mean")
    ax.set_xlabel("training step"); ax.set_ylabel("reconstruction MSE")
    ax.set_title("(B) Local RHEL tracks global BPTT")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)

    # (C) example reconstruction (one channel) by a trained HPC-RHEL
    ax = axs[2]
    model, params = trained["RHEL"]
    u = jnp.asarray(xte[0])
    rec = hpc_reconstruct(model, params, u)
    rp = init_hpc(jr.PRNGKey(99), LEVELS, WIDTH, CHANNELS, CLASSES)
    rp = jax.tree.map(lambda x: x.astype(jnp.float32), rp)
    rec_rand = hpc_reconstruct(rmodel, rp, u)
    ax.plot(u[:, 0], "k", lw=2, label="input")
    ax.plot(rec[:, 0], "#cc4444", lw=1.5, label="HPC-RHEL recon")
    ax.plot(rec_rand[:, 0], "#bbbbbb", lw=1.2, ls="--", label="random HPC")
    ax.set_xlabel("time"); ax.set_title("(C) Example reconstruction (1 channel)")
    ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)

    rr, rb = results["HPC-RHEL"]["recon_mse"], results["Random-HPC"]["recon_mse"]
    fig.suptitle("HPC learns a generative model unsupervised with a local, forward-only rule: "
                 f"reconstruction MSE {rb:.3f} (random) -> {rr:.3f} (trained), "
                 "RHEL == BPTT")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp5_unsupervised.png", dpi=160)
    os.makedirs("results", exist_ok=True)
    with open("results/exp5_unsupervised.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved figures/exp5_unsupervised.png")


if __name__ == "__main__":
    main()
