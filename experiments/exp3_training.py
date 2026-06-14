"""Experiment 3 — Tables 1-2 proxy: training parity of RHEL vs BPTT.

Trains linear and nonlinear HSSMs with both learning algorithms on the
synthetic delayed-cue long-range classification task (rhel.data), across
seeds, and compares test accuracy and learning curves. RHEL runs use the
paper's practical settings (float32, nudging eps = 0.1, loss rescaling
gamma = 1e4, App. A.5.3-A.5.4); BPTT runs use plain float32 autodiff.
Optimizer: Adam, as in the paper.

Run:  python experiments/exp3_training.py
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

from rhel.data import make_splits
from rhel.models import HSSM, cross_entropy_loss, init_hssm

SEQ_LEN, CHANNELS, CLASSES = 500, 4, 4
HIDDEN, SSM_DIM, BLOCKS = 32, 32, 2
BATCH, STEPS, EVAL_EVERY, LR = 32, 1500, 100, 1e-3
EPS_RHEL, GAMMA_RHEL = 0.1, 1e4
SEEDS = [0, 1, 2]


def accuracy(model, params, x, y, batch=100):
    correct = 0
    for i in range(0, len(x), batch):
        logits = jax.vmap(lambda u: model(params, u))(jnp.asarray(x[i:i + batch]))
        correct += int(jnp.sum(jnp.argmax(logits, -1) == jnp.argmax(jnp.asarray(y[i:i + batch]), -1)))
    return correct / len(x)


def train_one(model_type, alg, seed, data):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    eps, gamma = (EPS_RHEL, GAMMA_RHEL) if alg == "RHEL" else (0.0, 1.0)
    model = HSSM(model_type, SSM_DIM, learning_algorithm=alg, epsilon=eps,
                 nonlinear_step=0.05)
    params = init_hssm(jr.PRNGKey(seed), model_type, CHANNELS, CLASSES,
                       HIDDEN, SSM_DIM, BLOCKS)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(
            lambda p: cross_entropy_loss(model, p, xb, yb, grad_scaler=gamma)
        )(params)
        grads = jax.tree.map(lambda g: g / gamma, grads)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss / gamma

    rng = np.random.default_rng(seed)
    curve, best_val, test_at_best = [], 0.0, 0.0
    t0 = time.time()
    for s in range(STEPS):
        idx = rng.integers(0, len(xtr), BATCH)
        params, opt_state, loss = step(params, opt_state,
                                       jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]))
        if (s + 1) % EVAL_EVERY == 0:
            va = accuracy(model, params, xva, yva)
            curve.append({"step": s + 1, "loss": float(loss), "val_acc": va})
            if va >= best_val:
                best_val = va
                test_at_best = accuracy(model, params, xte, yte)
            print(f"  [{model_type}/{alg}/seed{seed}] step {s+1}: "
                  f"loss {float(loss):.4f} val {va:.3f} ({time.time()-t0:.0f}s)")
    return {"curve": curve, "best_val": best_val, "test_acc": test_at_best}


def main():
    data = make_splits(seq_len=SEQ_LEN, channels=CHANNELS, n_classes=CLASSES)
    results = {}
    for model_type in ("linear", "nonlinear"):
        for alg in ("BPTT", "RHEL"):
            runs = [train_one(model_type, alg, seed, data) for seed in SEEDS]
            accs = [r["test_acc"] for r in runs]
            results[f"{model_type}_{alg}"] = {
                "runs": runs,
                "test_mean": float(np.mean(accs)),
                "test_std": float(np.std(accs)),
            }
            print(f"== {model_type} {alg}: test acc "
                  f"{np.mean(accs)*100:.1f} +/- {np.std(accs)*100:.1f} %")

    os.makedirs("results", exist_ok=True)
    with open("results/exp3_training.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model_type in zip(axs, ("linear", "nonlinear")):
        for alg, color in (("BPTT", "tab:blue"), ("RHEL", "tab:red")):
            curves = results[f"{model_type}_{alg}"]["runs"]
            steps = [p["step"] for p in curves[0]["curve"]]
            vals = np.array([[p["val_acc"] for p in r["curve"]] for r in curves])
            mean, std = vals.mean(0), vals.std(0)
            ax.plot(steps, mean, color=color, label=alg)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        m = results[f"{model_type}_BPTT"], results[f"{model_type}_RHEL"]
        ax.set_title(f"{model_type} HSSM\n"
                     f"test: BPTT {m[0]['test_mean']*100:.1f}$\\pm${m[0]['test_std']*100:.1f}%  "
                     f"RHEL {m[1]['test_mean']*100:.1f}$\\pm${m[1]['test_std']*100:.1f}%")
        ax.set_xlabel("training step")
        ax.grid(True, ls="--", alpha=0.5)
        ax.axhline(1.0 / CLASSES, color="gray", ls=":", lw=1)
    axs[0].set_ylabel("validation accuracy")
    axs[0].legend()
    fig.suptitle("Delayed-cue long-range classification: RHEL matches BPTT (Tables 1-2 proxy)")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp3_training.png", dpi=160)
    print("saved figures/exp3_training.png")


if __name__ == "__main__":
    main()
