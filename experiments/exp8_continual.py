"""Experiment 8 — Continual learning: does finite-eps HPC beat BPTT?

Hypothesis (from "prospective configuration", Song et al. 2024): predictive
coding's habit of updating weights at *target-nudged* activities, rather than
at the free activities backprop uses, should reduce catastrophic forgetting.
In HPC this corresponds to the ONE-SIDED nudging estimator at finite eps: as
eps -> 0 it is exact BPTT; at finite eps it evaluates the local update at the
target-nudged echo, the prospective-configuration regime. We therefore sweep
eps (one-sided) on a domain-incremental continual-learning task and compare
retention against the eps -> 0 (BPTT) limit -- a fully matched comparison
(same architecture, same data, only eps differs).

Honesty note: HPC is a BPTT-exact algorithm, so a robust advantage is not
guaranteed. This script reports whatever the data show, including a negative
result, with error bars across seeds.

Run:  python experiments/exp8_continual.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhel.models import cross_entropy_loss
from rhel.predictive_coding import HPCClassifier, init_hpc

L, W, DELTA, ALPHA = 3, 12, 0.2, 0.2
SEQ, CH, CLS = 60, 4, 4
NTASK, STEPS_PER_TASK, BATCH, LR = 6, 150, 32, 2e-3
EPS_GRID = [0.0, 0.1, 0.3, 0.6, 1.0]   # 0.0 == BPTT
N_WORLDS = 6


def _task(n, mix, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, SEQ)
    freqs = 1.0 + np.arange(CLS)
    lab = rng.integers(0, CLS, n)
    x = np.zeros((n, SEQ, CH), np.float32)
    for i in range(n):
        ph = rng.uniform(0, 2 * np.pi, CH)
        amp = rng.uniform(0.7, 1.0, CH)
        x[i] = (amp * np.sin(2 * np.pi * freqs[lab[i]] * t[:, None] + ph)).astype(np.float32)
    return (x @ mix.T).astype(np.float32), np.eye(CLS, dtype=np.float32)[lab]


def make_world(wseed):
    mixes = [np.linalg.qr(np.random.default_rng(100 + wseed * 10 + t).normal(size=(CH, CH)))[0].astype(np.float32)
             for t in range(NTASK)]
    train = [_task(500, mixes[t], wseed * 17 + t) for t in range(NTASK)]
    test = [_task(300, mixes[t], 9000 + wseed * 17 + t) for t in range(NTASK)]
    return train, test


def evaluate(m, p, test):
    return [float(jnp.mean(jnp.argmax(jax.vmap(lambda u: m(p, u))(jnp.asarray(test[t][0][:300])), -1)
                          == jnp.argmax(jnp.asarray(test[t][1][:300]), -1)))
            for t in range(NTASK)]


def run(eps, train, test, seed):
    """Train sequentially; return (avg final acc, avg forgetting)."""
    alg = "BPTT" if eps == 0.0 else "RHEL"
    gamma = 1.0 if eps == 0.0 else 1e2
    m = HPCClassifier(L, W, DELTA, ALPHA, alg, epsilon=eps, centered=False)
    p = init_hpc(jr.PRNGKey(seed), L, W, CH, CLS)
    p = jax.tree.map(lambda x: x.astype(jnp.float32), p)
    opt = optax.adam(LR)
    opt_state = opt.init(p)

    @jax.jit
    def step(p, opt_state, xb, yb):
        _, g = jax.value_and_grad(lambda q: cross_entropy_loss(m, q, xb, yb, grad_scaler=gamma))(p)
        g = jax.tree.map(lambda x: x / gamma, g)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(p, upd), opt_state

    rng = np.random.default_rng(seed)
    acc_at_train = []   # accuracy on task t right after training it
    for t in range(NTASK):
        Xt, Yt = train[t]
        for s in range(STEPS_PER_TASK):
            idx = rng.integers(0, len(Xt), BATCH)
            p, opt_state = step(p, opt_state, jnp.asarray(Xt[idx]), jnp.asarray(Yt[idx]))
        acc_at_train.append(evaluate(m, p, test)[t])
    final = evaluate(m, p, test)
    avg_final = float(np.mean(final))
    # forgetting: how much each task dropped from its just-trained accuracy
    forgetting = float(np.mean([acc_at_train[t] - final[t] for t in range(NTASK - 1)]))
    return avg_final, forgetting


def main():
    results = {f"{e}": {"final": [], "forget": []} for e in EPS_GRID}
    for wseed in range(N_WORLDS):
        train, test = make_world(wseed)
        line = []
        for e in EPS_GRID:
            af, fg = run(e, train, test, seed=wseed)
            results[f"{e}"]["final"].append(af)
            results[f"{e}"]["forget"].append(fg)
            line.append(f"{e}:{af:.3f}")
            jax.clear_caches()  # release compiled fns to bound memory use
        print(f"world {wseed}: " + "  ".join(line), flush=True)

    summary = {e: {"final_mean": float(np.mean(results[e]["final"])),
                   "final_std": float(np.std(results[e]["final"])),
                   "forget_mean": float(np.mean(results[e]["forget"])),
                   "forget_std": float(np.std(results[e]["forget"]))}
               for e in results}
    bptt = summary["0.0"]
    print("\n== retention (avg final accuracy) ==")
    for e in EPS_GRID:
        s = summary[f"{e}"]
        tag = "  (BPTT)" if e == 0.0 else ""
        delta = s["final_mean"] - bptt["final_mean"]
        print(f"  eps={e}: {s['final_mean']*100:.1f} +/- {s['final_std']*100:.1f}%"
              f"  (vs BPTT {delta*100:+.1f}){tag}")

    # paired test: best non-zero eps vs BPTT across worlds
    best_eps = max([e for e in EPS_GRID if e != 0.0],
                   key=lambda e: summary[f"{e}"]["final_mean"])
    diffs = np.array(results[f"{best_eps}"]["final"]) - np.array(results["0.0"]["final"])
    paired = {"best_eps": best_eps, "mean_diff": float(diffs.mean()),
              "std_diff": float(diffs.std()),
              "wins": int((diffs > 0).sum()), "n": len(diffs)}
    print(f"\nbest eps={best_eps}: mean retention gain {diffs.mean()*100:+.2f} pts, "
          f"wins {paired['wins']}/{paired['n']} worlds")

    # figure
    epsx = [e if e > 0 else min(EPS_GRID[1:]) / 3 for e in EPS_GRID]  # place BPTT at left
    means = [summary[f"{e}"]["final_mean"] * 100 for e in EPS_GRID]
    stds = [summary[f"{e}"]["final_std"] * 100 for e in EPS_GRID]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.errorbar(epsx[1:], means[1:], yerr=stds[1:], marker="o", color="#cc4444",
                 capsize=3, label="one-sided HPC")
    ax1.axhline(means[0], color="#4c72b0", ls="--", label="BPTT ($\\epsilon\\!\\to\\!0$)")
    ax1.axhspan(means[0] - stds[0], means[0] + stds[0], color="#4c72b0", alpha=0.12)
    ax1.set_xscale("log"); ax1.set_xlabel(r"nudge strength $\epsilon$ (one-sided)")
    ax1.set_ylabel("avg. final accuracy (%)")
    ax1.set_title(f"(A) Retention vs $\\epsilon$  ({N_WORLDS} task-sequences)")
    ax1.legend(); ax1.grid(True, ls="--", alpha=0.4)

    fmeans = [summary[f"{e}"]["forget_mean"] * 100 for e in EPS_GRID]
    fstds = [summary[f"{e}"]["forget_std"] * 100 for e in EPS_GRID]
    ax2.errorbar(epsx[1:], fmeans[1:], yerr=fstds[1:], marker="s", color="#cc4444", capsize=3)
    ax2.axhline(fmeans[0], color="#4c72b0", ls="--", label="BPTT")
    ax2.set_xscale("log"); ax2.set_xlabel(r"nudge strength $\epsilon$ (one-sided)")
    ax2.set_ylabel("avg. forgetting (%)")
    ax2.set_title("(B) Forgetting vs $\\epsilon$")
    ax2.legend(); ax2.grid(True, ls="--", alpha=0.4)

    verdict = ("finite-eps HPC robustly beats BPTT"
               if paired["mean_diff"] > 0 and paired["wins"] >= 0.75 * paired["n"]
               else "finite-eps HPC matches BPTT within noise (no robust advantage)")
    fig.suptitle(f"Continual learning: {verdict}")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp8_continual.png", dpi=160)
    os.makedirs("results", exist_ok=True)
    with open("results/exp8_continual.json", "w") as f:
        json.dump({"summary": summary, "paired": paired, "verdict": verdict}, f, indent=2)
    print(f"\nVERDICT: {verdict}")
    print("saved figures/exp8_continual.png")


if __name__ == "__main__":
    main()
