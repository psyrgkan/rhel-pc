"""Experiment 9 — Dissipative temporal PC: controlled dissipation and the
local-credit-assignment tradeoff.

This experiment builds the dissipative counterpart of Hamiltonian PC to
attack the open problem from exp8 (conservative HPC matches but cannot beat
BPTT). It uses per-timestep equilibrium inference (rhel.temporal_pc): the
latents relax to the minimum of a local free energy each step, where
Equilibrium Propagation gives an exact, local gradient; learned transition
weights carry temporal memory. Relaxation IS the controlled dissipation.

Three panels, reported honestly:
  (A) Controlled dissipation. The per-step inference residual ||grad F||
      decays as the number of relaxation steps grows: the system dissipates
      energy and approaches the predictive-coding equilibrium.
  (B) The credit-assignment tradeoff. The local EqProp gradient agrees with
      the true (autodiff) gradient ONLY once the dynamics reach equilibrium:
      cosine rises from near zero (few relaxation steps, driven/non-settled
      regime) to ~0.95 (relaxed regime). Exact local credit assignment
      requires equilibrium -- which is exactly why conservative RHEL is
      special and why dissipation is needed for a PC-style local rule.
  (C) It trains, but does not beat BPTT. The local rule learns a temporally
      structured task well above chance; full BPTT through the same model
      does better. Locality and dissipation are bought at a cost in
      performance (truncated cross-timestep credit), the honest state of the
      open problem.

Run:  python experiments/exp9_dissipative_pc.py
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
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhel import temporal_pc as T
from rhel.predictive_coding import make_hpc_spec, pc_errors

L, WD, CH, CLS = 2, 12, 3, 4
SEQ = 25
RHO, ALPHA, ETA, BETA = 1.0, 0.1, 0.2, 0.02


def momentum_eqprop_cosine(key):
    """Cosine of naive *damped-momentum* temporal EqProp vs the true gradient.

    This is the alternative dissipative design: keep HPC's continuous momentum
    dynamics but add damping, and learn by free/nudged contrast over the
    (non-settling) trajectory. It FAILS -- the driven oscillatory dynamics
    never reach equilibrium, so the EqProp contrast is not the gradient.
    """
    Lm, Wm, gamma, beta = 3, 12, 0.3, 0.02
    spec = make_hpc_spec(Lm, Wm, 0.3, 0.05)

    def dscan(pcW, u, nudge=None, b=0.0):
        params = {"W": pcW}
        z0 = jnp.zeros(Lm * Wm)
        if nudge is None:
            nudge = jnp.zeros((SEQ, Lm * Wm))

        def step(c, inp):
            phi, pi = c
            uu, n = inp
            phi13 = phi + 0.5 * spec.grad_T(pi, params)
            pi23 = pi - spec.grad_V(phi13, uu, params)
            phi1 = phi13 + 0.5 * spec.grad_T(pi23, params)
            pi1 = jnp.exp(-gamma) * pi23 + b * n
            return (phi1, pi1), phi1
        (_, _), phis = jax.lax.scan(step, (z0, z0), (u, nudge))
        return phis

    def topfeat(phis):
        z = phis.reshape(SEQ, Lm, Wm)
        return jnp.mean(z[:, -1, :] ** 2, 0)

    ks = jr.split(key, Lm + 2)
    pcW = [jr.uniform(ks[0], (CH, Wm), minval=-1, maxval=1) / jnp.sqrt(Wm)]
    pcW += [jr.uniform(ks[l], (Wm, Wm), minval=-1, maxval=1) / jnp.sqrt(Wm) for l in range(1, Lm)]
    pcW = [w.astype(jnp.float64) for w in pcW]
    Wout = (jr.uniform(ks[-2], (CLS, Wm), minval=-1, maxval=1) / jnp.sqrt(Wm)).astype(jnp.float64)
    bout = jnp.zeros(CLS)
    u = jr.normal(jr.fold_in(key, 1), (SEQ, CH), dtype=jnp.float64)
    y = jax.nn.one_hot(1, CLS, dtype=jnp.float64)

    def loss_of_W(pcW):
        return -jnp.sum(y * jax.nn.log_softmax(Wout @ topfeat(dscan(pcW, u)) + bout))
    tg = jax.grad(loss_of_W)(pcW)

    phis_f = dscan(pcW, u)

    def head(phis):
        return -jnp.sum(y * jax.nn.log_softmax(Wout @ topfeat(phis) + bout))
    nudge = -jax.grad(head)(phis_f)
    phis_n = dscan(pcW, u, nudge=nudge, b=beta)

    def hebb(phis):
        def per(phi, uu):
            errs, z = pc_errors(phi, uu, {"W": pcW}, Lm, Wm)
            return [jnp.outer(errs[l], jnp.tanh(z[l])) for l in range(Lm)]
        t = jax.vmap(per)(phis, u)
        return [jnp.sum(x, 0) for x in t]
    hf, hn = hebb(phis_f), hebb(phis_n)
    eg = [(a - b) / beta for a, b in zip(hf, hn)]
    flat = lambda g: jnp.concatenate([jnp.ravel(x) for x in g])
    a, b = flat(eg), flat(tg)
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def ar_data(n, seed):
    """Class = temporal autocorrelation; sequences normalized so only the
    temporal structure (not variance) is discriminative."""
    rng = np.random.default_rng(seed)
    a = np.array([0.0, 0.4, 0.7, 0.9])
    lab = rng.integers(0, CLS, n)
    X = np.zeros((n, SEQ, CH), np.float32)
    for i in range(n):
        x = np.zeros((SEQ, CH))
        for k in range(1, SEQ):
            x[k] = a[lab[i]] * x[k - 1] + rng.normal(0, 1, CH)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        X[i] = x
    return X.astype(np.float32), np.eye(CLS, dtype=np.float32)[lab]


def residual_and_fidelity(p, u, y, n_steps):
    """Per-step inference residual ||grad F|| and EqProp-vs-autodiff cosine."""
    # residual: gradient of F at the relaxed state, averaged over the sequence
    zs = T.run_sequence(p, u, RHO, ALPHA, n_steps, ETA)
    zprev = jnp.concatenate([jnp.zeros_like(zs[:1]), zs[:-1]], 0)
    gF = jax.vmap(lambda z, zp, uu: jax.grad(
        lambda zf: T.free_energy(zf.reshape(L, WD), zp, uu, p, RHO, ALPHA))(z.reshape(-1)))(zs, zprev, u)
    resid = float(jnp.sqrt(jnp.mean(jnp.sum(gF ** 2, -1))))

    def loss_of_p(pp):
        return -jnp.sum(y * jax.nn.log_softmax(T.predict(pp, u, RHO, ALPHA, n_steps, ETA)))
    tg = jax.grad(loss_of_p)(p)
    eg, _ = T.eqprop_grads(p, u, y, RHO, ALPHA, n_steps, ETA, BETA)
    flat = lambda g: jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(g)])
    a, b = flat(eg), flat(tg)
    cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    return resid, cos


def train(method, xtr, ytr, xte, yte, n_steps, seed=0, steps=600, lr=0.01, bs=16):
    p = T.init_tpc(jr.PRNGKey(seed), L, WD, CH, CLS)
    p = jax.tree.map(lambda x: x.astype(jnp.float32), p)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-3))
    opt_state = opt.init(p)

    if method == "BPTT":
        @jax.jit
        def upd(p, opt_state, xb, yb):
            def loss(p):
                return jnp.mean(jax.vmap(lambda u, y: -jnp.sum(
                    y * jax.nn.log_softmax(T.predict(p, u, RHO, ALPHA, n_steps, ETA))))(xb, yb))
            l, g = jax.value_and_grad(loss)(p)
            up, opt_state = opt.update(g, opt_state, p)
            return optax.apply_updates(p, up), opt_state, l
    else:
        @jax.jit
        def upd(p, opt_state, xb, yb):
            gs, ls = jax.vmap(lambda u, y: T.eqprop_grads(p, u, y, RHO, ALPHA, n_steps, ETA, BETA))(xb, yb)
            g = jax.tree.map(lambda x: jnp.mean(x, 0), gs)
            up, opt_state = opt.update(g, opt_state, p)
            return optax.apply_updates(p, up), opt_state, jnp.mean(ls)

    rng = np.random.default_rng(seed)
    curve = []
    for it in range(steps):
        idx = rng.integers(0, len(xtr), bs)
        p, opt_state, _ = upd(p, opt_state, jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]))
        if (it + 1) % 75 == 0:
            pr = jnp.argmax(jax.vmap(lambda u: T.predict(p, u, RHO, ALPHA, n_steps, ETA))(jnp.asarray(xte[:300])), -1)
            curve.append({"step": it + 1,
                          "acc": float(jnp.mean(pr == jnp.argmax(jnp.asarray(yte[:300]), -1)))})
    return curve


def main():
    key = jr.PRNGKey(0)
    p = T.init_tpc(key, L, WD, CH, CLS)
    p = jax.tree.map(lambda x: x.astype(jnp.float64), p)
    u = jr.normal(jr.fold_in(key, 1), (SEQ, CH), dtype=jnp.float64)
    y = jax.nn.one_hot(2, CLS, dtype=jnp.float64)

    # (A) sweep relaxation depth (the dissipation knob): residual -> 0
    n_grid = [1, 2, 4, 8, 16, 32, 64]
    resids, coss = [], []
    for ns in n_grid:
        r, c = residual_and_fidelity(p, u, y, ns)
        resids.append(r); coss.append(c)
        print(f"n_steps={ns}: residual ||gradF||={r:.4f}  EqProp-cosine={c:.4f}")

    # (B) local-credit-assignment fidelity: naive damped-momentum design (fails)
    # vs per-timestep equilibrium relaxation (ours, high fidelity)
    momentum_cos = momentum_eqprop_cosine(jr.PRNGKey(3))
    equilibrium_cos = float(np.mean(coss[3:]))  # relaxed regime (>=8 steps)
    print(f"momentum-EqProp cosine={momentum_cos:.3f}   equilibrium-EqProp cosine={equilibrium_cos:.3f}")

    # (C) training: local EqProp vs BPTT
    xtr, ytr = ar_data(700, 0)
    xte, yte = ar_data(300, 1)
    curves = {m: train(m, xtr, ytr, xte, yte, n_steps=20) for m in ("EqProp", "BPTT")}
    best = {m: max(pt["acc"] for pt in curves[m]) for m in curves}
    print(f"best acc: EqProp(local) {best['EqProp']:.3f}  BPTT {best['BPTT']:.3f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    axs[0].loglog(n_grid, resids, "o-", color="#2a7")
    axs[0].set_xlabel("relaxation steps (dissipation)"); axs[0].set_ylabel(r"inference residual $\|\nabla F\|$")
    axs[0].set_title("(A) Controlled dissipation:\nrelaxation reaches equilibrium")
    axs[0].grid(True, which="both", ls="--", alpha=0.3)

    axs[1].bar([0, 1], [momentum_cos, equilibrium_cos],
               color=["#999", "#2a7"], edgecolor="black")
    axs[1].axhline(1.0, color="gray", ls=":", label="exact gradient")
    axs[1].axhline(0.0, color="black", lw=0.8)
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(["damped momentum\n(naive dissipative)", "per-step equilibrium\n(relaxation, ours)"])
    axs[1].set_ylim(-0.7, 1.1)
    axs[1].set_ylabel("cosine(local rule, true grad)")
    axs[1].set_title("(B) Local credit assignment needs\nequilibrium inference")
    axs[1].legend(); axs[1].grid(True, axis="y", ls="--", alpha=0.3)
    for i, v in enumerate([momentum_cos, equilibrium_cos]):
        axs[1].text(i, v + (0.05 if v > 0 else -0.1), f"{v:.2f}", ha="center")

    for m, c in (("BPTT", "#4c72b0"), ("EqProp", "#cc4444")):
        cv = curves[m]
        axs[2].plot([pt["step"] for pt in cv], [pt["acc"] * 100 for pt in cv], "o-", ms=3,
                    color=c, label=f"{m} (best {best[m]*100:.0f}%)")
    axs[2].axhline(100 / CLS, color="gray", ls=":", label="chance")
    axs[2].set_xlabel("training step"); axs[2].set_ylabel("test accuracy (%)")
    axs[2].set_title("(C) Local dissipative PC trains,\nbut does not beat BPTT")
    axs[2].legend(); axs[2].grid(True, ls="--", alpha=0.3)

    fig.suptitle("Dissipative temporal predictive coding: locality + dissipation, "
                 "at the cost of exactness vs BPTT")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp9_dissipative_pc.png", dpi=160)
    os.makedirs("results", exist_ok=True)
    with open("results/exp9_dissipative_pc.json", "w") as f:
        json.dump({"n_steps": n_grid, "residual": resids, "eqprop_cosine": coss,
                   "momentum_cosine": momentum_cos, "equilibrium_cosine": equilibrium_cos,
                   "best_acc": best,
                   "curves": {m: curves[m] for m in curves}}, f, indent=2)
    print("saved figures/exp9_dissipative_pc.png")


if __name__ == "__main__":
    main()
