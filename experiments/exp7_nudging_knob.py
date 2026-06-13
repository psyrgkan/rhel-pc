"""Experiment 7 — The nudging knob (Figure 5): exact <-> approximate.

RHEL's only approximation is the finite nudge strength epsilon. This makes
RHEL a one-parameter family that interpolates between an EXACT proxy of BPTT
(epsilon -> 0) and a biased, but numerically robust, contrastive estimator
(the regime shared with Equilibrium Propagation). Two things are shown:

  (A) Estimator bias vs epsilon (float64, isolating the linearization bias).
      RHEL uses a *centered* two-echo difference (+/- epsilon); we compare it
      against a *one-sided* nudge (the standard EqProp estimator, free + one
      nudged phase). The centered estimator has O(eps^2) bias, the one-sided
      O(eps), so RHEL's symmetric echo buys ~one extra order of accuracy --
      a concrete reason to flip the momenta both ways.

  (B) The practical window (float32). Small epsilon underflows (the eps-
      perturbation is lost to rounding, App. A.5.4); large epsilon incurs
      linearization bias. The relative gradient error is therefore U-shaped
      in epsilon, with a sweet spot -- the value the implementation must use.

Run:  python experiments/exp7_nudging_knob.py
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

from rhel.hamiltonian import hru_scan
from rhel.models import cross_entropy_loss
from rhel.predictive_coding import HPCClassifier, init_hpc, pc_errors

LEVELS, WIDTH, DELTA, ALPHA = 3, 24, 0.2, 0.1
SEQ_LEN, CHANNELS, CLASSES = 100, 4, 4


def setup(dtype, seed=0):
    key = jr.PRNGKey(seed)
    params = init_hpc(key, LEVELS, WIDTH, CHANNELS, CLASSES)
    params = jax.tree.map(lambda x: x.astype(dtype), params)
    u = jr.normal(jr.fold_in(key, 1), (SEQ_LEN, CHANNELS), dtype=dtype)
    y = jax.nn.one_hot(jr.randint(jr.fold_in(key, 2), (), 0, CLASSES), CLASSES, dtype=dtype)
    return params, u, y


def upstream_grad(model, params, u, y):
    def head_loss(phis_flat):
        feats = jnp.tanh(phis_flat @ params["read_w"].T + params["read_b"])
        logits = jnp.mean(feats, 0) @ params["dec_w"].T + params["dec_b"]
        return -jnp.sum(y * jax.nn.log_softmax(logits))
    phis = model.hru_apply(params["pc"], u, 0.0)
    return jax.grad(head_loss)(phis.reshape(SEQ_LEN, -1))


def hebb_sum(model, params, phi13s, u_rev):
    """Sum of local error x activity outer products over the echo trajectory."""
    p = params["pc"]
    def per_step(phi13, u):
        errs, z = pc_errors(phi13, u, p, model.levels, model.width)
        return [jnp.outer(errs[l], jnp.tanh(z[l])) for l in range(model.levels)]
    terms = jax.vmap(per_step)(phi13s, u_rev)
    return [jnp.sum(t, 0) for t in terms]


def echo_half(model, params, phiK, piK, u_rev, n0, n_seq, e):
    p = params["pc"]
    _, traj = hru_scan(model.spec, p, phiK, -piK + e * n0, u_rev, n_seq, e)
    return traj[2]  # phi_{k+1/3}


def rhel_grad(model, params, u, y, eps, scheme):
    """Estimate dL/dW via echo passes; 'centered' (RHEL) or 'one_sided'."""
    eps = jnp.asarray(eps, dtype=u.dtype)  # keep precision of the data
    g_seq = upstream_grad(model, params, u, y)
    p = params["pc"]
    z0 = jnp.zeros(LEVELS * WIDTH, dtype=u.dtype)
    (phiK, piK), _ = hru_scan(model.spec, p, z0, z0, u)
    g_rev = g_seq[::-1]
    n0 = g_rev[0]
    n_seq = jnp.concatenate([g_rev[1:], jnp.zeros_like(g_rev[:1])])
    u_rev = u[::-1]
    hp = hebb_sum(model, params, echo_half(model, params, phiK, piK, u_rev, n0, n_seq, eps), u_rev)
    if scheme == "centered":
        hm = hebb_sum(model, params, echo_half(model, params, phiK, piK, u_rev, n0, n_seq, -eps), u_rev)
        return [model.delta / (2 * eps) * (a - b) for a, b in zip(hp, hm)]
    h0 = hebb_sum(model, params, echo_half(model, params, phiK, piK, u_rev, n0, n_seq, 0.0), u_rev)
    return [model.delta / eps * (a - b) for a, b in zip(hp, h0)]


def rel_error(est, ref):
    e = jnp.concatenate([jnp.ravel(a - b) for a, b in zip(est, ref)])
    r = jnp.concatenate([jnp.ravel(b) for b in ref])
    return float(jnp.linalg.norm(e) / jnp.linalg.norm(r))


def main():
    # reference BPTT gradient w.r.t. the PC weights (float64)
    params64, u64, y64 = setup(jnp.float64)
    model64 = HPCClassifier(LEVELS, WIDTH, DELTA, ALPHA, "BPTT", epsilon=0.0)
    g_bptt = jax.grad(lambda p: cross_entropy_loss(model64, p, u64[None], y64[None]))(params64)
    ref = [g_bptt["pc"]["W"][l] for l in range(LEVELS)]

    # (A) float64 bias vs eps: centered (RHEL) vs one-sided (EqProp)
    eps64 = np.logspace(-6, 0.0, 16)
    err_centered = [rel_error(rhel_grad(model64, params64, u64, y64, e, "centered"), ref) for e in eps64]
    err_onesided = [rel_error(rhel_grad(model64, params64, u64, y64, e, "one_sided"), ref) for e in eps64]

    # (B) float32 practical window (underflow + linearization). Reference is a
    # genuine float32 BPTT gradient so the curve isolates finite-precision
    # effects of the *estimator*, not a precision mismatch with the reference.
    params32, u32, y32 = setup(jnp.float32)
    model32 = HPCClassifier(LEVELS, WIDTH, DELTA, ALPHA, "BPTT", epsilon=0.0)
    g_bptt32 = jax.grad(lambda p: cross_entropy_loss(model32, p, u32[None], y32[None]))(params32)
    ref32 = [g_bptt32["pc"]["W"][l] for l in range(LEVELS)]
    eps32 = np.logspace(-6, 0.0, 16)
    err_f32 = [rel_error(rhel_grad(model32, params32, u32, y32, np.float32(e), "centered"), ref32)
               for e in eps32]

    results = {
        "eps": eps64.tolist(),
        "rel_err_centered_f64": err_centered,
        "rel_err_one_sided_f64": err_onesided,
        "rel_err_centered_f32": err_f32,
        "best_eps_f32": float(eps32[int(np.argmin(err_f32))]),
        "best_rel_err_f32": float(np.min(err_f32)),
    }
    print(json.dumps({k: v for k, v in results.items() if "rel_err" not in k}, indent=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.loglog(eps64, err_centered, "o-", color="#cc4444", label="centered (RHEL, two echoes)")
    ax1.loglog(eps64, err_onesided, "s-", color="#4c72b0", label="one-sided (EqProp, one echo)")
    # reference slopes
    ax1.loglog(eps64, 0.5 * (eps64 / eps64[-1]) ** 2 * err_centered[-1] / 0.5,
               "k:", lw=1, alpha=0.6, label=r"$O(\epsilon^2)$")
    ax1.loglog(eps64, (eps64 / eps64[-1]) * err_onesided[-1],
               "k--", lw=1, alpha=0.6, label=r"$O(\epsilon)$")
    ax1.set_xlabel(r"nudge strength $\epsilon$")
    ax1.set_ylabel("relative gradient error vs BPTT")
    ax1.set_title("(A) Centered echo halves the order of the bias\n(float64)")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2.loglog(eps32, err_f32, "o-", color="#cc4444", label="RHEL (centered), float32")
    best = results["best_eps_f32"]
    ax2.axvline(best, color="gray", ls=":", label=fr"sweet spot $\epsilon\approx{best:.0e}$")
    ax2.set_xlabel(r"nudge strength $\epsilon$")
    ax2.set_ylabel("relative gradient error vs BPTT")
    ax2.set_title("(B) Practical window in float32\nunderflow (small $\\epsilon$) vs bias (large $\\epsilon$)")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.annotate("underflow", (eps32[1], err_f32[1]), fontsize=8, color="#555")
    ax2.annotate("linearization", (eps32[-2], err_f32[-2]), fontsize=8, color="#555", ha="right")

    fig.suptitle("The nudging knob: RHEL interpolates between an exact BPTT proxy "
                 "($\\epsilon\\to0$) and a robust contrastive estimator")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp7_nudging_knob.png", dpi=160)
    os.makedirs("results", exist_ok=True)
    with open("results/exp7_nudging_knob.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved figures/exp7_nudging_knob.png")


if __name__ == "__main__":
    main()
