"""Temporal predictive coding with equilibrium inference and local learning.

This module builds the dissipative counterpart of Hamiltonian Predictive
Coding, addressing the open problem from the negative result in exp8: how to
keep predictive coding's *local* learning while using *dissipation* to gain
equilibrium inference (the "prospective configuration" regime).

Key facts established empirically (exp9):
  * Equilibrium Propagation (EqProp) computes the exact gradient at a STATIC
    equilibrium (cosine -> 1 as the nudge beta -> 0).
  * EqProp on a driven, non-settling temporal trajectory does NOT (it is
    anti-correlated with the true gradient): exact local credit assignment
    requires the dynamics to relax to equilibrium.

We therefore use **per-timestep equilibrium inference**: at each step the
latents relax to the minimum of a local free energy (input + temporal
prediction), where EqProp is exact and local; temporal memory is carried by
learned transition weights V. Relaxation IS the (controlled) dissipation:
zero steps recovers a feedforward pass, many steps reaches the PC equilibrium.

Free energy at step k, with previous latents z^- held fixed:
    F_k = 1/2||u_k - W0 f(z0)||^2
        + sum_{l>=1} 1/2||z_{l-1} - W_l f(z_l)||^2          (hierarchical)
        + rho * sum_l 1/2||z_l - V_l f(z_l^-)||^2           (temporal)
        + alpha/2 ||z||^2                                    (leak)

All weights are learned by LOCAL rules: W_l and V_l by (error x presynaptic
activity), as in classical predictive coding. The cost relative to RHEL is
that inference uses W_l^T / V_l^T to propagate errors (weight symmetry), so
this regime keeps locality of plasticity but not RHEL's transport-freedom.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr


def _f(x):
    return jnp.tanh(x)


def free_energy(z, zprev, u, p, rho, alpha):
    """z, zprev: (L, W); u: (CH,). Returns scalar local free energy."""
    L = z.shape[0]
    e0 = u - p["W"][0] @ _f(z[0])
    F = 0.5 * jnp.sum(e0 ** 2)
    for l in range(1, L):
        e = z[l - 1] - p["W"][l] @ _f(z[l])
        F += 0.5 * jnp.sum(e ** 2)
    # temporal prediction errors (each level predicted from its own past)
    et = z - jax.vmap(lambda Vl, zp: Vl @ _f(zp))(p["V"], zprev)
    F += rho * 0.5 * jnp.sum(et ** 2)
    F += 0.5 * alpha * jnp.sum(z ** 2)
    return F


def relax(zprev, u, p, rho, alpha, n_steps, eta, beta=0.0, nudge_fn=None):
    """Relax latents to the per-step equilibrium by gradient descent on F.

    nudge_fn(z_top) -> scalar adds beta * nudge_fn to F (EqProp nudged phase).
    Returns the relaxed (L, W) latents.
    """
    gradF = jax.grad(lambda zf: free_energy(zf.reshape(zprev.shape), zprev, u, p, rho, alpha)
                     + (beta * nudge_fn(zf.reshape(zprev.shape)[-1]) if nudge_fn is not None else 0.0))

    def body(zf, _):
        return zf - eta * gradF(zf), None

    z0 = zprev.reshape(-1)  # warm-start from previous timestep
    zf, _ = jax.lax.scan(body, z0, None, length=n_steps)
    return zf.reshape(zprev.shape)


def run_sequence(p, u_seq, rho, alpha, n_steps, eta, beta=0.0, nudge_fn=None):
    """Per-timestep equilibrium inference over a sequence. Returns (K, L, W)."""
    L, Wd = p["V"].shape[0], p["V"].shape[1]

    def step(zprev, u):
        z = relax(zprev, u, p, rho, alpha, n_steps, eta, beta, nudge_fn)
        return z, z

    z0 = jnp.zeros((L, Wd))
    _, zs = jax.lax.scan(step, z0, u_seq)
    return zs


def init_tpc(key, L, Wd, data_dim, label_dim):
    ks = jr.split(key, L + 3)
    s = 1.0 / jnp.sqrt(Wd)
    W = [jr.uniform(ks[0], (data_dim, Wd), minval=-s, maxval=s)]
    W += [jr.uniform(ks[l], (Wd, Wd), minval=-s, maxval=s) for l in range(1, L)]
    V = jr.uniform(ks[L], (L, Wd, Wd), minval=-s, maxval=s)
    return {
        "W": W, "V": V,
        "Wout": jr.uniform(ks[L + 1], (label_dim, Wd), minval=-s, maxval=s),
        "bout": jnp.zeros((label_dim,)),
    }


def readout(p, zs):
    """Power features of the top latent, pooled over time -> logits."""
    h = jnp.mean(zs[:, -1, :] ** 2, axis=0)
    return p["Wout"] @ h + p["bout"]


def predict(p, u_seq, rho, alpha, n_steps, eta):
    zs = run_sequence(p, u_seq, rho, alpha, n_steps, eta)
    return readout(p, zs)


def _hebb(p, zs, u_seq, rho):
    """Local error x activity outer products, summed over time, per weight."""
    L = p["V"].shape[0]

    def per_step(z, zprev, u):
        e0 = u - p["W"][0] @ _f(z[0])
        dW = [jnp.outer(e0, _f(z[0]))]
        for l in range(1, L):
            e = z[l - 1] - p["W"][l] @ _f(z[l])
            dW.append(jnp.outer(e, _f(z[l])))
        et = z - jax.vmap(lambda Vl, zp: Vl @ _f(zp))(p["V"], zprev)
        dV = jax.vmap(lambda e, zp: rho * jnp.outer(e, _f(zp)))(et, zprev)
        return dW, dV

    zprev = jnp.concatenate([jnp.zeros_like(zs[:1]), zs[:-1]], axis=0)
    dW, dV = jax.vmap(per_step)(zs, zprev, u_seq)
    return [jnp.sum(w, 0) for w in dW], jnp.sum(dV, 0)


def eqprop_grads(p, u_seq, y, rho, alpha, n_steps, eta, beta):
    """Local EqProp gradient for the temporal PC (per-step equilibria)."""
    zs_f = run_sequence(p, u_seq, rho, alpha, n_steps, eta)
    logits = readout(p, zs_f)
    g = jax.nn.softmax(logits) - y
    loss = -jnp.sum(y * jax.nn.log_softmax(logits))

    def nudge_fn(ztop):
        # output loss as a function of a single top latent (uses Wout^T feedback)
        h = ztop ** 2
        return -jnp.sum(y * jax.nn.log_softmax(p["Wout"] @ h + p["bout"]))

    zs_n = run_sequence(p, u_seq, rho, alpha, n_steps, eta, beta=beta, nudge_fn=nudge_fn)
    hf_W, hf_V = _hebb(p, zs_f, u_seq, rho)
    hn_W, hn_V = _hebb(p, zs_n, u_seq, rho)
    # EqProp: (free - nudged)/beta ~ dL/dtheta  (descent gradient)
    gW = [(a - b) / beta for a, b in zip(hf_W, hn_W)]
    gV = (hf_V - hn_V) / beta
    h_f = jnp.mean(zs_f[:, -1, :] ** 2, axis=0)
    gWout = jnp.outer(g, h_f)
    gbout = g
    return {"W": gW, "V": gV, "Wout": gWout, "bout": gbout}, loss
