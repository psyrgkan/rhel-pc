"""Tests for the dissipative temporal predictive coding model."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from rhel import temporal_pc as T

L, WD, CH, CLS, SEQ = 2, 10, 3, 4, 20
RHO, ALPHA, NST, ETA = 1.0, 0.1, 25, 0.2


def _flat(g):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(g)])


def test_relaxation_reaches_equilibrium():
    """Inference residual ||grad F|| shrinks as relaxation steps grow."""
    key = jr.PRNGKey(0)
    p = jax.tree.map(lambda x: x.astype(jnp.float64), T.init_tpc(key, L, WD, CH, CLS))
    u = jr.normal(jr.fold_in(key, 1), (CH,), dtype=jnp.float64)
    zprev = jnp.zeros((L, WD))

    def residual(n):
        z = T.relax(zprev, u, p, RHO, ALPHA, n, ETA)
        g = jax.grad(lambda zf: T.free_energy(zf.reshape(L, WD), zprev, u, p, RHO, ALPHA))(z.reshape(-1))
        return jnp.linalg.norm(g)

    assert residual(40) < 0.1 * residual(2)


def test_eqprop_gradient_matches_autodiff():
    """The local per-step-equilibrium EqProp gradient is a high-fidelity proxy
    of the true (autodiff) gradient through the relaxation dynamics."""
    key = jr.PRNGKey(1)
    p = jax.tree.map(lambda x: x.astype(jnp.float64), T.init_tpc(key, L, WD, CH, CLS))
    u = jr.normal(jr.fold_in(key, 1), (SEQ, CH), dtype=jnp.float64)
    y = jax.nn.one_hot(2, CLS, dtype=jnp.float64)

    tg = jax.grad(lambda pp: -jnp.sum(
        y * jax.nn.log_softmax(T.predict(pp, u, RHO, ALPHA, NST, ETA))))(p)
    eg, _ = T.eqprop_grads(p, u, y, RHO, ALPHA, NST, ETA, beta=0.01)
    a, b = _flat(eg), _flat(tg)
    cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    assert cos > 0.9, f"EqProp/autodiff cosine too low: {cos}"
