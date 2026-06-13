"""Tests for Hamiltonian Predictive Coding (benefits #1 and #2)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from rhel.models import cross_entropy_loss
from rhel.predictive_coding import (
    HPCClassifier,
    hpc_reconstruction_loss,
    init_hpc,
    pc_local_hebbian_update,
)

LEVELS, WIDTH, K = 3, 12, 50


def _flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(tree)])


def _cos(a, b):
    a, b = _flat(a), _flat(b)
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def _setup(seed=0):
    key = jr.PRNGKey(seed)
    params = init_hpc(key, LEVELS, WIDTH, 4, 4)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
    u = jr.normal(jr.fold_in(key, 1), (K, 4), dtype=jnp.float64)
    y = jax.nn.one_hot(0, 4, dtype=jnp.float64)
    return params, u, y


def test_locality_hebbian_equals_rhel_gradient():
    """Benefit #1: the local error x activity update == the RHEL gradient."""
    params, u, y = _setup()
    model = HPCClassifier(LEVELS, WIDTH, 0.2, 0.1, "RHEL", epsilon=1e-6)
    grads = jax.grad(lambda p: cross_entropy_loss(model, p, u[None], y[None]))(params)

    def head_loss(phis_flat):
        feats = jnp.tanh(phis_flat @ params["read_w"].T + params["read_b"])
        logits = jnp.mean(feats, 0) @ params["dec_w"].T + params["dec_b"]
        return -jnp.sum(y * jax.nn.log_softmax(logits))

    phis = model.hru_apply(params["pc"], u, 0.0)
    g_seq = jax.grad(head_loss)(phis.reshape(K, -1))
    dW = pc_local_hebbian_update(model, params, u, g_seq, 1e-6)
    for l in range(LEVELS):
        diff = jnp.max(jnp.abs(dW[l] - grads["pc"]["W"][l])) / (
            jnp.max(jnp.abs(grads["pc"]["W"][l])) + 1e-30)
        assert diff < 1e-6, f"layer {l} rel diff {diff}"


def test_unsupervised_rhel_matches_bptt():
    """Benefit #2: RHEL == BPTT on the unsupervised reconstruction objective."""
    params, u, _ = _setup(1)
    u_batch = u[None]
    grads = {}
    for alg in ("BPTT", "RHEL"):
        model = HPCClassifier(LEVELS, WIDTH, 0.5, 0.05, alg, epsilon=1e-6)
        grads[alg] = jax.grad(
            lambda p: hpc_reconstruction_loss(model, p, u_batch))(params)
    assert _cos(grads["BPTT"], grads["RHEL"]) > 1 - 1e-7
    ratio = jnp.linalg.norm(_flat(grads["RHEL"])) / jnp.linalg.norm(_flat(grads["BPTT"]))
    assert abs(ratio - 1) < 1e-4
