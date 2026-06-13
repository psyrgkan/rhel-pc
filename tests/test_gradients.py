"""Numerical checks of the RHEL theory.

1. Time-reversal symmetry (Corollary A.6): an un-nudged echo pass exactly
   retraces the forward trajectory backwards (up to float error).
2. Theorem 3.2: RHEL gradients of a single HRU match BPTT gradients.
3. Theorem 3.3: RHEL chained with autodiff through a deep HSSM matches
   end-to-end BPTT gradients.

All checks run in float64 with small nudging eps.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest

from rhel.hamiltonian import hru_scan, make_hru, rhel_backward
from rhel.models import (
    HSSM,
    cross_entropy_loss,
    init_hssm,
    init_linear_hru,
    init_nonlinear_hru,
    make_linear_spec,
    make_nonlinear_spec,
)

K, SSM, H = 60, 8, 5


def _flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(tree)])


def _cos(a, b):
    a, b = _flat(a), _flat(b)
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


def _setup(model_type):
    key = jr.PRNGKey(0)
    ku, kp = jr.split(key)
    u = jr.normal(ku, (K, H))
    if model_type == "linear":
        spec = make_linear_spec()
        params = init_linear_hru(kp, SSM, H)["ssm"]
    else:
        spec = make_nonlinear_spec(0.1)
        params = init_nonlinear_hru(kp, SSM, H)["ssm"]
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)
    return spec, params, u.astype(jnp.float64)


@pytest.mark.parametrize("model_type", ["linear", "nonlinear"])
def test_time_reversal_symmetry(model_type):
    spec, params, u = _setup(model_type)
    z = jnp.zeros(SSM)
    (phiK, piK), traj = hru_scan(spec, params, z, z, u)
    phis = jnp.concatenate([z[None], traj[0]])  # states 0..K
    pis = jnp.concatenate([z[None], traj[1]])
    # echo with eps = 0: start from conjugated final state, reversed inputs
    (phiE, piE), etraj = hru_scan(spec, params, phiK, -piK, u[::-1])
    ephis = etraj[0]
    # Phi^e_j should equal Sigma_z Phi_{K-j}
    assert jnp.max(jnp.abs(ephis - phis[::-1][1:])) < 1e-10
    assert jnp.max(jnp.abs(phiE - phis[0])) < 1e-10
    assert jnp.max(jnp.abs(piE + pis[0])) < 1e-10


@pytest.mark.parametrize("model_type", ["linear", "nonlinear"])
def test_single_hru_rhel_matches_bptt(model_type):
    spec, params, u = _setup(model_type)
    kw = jr.PRNGKey(1)
    w = jr.normal(kw, (K, SSM))  # random readout so the loss touches all steps

    def loss(apply, p, uu):
        phis = apply(p, uu, 1e-6)
        return jnp.sum(w * jnp.tanh(phis))

    bptt = make_hru(spec, SSM, "BPTT")
    rhel = make_hru(spec, SSM, "RHEL")
    g_bptt = jax.grad(lambda p, uu: loss(bptt, p, uu), argnums=(0, 1))(params, u)
    g_rhel = jax.grad(lambda p, uu: loss(rhel, p, uu), argnums=(0, 1))(params, u)

    for gb, gr in zip(g_bptt, g_rhel):
        c = _cos(gb, gr)
        r = jnp.linalg.norm(_flat(gr)) / jnp.linalg.norm(_flat(gb))
        assert c > 1 - 1e-8, f"cosine {c}"
        assert abs(r - 1) < 1e-5, f"norm ratio {r}"


@pytest.mark.parametrize("model_type", ["linear", "nonlinear"])
def test_hssm_rhel_matches_bptt(model_type):
    key = jr.PRNGKey(2)
    ku, kp, ky = jr.split(key, 3)
    u = jr.normal(ku, (K, 3), dtype=jnp.float64)
    y = jax.nn.one_hot(jr.randint(ky, (), 0, 4), 4, dtype=jnp.float64)
    params = init_hssm(kp, model_type, data_dim=3, label_dim=4, hidden_dim=H,
                       ssm_dim=SSM, num_blocks=3)
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)

    models = {
        alg: HSSM(model_type, SSM, learning_algorithm=alg,
                  epsilon=1e-6, nonlinear_step=0.1)
        for alg in ("BPTT", "RHEL")
    }
    grads = {
        alg: jax.grad(lambda p: cross_entropy_loss(m, p, u[None], y[None]))(params)
        for alg, m in models.items()
    }
    c = _cos(grads["BPTT"], grads["RHEL"])
    r = jnp.linalg.norm(_flat(grads["RHEL"])) / jnp.linalg.norm(_flat(grads["BPTT"]))
    assert c > 1 - 1e-7, f"cosine {c}"
    assert abs(r - 1) < 1e-4, f"norm ratio {r}"
