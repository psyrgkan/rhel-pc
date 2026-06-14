"""Hamiltonian SSMs (HSSMs) built from linear and nonlinear HRUs.

Architecture follows App. A.4.2 (Algorithm 2) of the paper, which itself
follows LinOSS (Rusch & Rus, 2024):

    u -> affine encoder -> N x [ norm -> HRU -> C/D readout -> GELU -> GLU
                                 -> residual ] -> affine decoder

The recurrent core of each block is an HRU (rhel.hamiltonian) trained
either by BPTT (plain autodiff through the leapfrog scan) or by RHEL
(custom VJP with echo passes).  Everything outside the HRUs (encoders,
GLUs, readouts) is differentiated by ordinary autodiff in both cases,
exactly as prescribed by Remark 8 / App. A.4.7: RHEL chains with AD at HRU
boundaries.

Deviations from the official implementation (documented in
docs/replication_report.md): LayerNorm instead of BatchNorm, no dropout,
real-valued (not complex) linear SSM parameters.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from .hamiltonian import HamiltonianSpec, make_hru

# ---------------------------------------------------------------------------
# Hamiltonians (reparametrized to fold the time step in, App. A.4.6)
# ---------------------------------------------------------------------------


def _logcosh(x):
    # log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2), numerically stable
    return jnp.abs(x) + jnp.log1p(jnp.exp(-2.0 * jnp.abs(x))) - jnp.log(2.0)


def linear_kinetic(pi, p):
    delta = jax.nn.sigmoid(p["delta_raw"])
    return 0.5 * jnp.sum(delta * pi**2)


def linear_potential(phi, u, p):
    """V = 1/2 phi^T (delta A) phi - phi^T (delta B) u  (Eq. 36 + Eq. 52)."""
    delta = jax.nn.sigmoid(p["delta_raw"])
    A = jax.nn.relu(p["A_raw"])
    Bu = p["B"] @ u
    return 0.5 * jnp.sum(delta * A * phi**2) - jnp.sum(delta * Bu * phi)


def nonlinear_kinetic(step):
    def kinetic(pi, p):
        return 0.5 * step * jnp.sum(pi**2)

    return kinetic


def nonlinear_potential(step):
    """V = alpha/2 |phi|^2 + sum_i c_i log cosh(W_i phi_i + (Bu)_i + b_i)/W_i.

    Eq. 48 with the trainable per-unit scale c = sigmoid(c_raw) of Eq. 55
    (UnICORNN-style trainable effective time step on the nonlinear term).
    The global integrator step is folded in as a fixed scalar `step`.
    """

    def potential(phi, u, p):
        c = jax.nn.sigmoid(p["c_raw"])
        W = p["W_diag"]
        Bu = p["B"] @ u
        quad = 0.5 * p["alpha"][0] * jnp.sum(phi**2)
        nl = jnp.sum(c * _logcosh(W * phi + Bu + p["b"]) / W)
        return step * (quad + nl)

    return potential


def make_linear_spec():
    return HamiltonianSpec(kinetic=linear_kinetic, potential=linear_potential)


def make_nonlinear_spec(step):
    return HamiltonianSpec(kinetic=nonlinear_kinetic(step), potential=nonlinear_potential(step))


# ---------------------------------------------------------------------------
# Parameter initialization (App. A.5.3)
# ---------------------------------------------------------------------------


def _uniform(key, shape, lo, hi):
    return jr.uniform(key, shape, minval=lo, maxval=hi)


def init_linear_hru(key, ssm_dim, hidden_dim):
    kA, kB, kC, kD, kd = jr.split(key, 5)
    s = 1.0 / jnp.sqrt(hidden_dim)
    return {
        "ssm": {
            "A_raw": _uniform(kA, (ssm_dim,), 0.0, 1.0),
            "B": _uniform(kB, (ssm_dim, hidden_dim), -s, s),
            "delta_raw": _uniform(kd, (ssm_dim,), 0.0, 1.0),
        },
        "C": _uniform(kC, (hidden_dim, ssm_dim), -1.0 / jnp.sqrt(ssm_dim), 1.0 / jnp.sqrt(ssm_dim)),
        "D": jr.normal(kD, (hidden_dim,)),
    }


def init_nonlinear_hru(key, ssm_dim, hidden_dim):
    kW, kB, kC, kD, kc, ka = jr.split(key, 6)
    s = 1.0 / jnp.sqrt(hidden_dim)
    return {
        "ssm": {
            "W_diag": _uniform(kW, (ssm_dim,), 0.5, 1.0),
            "B": _uniform(kB, (ssm_dim, hidden_dim), -s, s),
            "b": jnp.zeros((ssm_dim,)),
            "c_raw": _uniform(kc, (ssm_dim,), -1.0, 1.0),
            "alpha": _uniform(ka, (1,), 0.1, 1.0),
        },
        "C": _uniform(kC, (hidden_dim, ssm_dim), -1.0 / jnp.sqrt(ssm_dim), 1.0 / jnp.sqrt(ssm_dim)),
        "D": jr.normal(kD, (hidden_dim,)),
    }


def init_glu(key, dim):
    k1, k2 = jr.split(key)
    s = 1.0 / jnp.sqrt(dim)
    return {
        "w1": _uniform(k1, (dim, dim), -s, s),
        "b1": jnp.zeros((dim,)),
        "w2": _uniform(k2, (dim, dim), -s, s),
        "b2": jnp.zeros((dim,)),
    }


def init_hssm(key, model_type, data_dim, label_dim, hidden_dim, ssm_dim, num_blocks):
    keys = jr.split(key, num_blocks + 2)
    init_block = init_linear_hru if model_type == "linear" else init_nonlinear_hru
    kenc, kdec = keys[-2:]
    s = 1.0 / jnp.sqrt(data_dim)
    blocks = []
    for kb in keys[:num_blocks]:
        kh, kg = jr.split(kb)
        blocks.append({"hru": init_block(kh, ssm_dim, hidden_dim), "glu": init_glu(kg, hidden_dim)})
    return {
        "enc_w": _uniform(kenc, (hidden_dim, data_dim), -s, s),
        "enc_b": jnp.zeros((hidden_dim,)),
        "blocks": blocks,
        "dec_w": _uniform(kdec, (label_dim, hidden_dim), -1.0 / jnp.sqrt(hidden_dim), 1.0 / jnp.sqrt(hidden_dim)),
        "dec_b": jnp.zeros((label_dim,)),
    }


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def glu_apply(p, x):
    return (x @ p["w1"].T + p["b1"]) * jax.nn.sigmoid(x @ p["w2"].T + p["b2"])


def layer_norm(x, eps=1e-6):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps)


class HSSM:
    """Functional HSSM. Holds static configuration; params live in a pytree."""

    def __init__(self, model_type, ssm_dim, learning_algorithm="BPTT", epsilon=0.0,
                 nonlinear_step=0.05, classification=True):
        self.model_type = model_type
        self.classification = classification
        self.epsilon = epsilon
        spec = make_linear_spec() if model_type == "linear" else make_nonlinear_spec(nonlinear_step)
        self.spec = spec
        self.hru_apply = make_hru(spec, ssm_dim, learning_algorithm)

    def block(self, p, x_seq):
        skip = x_seq
        x = layer_norm(x_seq)
        phis = self.hru_apply(p["hru"]["ssm"], x, self.epsilon)  # (K, ssm_dim)
        y = phis @ p["hru"]["C"].T + x * p["hru"]["D"]
        y = jax.nn.gelu(y)
        y = glu_apply(p["glu"], y)
        return skip + y

    def __call__(self, params, u_seq):
        x = u_seq @ params["enc_w"].T + params["enc_b"]
        for p in params["blocks"]:
            x = self.block(p, x)
        if self.classification:
            x = jnp.mean(x, axis=0)
            return x @ params["dec_w"].T + params["dec_b"]  # logits
        return x @ params["dec_w"].T + params["dec_b"]  # (K, label_dim)


def cross_entropy_loss(model, params, u_batch, y_batch, grad_scaler=1.0):
    logits = jax.vmap(lambda u: model(params, u))(u_batch)
    ll = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(y_batch * ll, axis=-1)) * grad_scaler


def mse_loss(model, params, u_batch, y_batch, grad_scaler=1.0):
    pred = jax.vmap(lambda u: model(params, u))(u_batch)
    return jnp.mean((pred - y_batch) ** 2) * grad_scaler
