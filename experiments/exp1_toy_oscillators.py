"""Experiment 1 — Fig. 2 (B)-(C) replication: six coupled harmonic oscillators.

Checks Theorem 3.1: RHEL's time-resolved estimators equal the adjoint state
method's sensitivities at all times,

    lambda(t)   = lim_{eps->0} Delta_Phi(t, eps)
    g_theta(t)  = lim_{eps->0} Delta_theta(t, eps).

Hamiltonian (Eq. 8):
    H = sum_i pi_i^2 / (2 m_i) + 1/2 sum_i k_i phi_i^2
        + 1/2 sum_{i<j} k_ij (phi_j - phi_i)^2 + u phi_1

The input u(t) drives oscillator 1; oscillator 4 is read out against a
target y(t) with loss l = 1/2 (phi_4 - y)^2. The reference sensitivities
are computed with reverse-mode AD through the (symplectic) leapfrog
discretization, i.e. the discrete adjoint, which converges to the
continuous ASM for small steps.

Run:  python experiments/exp1_toy_oscillators.py
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rhel.hamiltonian import HamiltonianSpec, hru_scan, leapfrog_step

N = 6           # oscillators
DELTA = 0.01    # integrator step
K = 2000        # steps  (T = 20 time units)
EPS = 1e-5      # nudging strength

# --- Hamiltonian of Eq. (8), step folded in (App. A.4.6) -------------------


def kinetic(pi, p):
    return DELTA * jnp.sum(pi**2 / (2.0 * p["m"]))


def potential(phi, u, p):
    pair = phi[None, :] - phi[:, None]          # pair[i, j] = phi_j - phi_i
    mask = jnp.triu(jnp.ones((N, N)), 1)
    coupling = 0.5 * jnp.sum(mask * p["kij"] * pair**2)
    return DELTA * (0.5 * jnp.sum(p["k"] * phi**2) + coupling + u[0] * phi[0])


SPEC = HamiltonianSpec(kinetic=kinetic, potential=potential)


def make_problem(key):
    km, kk, kc = jr.split(key, 3)
    params = {
        "m": jr.uniform(km, (N,), minval=0.9, maxval=1.1),
        "k": jr.uniform(kk, (N,), minval=0.8, maxval=1.2),
        "kij": jr.uniform(kc, (N, N), minval=0.05, maxval=0.15),
    }
    t = jnp.arange(K) * DELTA
    u_seq = jnp.sin(1.5 * t)[:, None]
    y_seq = 0.5 * jnp.sin(0.8 * t + 0.5)
    return params, u_seq, y_seq


def loss_grads(phis, y_seq):
    """dl/dphi_k for l_k = 1/2 (phi_4 - y_k)^2 (only oscillator 4 read out)."""
    g = jnp.zeros_like(phis)
    return g.at[:, 3].set(phis[:, 3] - y_seq)


# --- Reference: discrete adjoint via AD with per-step probes ---------------


def adjoint_reference(params, u_seq, y_seq):
    """lambda_k = dL/dPhi_k and g_theta(k) via zero probes xi_k, zeta_k."""
    zeros_state = jnp.zeros((K, N))
    zetas = jax.tree.map(lambda x: jnp.zeros((K,) + x.shape), params)

    def run(xis_phi, xis_pi, zetas):
        def step(carry, inp):
            phi, pi = carry
            u, xp, xq, zeta = inp
            p_k = jax.tree.map(lambda a, b: a + b, params, zeta)
            phi1, pi23, _ = leapfrog_step(SPEC, p_k, phi, pi, u)
            phi1, pi1 = phi1 + xp, pi23 + xq
            return (phi1, pi1), phi1

        z = jnp.zeros(N)
        _, phis = jax.lax.scan(step, (z, z), (u_seq, xis_phi, xis_pi, zetas))
        return 0.5 * jnp.sum((phis[:, 3] - y_seq) ** 2)

    g = jax.grad(run, argnums=(0, 1, 2))(zeros_state, zeros_state, zetas)
    lam_phi, lam_pi, g_theta = g          # each indexed by forward step k
    return lam_phi, lam_pi, g_theta


# --- RHEL: echo passes with time-resolved readout ---------------------------


def rhel_estimates(params, u_seq, y_seq, eps):
    z = jnp.zeros(N)
    (phiK, piK), traj = hru_scan(SPEC, params, z, z, u_seq)
    phis = traj[0]
    g_seq = loss_grads(phis, y_seq)
    g_rev = g_seq[::-1]
    n0 = g_rev[0]
    n_seq = jnp.concatenate([g_rev[1:], jnp.zeros_like(g_rev[:1])], axis=0)

    def echo(sign):
        e = sign * eps
        pi0 = -piK + e * n0
        (pE, qE), etraj = hru_scan(SPEC, params, phiK, pi0, u_seq[::-1], n_seq, e)
        ephis, epis, phi13s, pi_pres, pi23s = etraj
        # include the (nudged) initial echo state as emission "-1"
        ephis = jnp.concatenate([phiK[None], ephis])
        epis = jnp.concatenate([pi0[None], epis])
        return ephis, epis, (phi13s, pi_pres, pi23s)

    ephi_p, epi_p, h_p = echo(+1.0)
    ephi_m, epi_m, h_m = echo(-1.0)

    # Delta_Phi(j) = Sigma_x (Phi^e_j(+eps) - Phi^e_j(-eps)) / (2 eps).
    # Echo emission j sits at forward state Phi_{K-1-j}; emission -1 (the
    # nudged initial state) carries the final-time sensitivity dL/dPhi_K.
    # Drop the last emission (dL/dPhi_0, the initial-state gradient) so the
    # sequence aligns with the K probed forward states Phi_1..Phi_K.
    dphi = ((epi_p - epi_m) / (2 * eps))[:-1]   # estimate of dL/dphi
    dpi = ((ephi_p - ephi_m) / (2 * eps))[:-1]  # estimate of dL/dpi

    grad_H = jax.grad(SPEC.half_step_hamiltonian, argnums=4)

    def per_step(h):
        return jax.vmap(lambda a, b, c, u: grad_H(a, b, c, u, params))(*h, u_seq[::-1])

    gp, gm = per_step(h_p), per_step(h_m)
    d_theta = jax.tree.map(lambda a, b: -(a - b) / (2 * eps), gp, gm)

    # Echo step j corresponds to forward step K-j; flip to forward time order.
    flip = lambda x: x[::-1]
    return jax.tree.map(flip, (dphi, dpi, d_theta))


def main():
    params, u_seq, y_seq = make_problem(jr.PRNGKey(0))
    lam_phi, lam_pi, g_theta = adjoint_reference(params, u_seq, y_seq)
    d_phi, d_pi, d_theta = rhel_estimates(params, u_seq, y_seq, EPS)

    # Echo step j lands on forward state K-1-(j) after flip; the probe xi_k
    # sits after step k+1 in scan emission order -> both are indexed by the
    # same forward emission index. Align d_* (length K, j flipped) to lam_*.
    t = jnp.arange(K) * DELTA

    def relerr(a, b):
        return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))

    metrics = {
        "max_rel_err_lambda_phi": relerr(d_phi, lam_phi),
        "max_rel_err_lambda_pi": relerr(d_pi, lam_pi),
        "max_rel_err_g_k": relerr(d_theta["k"], g_theta["k"]),
        "max_rel_err_g_m": relerr(d_theta["m"], g_theta["m"]),
        "max_rel_err_g_kij": relerr(d_theta["kij"], g_theta["kij"]),
        "total_grad_rel_err": {
            name: relerr(jnp.sum(d_theta[name], 0), jnp.sum(g_theta[name], 0))
            for name in ("m", "k", "kij")
        },
        "eps": EPS, "delta": DELTA, "K": K,
    }
    print(json.dumps(metrics, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(t, g_theta["kij"][:, 0, 3], "k:", lw=2.5, label=r"ASM $g_{k_{14}}(t)$")
    axs[0].plot(t, d_theta["kij"][:, 0, 3], color="darkgreen", lw=1,
                label=r"RHEL $\Delta_{k_{14}}(t)$")
    axs[0].plot(t, g_theta["m"][:, 3], "k--", lw=2.5, label=r"ASM $g_{m_4}(t)$")
    axs[0].plot(t, d_theta["m"][:, 3], color="yellowgreen", lw=1,
                label=r"RHEL $\Delta_{m_4}(t)$")
    axs[0].set_xlabel("time"); axs[0].set_title("parameter sensitivities")
    axs[0].legend(fontsize=8)
    axs[1].plot(t, lam_phi[:, 0], "k:", lw=2.5, label=r"ASM $\lambda_{\phi_1}(t)$")
    axs[1].plot(t, d_phi[:, 0], color="orange", lw=1, label=r"RHEL $\Delta_{\phi_1}(t)$")
    axs[1].plot(t, lam_phi[:, 3], "k--", lw=2.5, label=r"ASM $\lambda_{\phi_4}(t)$")
    axs[1].plot(t, d_phi[:, 3], color="royalblue", lw=1, label=r"RHEL $\Delta_{\phi_4}(t)$")
    axs[1].set_xlabel("time"); axs[1].set_title("state sensitivities")
    axs[1].legend(fontsize=8)
    fig.suptitle("Six coupled oscillators: RHEL vs adjoint state method (Theorem 3.1)")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/exp1_toy_oscillators.png", dpi=160)
    os.makedirs("results", exist_ok=True)
    with open("results/exp1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved figures/exp1_toy_oscillators.png")


if __name__ == "__main__":
    main()
