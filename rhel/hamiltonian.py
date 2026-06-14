"""Core RHEL machinery: leapfrog HRUs and the RHEL custom-VJP.

Replicates the discrete-time theory of

    Pourcel & Ernoult, "Learning long range dependencies through time
    reversal symmetry breaking", NeurIPS 2025 (arXiv:2506.05259).

A Hamiltonian Recurrent Unit (HRU, Def. A.3 of the paper) integrates a
separable Hamiltonian

    H[Phi, theta, u] = T[pi, theta] + V[phi, theta, u],   Phi = (phi, pi)

with the (time-reversible, symplectic) leapfrog integrator

    M_{H,d} = M_{T,d/2} o M_{V,d} o M_{T,d/2}:
        phi_{k+1/3} = phi_k     + 1/2 grad_pi  T(pi_k)
        pi_{k+2/3}  = pi_k      -     grad_phi V(phi_{k+1/3}, u_k)
        phi_{k+1}   = phi_{k+1/3} + 1/2 grad_pi T(pi_{k+2/3})
        pi_{k+1}    = pi_{k+2/3}

We use the "reparametrized" Hamiltonian convention of App. A.4.6: any time
step delta is folded into T and V themselves so the integrator runs with a
unit step.  Parameter gradients w.r.t. delta then come for free.

RHEL (Alg. 1) replaces the backward pass with two extra *forward* echo
passes of the same dynamics.  Starting from the momentum-flipped final
state, nudged by +/- eps times the incoming loss gradients, parameter and
input gradients are read out as finite differences of Hamiltonian
gradients along the two echo trajectories:

    Delta_theta(k) = -1/(2 eps) [ grad_theta H_1/2(Phi^e_k(+eps))
                                 - grad_theta H_1/2(Phi^e_k(-eps)) ]

with H_1/2 the Hamiltonian evaluated at the two intermediate leapfrog
states (Eq. 14).  Summed over k this equals the BPTT gradient in the limit
eps -> 0 (Theorem 3.2).
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class HamiltonianSpec:
    """A separable Hamiltonian H = T(pi, params) + V(phi, u, params).

    Both terms map to a scalar; `params` is an arbitrary pytree. Any time
    step must be folded into T and V (unit-step leapfrog, App. A.4.6).
    """

    kinetic: Callable[[jax.Array, Any], jax.Array]
    potential: Callable[[jax.Array, jax.Array, Any], jax.Array]

    def grad_T(self, pi, params):
        return jax.grad(self.kinetic, argnums=0)(pi, params)

    def grad_V(self, phi, u, params):
        return jax.grad(self.potential, argnums=0)(phi, u, params)

    def half_step_hamiltonian(self, phi13, pi_pre, pi23, u, params):
        """H_1/2 of Eq. (14): mean of H at the two intermediate leapfrog states.

        Both intermediate states share the position phi_{k+1/3}; they differ
        in momentum (pi_k before the kick, pi_{k+2/3} after it).
        """
        t_half = 0.5 * (self.kinetic(pi_pre, params) + self.kinetic(pi23, params))
        return t_half + self.potential(phi13, u, params)


def leapfrog_step(spec, params, phi, pi, u):
    """One unit-step leapfrog update; also returns the intermediate states."""
    phi13 = phi + 0.5 * spec.grad_T(pi, params)
    pi23 = pi - spec.grad_V(phi13, u, params)
    phi1 = phi13 + 0.5 * spec.grad_T(pi23, params)
    return phi1, pi23, phi13


def hru_scan(spec, params, phi0, pi0, u_seq, nudge_seq=None, eps=0.0):
    """Run an HRU over a sequence, optionally nudging the momenta.

    The nudge `eps * nudge_seq[k]` is added to the momentum *after* the k-th
    leapfrog step (Eq. 13 / Eq. 39).  Returns the trajectories of the full
    states after each step (post-nudge) and of the intermediate states
    needed by the RHEL learning rule.
    """
    if nudge_seq is None:
        nudge_seq = jnp.zeros((u_seq.shape[0],) + pi0.shape, dtype=pi0.dtype)

    def step(carry, inp):
        phi, pi = carry
        u, n = inp
        pi_pre = pi
        phi1, pi23, phi13 = leapfrog_step(spec, params, phi, pi, u)
        pi1 = pi23 + eps * n
        return (phi1, pi1), (phi1, pi1, phi13, pi_pre, pi23)

    (phiK, piK), traj = jax.lax.scan(step, (phi0, pi0), (u_seq, nudge_seq))
    return (phiK, piK), traj


def make_hru(spec: HamiltonianSpec, state_dim: int, learning_algorithm: str = "BPTT",
             centered: bool = True):
    """Build `apply(params, u_seq, eps) -> phi trajectory` for an HRU.

    With learning_algorithm == "BPTT", gradients are computed by plain
    reverse-mode AD through the leapfrog scan. With "RHEL", a custom VJP
    implements Alg. 1: the backward pass re-runs the same forward dynamics
    from the momentum-flipped final state (echo passes) with nudging, and
    reads gradients out of the trajectory contrast. The initial state is
    fixed to Phi = 0 as in the paper's HSSMs.

    centered=True uses the symmetric +/-eps two-echo difference (RHEL, O(eps^2)
    bias, exact BPTT proxy). centered=False uses the one-sided free + single
    nudged echo (the Equilibrium-Propagation estimator, O(eps) bias): at finite
    eps this evaluates the update at target-nudged activities, the
    "prospective configuration" regime that departs from BPTT.
    """

    def forward(params, u_seq):
        zeros = jnp.zeros((state_dim,), dtype=u_seq.dtype)
        (_, _), traj = hru_scan(spec, params, zeros, zeros, u_seq)
        phis = traj[0]
        return phis

    if learning_algorithm == "BPTT":
        return lambda params, u_seq, eps: forward(params, u_seq)

    @jax.custom_vjp
    def apply(params, u_seq, eps):
        return forward(params, u_seq)

    def apply_fwd(params, u_seq, eps):
        zeros = jnp.zeros((state_dim,), dtype=u_seq.dtype)
        (phiK, piK), traj = hru_scan(spec, params, zeros, zeros, u_seq)
        phis = traj[0]
        # Time-reversal symmetry: the echo pass can reconstruct the whole
        # trajectory from the final state alone, so it is the only residual.
        return phis, (params, u_seq, eps, phiK, piK)

    def apply_bwd(res, g_seq):
        params, u_seq, eps, phiK, piK = res
        grads_p, grads_u, _ = rhel_backward(spec, params, u_seq, eps, phiK, piK, g_seq,
                                            centered=centered)
        return grads_p, grads_u, jnp.zeros(())

    apply.defvjp(apply_fwd, apply_bwd)
    return apply


def rhel_backward(spec, params, u_seq, eps, phiK, piK, g_seq, return_per_step=False,
                  centered=True):
    """Alg. 1: echo passes and the contrastive gradient estimator.

    g_seq[k] = dL/dphi_{k+1} (upstream gradients w.r.t. the position
    trajectory emitted by the forward pass), forward time order.

    centered=True: symmetric +/-eps difference (RHEL). centered=False:
    one-sided free + nudged difference (Equilibrium Propagation).

    Returns (parameter grads, input-sequence grads, initial-state grads).
    """
    K = u_seq.shape[0]
    u_rev = u_seq[::-1]
    # Nudges in echo time: the initial conjugated state is nudged with the
    # final-time loss gradient g_K; echo step j is nudged with g_{K-1-j};
    # the last step receives no nudge (there is no loss at the initial state).
    g_rev = g_seq[::-1]
    n0 = g_rev[0]
    n_seq = jnp.concatenate([g_rev[1:], jnp.zeros_like(g_rev[:1])], axis=0)

    def one_echo(sign):
        e = sign * eps
        # Phi^e_0 = Sigma_z Phi_0 + eps Sigma_x grad_Phi l[Phi_0]; the loss
        # only touches positions, so only the momentum is nudged.
        phi0 = phiK
        pi0 = -piK + e * n0
        (phiE, piE), traj = hru_scan(spec, params, phi0, pi0, u_rev, n_seq, e)
        phis, pis, phi13s, pi_pres, pi23s = traj
        return (phiE, piE), (phi13s, pi_pres, pi23s)

    # +eps echo always; the second echo is -eps (centered) or 0 (one-sided).
    (state_p, halves_p) = one_echo(+1.0)
    (state_m, halves_m) = one_echo(-1.0 if centered else 0.0)
    denom = (2.0 * eps) if centered else eps

    grad_H = jax.grad(spec.half_step_hamiltonian, argnums=(3, 4))

    def per_step(halves):
        phi13s, pi_pres, pi23s = halves
        return jax.vmap(lambda a, b, c, u: grad_H(a, b, c, u, params))(
            phi13s, pi_pres, pi23s, u_rev
        )

    gu_p, gp_p = per_step(halves_p)
    gu_m, gp_m = per_step(halves_m)

    grads_params = jax.tree.map(
        lambda a, b: -(jnp.sum(a, axis=0) - jnp.sum(b, axis=0)) / denom, gp_p, gp_m
    )
    grads_u = -(gu_p - gu_m) / denom
    grads_u = grads_u[::-1]

    phiE_p, piE_p = state_p
    phiE_m, piE_m = state_m
    grad_x0 = (
        (piE_p - piE_m) / denom,  # dL/dphi_0
        (phiE_p - phiE_m) / denom,  # dL/dpi_0
    )
    if return_per_step:
        per_step_theta = jax.tree.map(lambda a, b: -(a - b) / denom, gp_p, gp_m)
        return grads_params, grads_u, grad_x0, per_step_theta
    return grads_params, grads_u, grad_x0
