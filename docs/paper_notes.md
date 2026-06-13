# Paper notes — "Learning long range dependencies through time reversal symmetry breaking"

Guillaume Pourcel, Maxence Ernoult — NeurIPS 2025 (oral), [arXiv:2506.05259](https://arxiv.org/abs/2506.05259),
[official code](https://github.com/guillaumepourcel/rhel).

## Problem

Deep state space models (SSMs) are RNNs that could in principle be *embodied
in physical dynamical systems*. Training them there requires a credit
assignment algorithm that (i) uses only forward passes (no separate backward
solver) and (ii) never materializes the state Jacobian. RTRL satisfies
neither practically (cubic memory); zeroth-order/forward-gradient methods
satisfy both but suffer variance or bias. Hamiltonian Echo Backprop (HEB,
Lopez-Pastor & Marquardt 2023) showed that *non-dissipative Hamiltonian*
systems can compute gradients through their own dynamics, but only for
static problems, learnable initial states, and with theoretical-physics
machinery that is hard to map onto ML practice.

## Core idea: RHEL

For a Hamiltonian system Φ = (φ, π) with ∂t Φ = J ∇Φ H[Φ, θ, u(t)],
time-reversal symmetry means: flip the momenta at the end of a trajectory
and the system exactly retraces its history (the "echo"). RHEL perturbs the
echo: while replaying inputs backward, it nudges the trajectory with
strength ±ε in the direction of decreasing loss. The broken symmetry — the
gap between the +ε and −ε echo trajectories — encodes exactly the adjoint
state, and parameter gradients are read out as finite differences of
Hamiltonian gradients along the two echoes:

- Δθ(t, ε) = −(1/2ε) [∇θH(Φᵉ(t, ε)) − ∇θH(Φᵉ(t, −ε))]
- ΔΦ(t, ε) = (1/2ε) Σx (Φᵉ(t, ε) − Φᵉ(t, −ε))

Three forward passes total (1 free + 2 echo), regardless of model size; no
Jacobians; zero gradient-estimation variance (deterministic estimator,
bias O(ε²)).

## Theory

- **Theorem 3.1 (continuous time).** ΔΦ and Δθ converge (ε→0) to the
  continuous adjoint state method's λ(t) and gθ(t) at *every time t*.
- **Discrete time: HRUs.** A Hamiltonian Recurrent Unit integrates a
  *separable* Hamiltonian H = T[π] + V[φ] with the leapfrog integrator
  M = M_{T,δ/2} ∘ M_{V,δ} ∘ M_{T,δ/2}, which is symplectic and exactly
  time-reversible (M⁻¹ = Σz M Σz). Time-reversal symmetry therefore holds
  *exactly* in discrete time. (LinOSS's IMEX integrator is *not* reversible
  — App. A.4.4 — hence the leapfrog rearrangement.)
- **Theorem 3.2.** Discrete RHEL (echo passes with nudges added after each
  leapfrog step, gradients read at the intermediate "half" states via
  H^{1/2}, Eq. 14) equals BPTT through the leapfrog graph, at every step.
- **Theorem 3.3 (HSSMs).** Stacks of HRUs (each unit's position trajectory
  feeding the next) are trained by chaining: the top HRU's echo is nudged
  by loss gradients; each HRU also estimates *input* gradients Δu which
  nudge the echo of the HRU below. In practice RHEL is wrapped as a custom
  VJP at each HRU boundary and chained with ordinary autodiff through the
  feedforward glue (Remark 8, Alg. 3).

## Models

- **Linear HRU** (LinOSS-style oscillatory SSM): V = ½ φᵀAφ − φᵀBu,
  A diagonal ≥ 0; trainable per-channel step δ via Hamiltonian
  reparametrization (App. A.4.6); parallel-scannable.
- **Nonlinear HRU** (UnICORNN-style): V = α/2‖φ‖² + Σ A⁻¹ log cosh(Aφ + Bu + b),
  sequential leapfrog.
- **HSSM**: encoder → N × [norm → HRU → C/D readout → GELU → GLU →
  residual] → decoder (LinOSS macro-architecture).

## Experiments (paper)

- **Fig. 2**: toy 6-oscillator chain — RHEL estimators overlap the ASM
  sensitivities at all times.
- **Fig. 4**: static gradient check on 6-block linear/nonlinear HSSMs (an
  SCP1 sample): cosine similarity and norm ratio ≈ 1 for all parameters
  and blocks.
- **Table 1** (UEA classification, 5 seeds, test acc %):
  RHEL ≈ BPTT on Worms (75.0/78.3), SCP1 (86.1/86.4), SCP2 (61.4/63.9),
  Ethanol (29.9/29.9), Heartbeat (73.5/73.9), Motor (51.6/48.1);
  averages 62.9 (RHEL) vs 63.4 (BPTT) linear, 57.8 vs 58.8 nonlinear.
- **Table 2** (PPG-DaLiA regression, seq. length ~50k, test MSE ×10⁻²):
  linear 9.5 (RHEL) vs 9.1 (BPTT); nonlinear 8.4 vs 7.8.
- **App. A.5.4**: with float32 and ε = 0.1, naive RHEL underflows (the
  ε-perturbation drowns in rounding); rescaling the loss by γ during the
  echo (and dividing the gradient by γ) restores near-perfect gradient
  alignment; for nonlinear models too large a γ reintroduces linearization
  error — γ ≈ 10⁴–10⁶ is the sweet spot.

## Limitations / open directions (paper's own)

- Restricted to non-dissipative systems; naive dissipation provably breaks
  the estimator (App. A.3.1).
- Learning rule still needs explicit ∇θH (not yet a fully "black-box"
  physical procedure; App. A.3.2 sketches a homeostatic-knob variant).
- Not online: inputs must be replayed in reverse.
- On digital hardware RHEL matches but does not beat BPTT in time/memory;
  the payoff is for analog/physical substrates.
