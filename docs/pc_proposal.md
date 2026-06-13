# Applying RHEL to Predictive Coding: Hamiltonian Predictive Coding (HPC)

This document is the second half of the `/goal`: having replicated RHEL, we
propose how its dynamics and learning model can be applied to a predictive
coding (PC) network, and back the proposal with a working prototype
(`rhel/predictive_coding.py`, `experiments/exp4_hpc_prototype.py`).

## 1. Why predictive coding is a natural target for RHEL

Predictive coding is a hierarchical generative model in which each level
predicts the level below and only the **prediction errors** propagate. In
the standard formulation (Rao & Ballard 1999; Whittington & Bogacz 2017),
value units `z^(l)` and a generative weight `W^(l)` define a free energy

```
F(z; θ, u) = ½‖u − W₀ f(z₁)‖²  +  Σ_{l=1}^{L−1} ½‖z_l − W_l f(z_{l+1})‖²  +  (α/2)‖z‖²
```

with `u` the clamped sensory input and `f` a unit nonlinearity. Inference
relaxes the value units along `ż = −∂F/∂z` to an equilibrium `z*`, and
learning then applies the **local Hebbian rule**

```
ΔW_l ∝ e_l f(z_{l+1})ᵀ,     e_l = z_{l−1} − W_l f(z_l)   (prediction error).
```

PC is attractive precisely because that weight update is local — it needs
only the error at a synapse's output and the activity at its input — which
is why PC is a leading candidate for a biologically plausible / analog
alternative to backprop. But classical PC has three frictions that RHEL
addresses directly:

1. **It is dissipative and relaxational.** Inference is gradient descent on
   `F` to equilibrium; each training step waits for (an approximation of) a
   fixed point, and on temporal data the "inference loop inside every time
   step" is expensive and hard to map onto continuous physical dynamics.
2. **Temporal credit assignment is unsolved cleanly.** For *sequences*, PC
   either treats time with a separate (often heuristic) recurrence or falls
   back to BPTT, losing locality.
3. **Equilibrium PC only matches backprop under assumptions** (small
   nudging / fixed-prediction approximations); the exactness is asymptotic
   and the inference must converge.

RHEL offers a different bargain: keep PC's local, error-based weight update,
but obtain it from **conservative second-order dynamics run forward in
time**, with an exactness guarantee (Theorem 3.2) that does not require any
equilibrium.

## 2. The construction: Hamiltonian Predictive Coding

Give every value unit a **momentum** `π` and let the PC free energy play the
role of the potential energy. Define the separable Hamiltonian

```
H[Φ, θ, u] = ½‖π‖²  +  F(φ; θ, u),        Φ = (φ, π),
```

so the entire PC hierarchy becomes a single **Hamiltonian Recurrent Unit**
(HRU, Def. A.3 of the paper). Three things follow immediately:

- **Inference = the leapfrog forward pass.** The value units obey
  `φ̈ = −∂F/∂φ`, i.e. the PC prediction-error forces become a *restoring
  force* on oscillating units rather than a relaxation. Because the
  Hamiltonian is separable (kinetic depends only on `π`, potential only on
  `φ`), the leapfrog integrator `M_{T,δ/2}∘M_{V,δ}∘M_{T,δ/2}` is symplectic
  and **exactly time-reversible** — the precondition for RHEL.

- **Learning = RHEL, and it reduces to the PC Hebbian rule.** The RHEL
  parameter estimator for a connection `W_l` is

  ```
  ΔW_l = −(δ/2ε) [ ∇_{W_l} F(φᵉ_+(·)) − ∇_{W_l} F(φᵉ_−(·)) ]
       = +(δ/2ε) Σ_k [ e_l^+(k) f(z_{l+1}^+(k))ᵀ − e_l^−(k) f(z_{l+1}^−(k))ᵀ ].
  ```

  This is **exactly** PC's `e · f(z)ᵀ` outer product — the same local
  quantity — but evaluated *contrastively* along the two echo trajectories
  (`+ε` and `−ε` nudging) instead of at a relaxed equilibrium. The error
  `e_l` and presynaptic activity `f(z_{l+1})` are both available at the
  synapse, so locality is preserved. `experiments/exp4_hpc_prototype.py`
  verifies this hand-derived Hebbian expression matches the generic RHEL
  autodiff estimator to ~1e-9.

- **Temporal credit assignment for free, forward in time.** Because the
  whole sequence model is an HRU, Theorem 3.2 gives RHEL ≡ BPTT-through-
  time at every step, with three forward passes and no Jacobian — solving
  PC's sequence problem without abandoning locality. Stacking PC blocks and
  chaining via Theorem 3.3 / Alg. 1 extends this to deep temporal PC.

### What changes vs. classical PC

| | Classical PC | Hamiltonian PC (this proposal) |
|---|---|---|
| Value-unit dynamics | 1st-order relaxation `ż = −∂F/∂z` (dissipative) | 2nd-order conservative `φ̈ = −∂F/∂φ` |
| Inference | iterate to equilibrium `z*` | one symplectic forward pass |
| Weight update | `e·f(z)ᵀ` at equilibrium | `e·f(z)ᵀ` contrasted over ±ε echoes |
| Exactness vs backprop | asymptotic, needs convergence | exact as ε→0 (Thm 3.2), no equilibrium |
| Temporal data | extra recurrence / BPTT | native (HRU is already a sequence model) |
| Backward pass | — (local) but needs settling | same forward dynamics, momenta flipped |

## 3. What the prototype shows

`rhel/predictive_coding.py` implements HPC; four experiments back the two
benefits we focus on (locality and energy-based unsupervised learning),
plus the supervised sanity check. Figure 1 (`figures/concept_schematic.png`)
is the mechanism diagram.

**Supervised sanity (exp4).** A 3-level HPC classifier checks exactness and
trains as well by RHEL as by BPTT — **HPC BPTT 96.3 ± 1.7 %, HPC RHEL
95.8 ± 2.2 %** (chance 25 %), a 0.5-point gap like the paper's Table 1–2
margins. (The supervised task uses a *persistent* cue: a non-dissipative
system cannot forget injected noise — the paper's own caveat, App. A.4.4 —
so the discriminative signal must persist across the sequence.)

### Benefit #1 — locality / no weight transport (exp6, Fig. `exp6_locality.png`)

For the PC Hamiltonian the RHEL update decomposes *exactly* into the classic
predictive-coding Hebbian rule. We verify, on a 3-level HPC, that the
hand-written local update

    ΔW_l = (δ/2ε) Σ_k [ e_l⁺(k) f(z_{l+1}⁺(k))ᵀ − e_l⁻(k) f(z_{l+1}⁻(k))ᵀ ]

equals the generic autodiff RHEL gradient `dL/dW_l` at **every layer**
(cosine **1.0**, max relative difference **~1e-8**), see panels (A)–(B). The
update at synapse `W_l[i,j]` uses only the prediction error at its output
and the presynaptic activity at its input — both physically available there.

Crucially, RHEL's backward (echo) pass is the *same* forward dynamics run on
the *same* weights `W`; it never forms `Wᵀ` or a separate error network.
Panel (C) quantifies how much this matters: standard backprop's gradient
*depends* on transported weights — replacing the backward weights with
random ones (feedback-alignment control) misaligns the gradient to cosine
**−0.24** — a dependence RHEL structurally avoids. This is the strongest
"why PC" answer: HPC achieves temporal credit assignment with a fully local,
weight-transport-free, forward-only rule, and it does so *exactly* (not
approximately, as equilibrium PC does).

### Benefit #2 — energy-based unsupervised learning (exp5, Fig. `exp5_unsupervised.png`)

Because PC is generative, HPC can learn with **no labels** by minimizing its
free energy (the sensory prediction error) — a task-driven SSM cannot. On a
clean structured-waveform autoencoding task:

- **Generative learning happens.** Held-out reconstruction MSE falls from
  **0.182** (a random HPC reservoir) to **0.015** after unsupervised
  training — 12× better than the reservoir and 25× better than predicting
  the mean (panel A). Panel C shows a trained HPC reconstructing a held-out
  waveform that the random reservoir cannot.
- **The local rule matches global backprop.** The RHEL and BPTT
  reconstruction-loss curves are **identical step-for-step** (panel B):
  HPC's local, forward-only learning loses nothing against end-to-end
  autodiff on the unsupervised objective.
- **Caveat we surface honestly.** Conservative HPC needs a *weak* leak
  (α small) and a large enough step for its latents to track the input;
  with a strong leak it collapses to predicting the mean (non-relaxational
  inference is a genuine limitation — see report). On this frequency task a
  linear probe saturates at 100 % even for a *random* reservoir, so
  reconstruction — not the probe — is the honest measure of unsupervised
  learning. (A classic dissipative-PC baseline that time-pools its input
  sits at chance on the probe because pooling destroys the oscillation.)

## 4. Honest limitations and how they shape the design

- **Non-dissipative only.** Classical PC's relaxation *is* dissipative;
  HPC trades it for conservative dynamics, which limits tasks requiring
  forgetting. Mitigations the paper suggests carry over: embed the
  conservative PC core inside a larger system with controlled artificial
  dissipation, or alternate conservative-inference phases with explicit
  resets between sequences.
- **Not yet fully "black-box".** The Hebbian rule still needs `∇_W F`
  (= error × activity). For PC this is benign — that product is the
  canonical local plasticity signal — but a truly knob-free physical
  version would follow App. A.3.2's homeostatic-control sketch.
- **Replay, not online.** RHEL replays inputs in reverse during the echo;
  an online PC implementation would need the streaming variant flagged as
  future work in the paper.
- **Precision.** Like RHEL, HPC needs the loss-rescaling trick (γ≈1e4) in
  float32 to avoid underflow of the ε-perturbation (App. A.5.4); the
  prototype uses it.

## 5. Figures for a paper-worthy result

What a submission needs, and what already exists in this repo:

| # | Figure | Status |
|---|--------|--------|
| 1 | Concept schematic: hierarchy + 3-pass RHEL + local update | done (`concept_schematic.png`) |
| 2 | Correctness: RHEL == BPTT gradients, per layer (cosine/ratio) | done (`exp4`, `exp2`) |
| 3 | **Locality / no weight transport**: per-synapse `error×activity` match + feedback-alignment control | done (`exp6_locality.png`) |
| 4 | **Energy-based unsupervised learning**: reconstruction trained≪random, RHEL==BPTT | done (`exp5_unsupervised.png`) |
| 5 | The ε knob: centered O(ε²) vs one-sided O(ε) bias + float32 sweet spot | done (`exp7_nudging_knob.png`) |
| 6 | A "PC beats BPTT" result (continual learning / sample efficiency) | **to build** |
| 7 | Scaling table on standard benchmarks (sMNIST/sCIFAR, UEA, PPG) | needs data access |

Figures 1–4 establish the core thesis ("HPC makes RHEL's credit assignment a
local, energy-based, weight-transport-free rule, matching backprop"). Figures
5–6 are what would let the paper claim a *behavioural* advantage of PC, not
just an implementational one; both require operating in the finite-nudge
regime where HPC deliberately departs from BPTT (see the strategic fork in
the project discussion). Theory the text still needs: a proposition stating
the exact decomposition of the RHEL estimator into the local PC rule
(formalizing exp6), and a characterization of the finite-ε estimator as a
temporal Equilibrium-Propagation update.

## 6. Concrete next steps

1. **Deep / hierarchical temporal PC** via Alg. 1 chaining across several
   HPC blocks (input-gradient nudging between levels), benchmarked against
   BPTT on the UEA tasks once data access is available.
2. **Precision-weighted PC.** Add learnable precisions Σ_l on the error
   terms (`½ e_lᵀ Σ_l⁻¹ e_l`); these remain a separable potential, so RHEL
   still applies and the precision update is also a local contrast.
3. **Convolutional / weight-tied PC** for images-as-sequences, testing
   whether the conservative inference + RHEL recovers Rao–Ballard receptive
   fields.
4. **Analog mapping.** The payoff mirrors the paper's: a PC network whose
   inference *and* learning are one conservative physical process (photonic
   / spintronic oscillators), with weight updates that are local error-
   activity contrasts — a self-learning predictive-coding machine.

## References

- Pourcel & Ernoult, *Learning long range dependencies through time reversal
  symmetry breaking*, NeurIPS 2025 (arXiv:2506.05259).
- Lopez-Pastor & Marquardt, *Self-learning machines based on Hamiltonian
  echo backpropagation*, PRX 13:031020 (2023).
- Rao & Ballard, *Predictive coding in the visual cortex*, Nat. Neurosci.
  2(1):79–87 (1999).
- Whittington & Bogacz, *An approximation of the error backpropagation
  algorithm in a predictive coding network…*, Neural Comput. 29(5) (2017).
