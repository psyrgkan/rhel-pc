# Replication report

Independent JAX re-implementation of Pourcel & Ernoult, *Learning long range
dependencies through time reversal symmetry breaking* (NeurIPS 2025,
arXiv:2506.05259). This report records what was reproduced, the numbers, and
every deviation from the paper / official code.

## Summary

| Paper claim | Our check | Result |
|---|---|---|
| Thm 3.1: RHEL ≡ continuous adjoint state method, at all t | exp1, 6-oscillator toy, float64 | max rel. error **< 2e-12** for state sensitivities λ(t) and parameter sensitivities g(t) |
| Thm 3.2: discrete RHEL ≡ BPTT on an HRU | tests + exp2, float64 | cosine **1.0**, norm ratio **1.0** |
| Thm 3.3: RHEL chaining ≡ BPTT on a deep HSSM | tests + exp2, 6-block HSSMs | cosine **1.0**, norm ratio **1.0** |
| Cor. A.6: exact discrete time-reversal symmetry | tests | echo retraces forward traj to **< 1e-10** |
| Fig. 4: static RHEL vs BPTT gradient match | exp2 | linear cos ∈ [0.99973, 1.0]; nonlinear cos = 1.0 |
| App. A.5.4: float32 underflow fixed by loss rescaling γ | exp2 panel B | see table below |
| Tables 1–2: RHEL ≈ BPTT training performance | exp3 (synthetic task) | RHEL tracks BPTT throughout training |

## Experiment 1 — Theorem 3.1 (toy oscillators, Fig. 2)

Six coupled harmonic oscillators (Eq. 8), input on oscillator 1, target on
oscillator 4, leapfrog step δ=0.01, K=2000 steps, ε=1e-5, float64. We
compare RHEL's time-resolved estimators against the discrete adjoint
(reverse-mode AD with per-step probe variables, which is the exact discrete
counterpart of the continuous ASM).

```
max rel err  λ_φ(t)   = 1.4e-12      max rel err  g_k(t)   = 3.2e-12
max rel err  λ_π(t)   = 1.5e-12      max rel err  g_m(t)   = 4.3e-12
                                     max rel err  g_kij(t) = 1.7e-12
```

RHEL and the ASM agree at **every timestep**, not just on the integrated
gradient — this is the strong form of Theorem 3.1. Figure:
`figures/exp1_toy_oscillators.png`.

## Experiment 2 — Theorem 3.2/3.3 + Fig. 4 + App. A.5.4

Six-block linear and nonlinear HSSMs at SCP1 scale (hidden 64, state 256).
Because Theorems 3.2/3.3 are data-independent, the unreachable SCP1 sample
is replaced by a synthetic one (smooth random walks); this does not affect
the gradient-equivalence claim.

**Panel A (float64, ε=1e-6):** layer-wise cosine similarity and norm ratio
between RHEL and BPTT parameter gradients.

```
linear HSSM    cos ∈ [0.99973, 1.00000]   (A, B, δ across 6 blocks)
nonlinear HSSM cos = 1.00000              (A, B, b, δ, α across 6 blocks)
```

**Panel B (float32, ε=0.1):** mean cosine vs the loss-rescaling factor γ —
reproducing the underflow analysis (Figs. 5–6 of the paper).

| γ | 1e0 | 1e2 | 1e4 | 1e6 |
|---|---|---|---|---|
| linear    | 0.157 | 0.904 | 1.000 | 1.000 |
| nonlinear | 1.000 | 1.000 | 0.988 | 0.362 |

The two trends match the paper exactly: the **linear** model needs *large* γ
to lift the ε-perturbation out of float32 underflow, while the **nonlinear**
model degrades at *large* γ because the amplified nudge breaks the
finite-difference linearization (App. A.5.4, "optimal scaling balances
underflow and linearization errors"). Figure:
`figures/exp2_gradient_comparison.png`.

## Experiment 3 — training parity (Tables 1–2 proxy)

The UEA / PPG-DaLiA datasets are unreachable from this environment, so
training parity is assessed on a synthetic delayed-cue long-range
classification task (`rhel/data.py`): a class-specific sinusoidal cue in the
first 20 of 500 steps, mean-pooled readout. Linear and nonlinear HSSMs
(hidden 32, state 32, 2 blocks) trained by Adam, 3 seeds, RHEL with ε=0.1,
γ=1e4 (float32); BPTT with plain float32 AD.

Result: RHEL's validation-accuracy curve overlaps BPTT's throughout
training for both models, and final test accuracies are statistically
comparable, reproducing the Table 1–2 qualitative finding (RHEL slightly
trails BPTT but matches within noise). Numbers:
`results/exp3_training.json`; figure: `figures/exp3_training.png`.

## Experiment 4 — the predictive-coding proposal (HPC)

See `docs/pc_proposal.md`. A 3-level Hamiltonian Predictive Coding network.

```
(1) exactness  cos(RHEL, BPTT) = 1.000000000000,  norm ratio = 1.0   (float64)
(2) locality   contrastive-Hebbian ΔW vs generic RHEL grad: max rel diff ~ 8e-9
(3) training   persistent-cue task, 2 seeds, test accuracy:
                 HPC BPTT 96.3 ± 1.7 %   HPC RHEL 95.8 ± 2.2 %
```

The 0.5-point RHEL−BPTT gap mirrors the paper's own Table 1–2 margins.

The locality check is the key novelty: the hand-derived PC update
`+(δ/2ε)[e·f(z)ᵀ]_+^−` equals the autodiff RHEL gradient to ~1e-9, i.e. RHEL
*is* the predictive-coding Hebbian rule applied contrastively over echo
trajectories. For HPC the float32 loss-rescaling sweet spot is γ≈1e2 (the PC
loss magnitudes differ from the HSSM, so the HSSM's γ=1e4 over-scales and
adds linearization error — re-grid per model, exactly the App. A.5.4
caveat).

## Experiment 6 — locality / no weight transport (benefit #1)

`experiments/exp6_locality.py`, figure `figures/exp6_locality.png`.

```
per-layer cosine(local Hebbian update, autodiff RHEL grad) = 1.0   (max rel diff ~1e-8)
backprop with random feedback weights: gradient cosine = -0.24
```

The first line shows the RHEL update at each generative weight equals the
local `error × presynaptic-activity` contrast across the two echo
trajectories — fully local, same weights forward and backward. The second
line is a feedback-alignment control: backprop's gradient *depends* on
transported weights, whereas RHEL's echo dynamics reuse the forward weights
by construction, so there is nothing to transport.

## Experiment 5 — energy-based unsupervised learning (benefit #2)

`experiments/exp5_unsupervised.py`, figure `figures/exp5_unsupervised.png`.
Unsupervised autoencoding of clean structured waveforms (no labels).

```
held-out reconstruction MSE:  random HPC 0.182  ->  trained 0.015   (predict-mean 0.368)
RHEL vs BPTT reconstruction curves: identical step-for-step
linear probe (frequency task): HPC 100% (even random), dissipative-PC / raw ~25% (chance)
```

Generative learning genuinely happens (12× below a random reservoir) and the
local RHEL rule matches global BPTT. Honest caveats: conservative HPC needs a
weak leak or it collapses to predicting the mean (non-relaxational inference
is a real limitation, consistent with App. A.4.4); and on this
frequency-coded task a linear probe saturates even for a random reservoir, so
reconstruction — not the probe — is the meaningful learning signal.

## Experiment 7 — the nudging knob (Figure 5)

`experiments/exp7_nudging_knob.py`, figure `figures/exp7_nudging_knob.png`.

```
float64 bias vs eps:  centered (RHEL) slope 2.0  [O(eps^2)],  one-sided (EqProp) slope 1.0 [O(eps)]
float32 practical window: relative gradient error U-shaped, sweet spot eps ~ 0.16
```

RHEL's centered two-echo difference is one order more accurate than a
one-sided nudge (the standard Equilibrium-Propagation estimator) — concrete
justification for flipping the momenta both ways. In float32 the ε→0 limit is
unreachable (underflow), so there is a sweet spot; it lands near ε≈0.1, the
value used throughout, validating the hyperparameter choice (App. A.5.4).

## Figures produced

| File | Shows |
|---|---|
| `concept_schematic.png` | Fig. 1: HPC mechanism (hierarchy + 3-pass RHEL, local update) |
| `exp1_toy_oscillators.png` | Thm 3.1: RHEL == adjoint state method |
| `exp2_gradient_comparison.png` | Thm 3.2/3.3 + underflow vs γ |
| `exp4_hpc_prototype.png` | HPC supervised: RHEL == BPTT, trains to 96 % |
| `exp5_unsupervised.png` | Benefit #2: unsupervised generative learning, RHEL==BPTT |
| `exp6_locality.png` | Benefit #1: RHEL == local PC Hebbian rule, no weight transport |
| `exp7_nudging_knob.png` | Fig. 5: centered O(ε²) vs one-sided O(ε) bias; float32 sweet spot |

## Deviations from the paper / official code

These do not affect the reproduced theoretical claims; they trade some
absolute-performance fidelity for a clean, self-contained implementation.

1. **Independent implementation.** Written from the paper's equations
   (Eqs. 9–18, 36–55, Alg. 1–4) in plain JAX, not forked from the official
   Equinox code. Cross-checked structurally against it.
2. **Architecture simplifications in the HSSM glue:** LayerNorm instead of
   BatchNorm; dropout omitted; real-valued linear SSM parameters instead of
   the paper's complex doubling (App. A.5.3 notes the doubling was needed to
   match LinOSS's *absolute* numbers, not for the RHEL≡BPTT property).
3. **Synthetic data** for exp3/exp4 (UEA & PPG-DaLiA hosts blocked by the
   environment's network policy). The data-independent theory experiments
   (exp1/exp2 and all tests) are reproduced exactly.
4. **Initial state fixed to Φ=0** in the HRUs (as in the official HSSM),
   so we do not estimate initial-state gradients during training (the
   estimator for them is implemented and tested in `rhel_backward`).
5. **Compute:** CPU/float, single machine. The paper used V100/A100 GPUs;
   we do not attempt the ~50k-length PPG run.

## How to reproduce

```bash
pip install -r requirements.txt
pytest tests/ -q
python experiments/exp1_toy_oscillators.py
python experiments/exp2_gradient_comparison.py
python experiments/exp3_training.py
python experiments/exp4_hpc_prototype.py
```
