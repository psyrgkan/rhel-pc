# RHEL replication + Hamiltonian Predictive Coding

A from-scratch replication of **"Learning long range dependencies through
time reversal symmetry breaking"** (Pourcel & Ernoult, NeurIPS 2025 oral,
[arXiv:2506.05259](https://arxiv.org/abs/2506.05259)), together with a
proposal — backed by a working prototype — for applying its dynamics and
learning model (**RHEL**) to a **predictive coding** network.

This is an independent re-implementation in JAX, written to understand the
method, not a fork of the [official code](https://github.com/guillaumepourcel/rhel).

## What is RHEL, in one paragraph

A non-dissipative Hamiltonian system has *time-reversal symmetry*: flip the
momenta at the end of a trajectory and it exactly retraces its history.
RHEL perturbs ("nudges") this reverse "echo" by ±ε in the direction of
decreasing loss; the gap between the +ε and −ε echo trajectories encodes
the adjoint state exactly, so loss gradients fall out as finite differences
of Hamiltonian gradients along the two echoes — **three forward passes, no
Jacobian, no variance**, provably equal to BPTT (Theorem 3.2). See
[`docs/paper_notes.md`](docs/paper_notes.md) for the full summary.

## Layout

```
rhel/
  hamiltonian.py       # leapfrog HRU + RHEL custom-VJP (Alg. 1, Eq. 13-14)
  models.py            # linear & nonlinear HRUs, HSSM stack (App. A.4)
  data.py              # synthetic sequence tasks (classification, waveforms)
  predictive_coding.py # Hamiltonian Predictive Coding (the proposal)
  eval.py              # frozen-feature linear probe
experiments/
  exp1_toy_oscillators.py    # Fig. 2: RHEL == adjoint state method (Thm 3.1)
  exp2_gradient_comparison.py# Fig. 4: RHEL == BPTT grads + underflow (A.5.4)
  exp3_training.py           # Tables 1-2 proxy: RHEL vs BPTT training
  exp4_hpc_prototype.py      # HPC supervised: exactness, locality, training
  exp5_unsupervised.py       # benefit #2: energy-based unsupervised learning
  exp6_locality.py           # benefit #1: local rule, no weight transport
  exp7_nudging_knob.py       # Fig. 5: centered O(eps^2) vs one-sided O(eps) bias
  exp8_continual.py          # continual learning: finite-eps HPC vs BPTT (negative)
  exp9_dissipative_pc.py     # dissipative temporal PC (controlled dissipation)
  make_concept_figure.py     # Fig. 1 schematic
rhel/temporal_pc.py          # dissipative temporal PC (equilibrium inference, local)
tests/                       # time-reversal symmetry, Thm 3.2/3.3, PC locality
docs/
  paper_notes.md             # detailed paper summary
  pc_proposal.md             # the predictive-coding proposal
  replication_report.md      # results, deviations, what matched
paper/
  main.tex                   # the submission-ready paper (NeurIPS format)
  reviewer_critique.md       # adversarial self-review of the paper
figures/ results/            # generated artifacts
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                          # gradient-equivalence theory checks
python experiments/exp1_toy_oscillators.py
python experiments/exp2_gradient_comparison.py
python experiments/exp3_training.py        # ~1h on CPU
python experiments/exp4_hpc_prototype.py
```

## Headline results (reproduced here)

- **Theorem 3.1** (continuous, toy oscillators): RHEL's time-resolved
  estimators match the adjoint state method to **<2e-12** relative error.
- **Theorem 3.2 / 3.3** (single HRU and deep HSSM): RHEL gradients equal
  BPTT gradients — cosine similarity `1.0` in float64 (tests + exp2).
- **App. A.5.4**: in float32 the ε-perturbation underflows; loss rescaling
  by γ≈1e4 restores near-perfect gradient alignment (exp2, panel B).
- **Training parity**: RHEL tracks BPTT throughout optimization on linear
  and nonlinear HSSMs (exp3).
- **The proposal — Hamiltonian Predictive Coding** ([`docs/pc_proposal.md`](docs/pc_proposal.md)):
  a PC hierarchy given momenta becomes one HRU.
  - *Locality (exp6):* RHEL's weight update **is** PC's local error×activity
    Hebbian rule (cosine `1.0` per layer), using the same weights forward and
    backward — no weight transport (vs cosine `−0.24` for backprop with
    randomized feedback).
  - *Energy-based unsupervised learning (exp5):* HPC learns a generative
    model with no labels — reconstruction MSE `0.182` (random) → `0.015`
    (trained) — and the local RHEL rule matches global BPTT step-for-step.
  - *Supervised (exp4):* HPC RHEL `95.8%` vs BPTT `96.3%`.

## Data availability

The UEA classification and PPG-DaLiA hosts used in the paper are not
reachable from this environment, so the *training* experiments (exp3/exp4)
use a synthetic long-range task with matched difficulty. The *theoretical*
claims (Theorems 3.1–3.3, the underflow analysis) are data-independent and
are reproduced exactly. See `docs/replication_report.md`.
