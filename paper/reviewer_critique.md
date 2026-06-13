# Adversarial review: *Hamiltonian Predictive Coding*

A deliberately critical, NeurIPS-style review of `main.tex`, written to find
the weak spots before a real reviewer does. Each issue has a **severity**
(blocker / major / minor) and a concrete remedy. The ordering is by how likely
it is to sink the paper.

**Summary of the work.** The paper gives predictive-coding value units momenta,
turning the PC free energy into the potential of a conservative Hamiltonian, so
the hierarchy becomes one Hamiltonian recurrent unit trainable by RHEL (Pourcel
& Ernoult, 2025). It argues two benefits — a local, weight-transport-free
learning rule, and energy-based unsupervised learning — plus a bias analysis of
the nudging estimator. Claims are supported on synthetic sequence tasks.

**Overall.** The construction is clean and the locality story is genuinely
appealing, but in its current form this reads as a *corollary of RHEL applied
to a specific Hamiltonian*, with no demonstrated advantage over backpropagation
and only synthetic experiments. Borderline; leaning reject without a
substantially stronger empirical case. Below is what would move it.

---

## Blockers (must fix before this is publishable)

### B1. No demonstrated advantage over backpropagation — the "so what?"
The headline results show HPC *equals* BPTT (by design — RHEL is an exact BPTT
proxy). The paper never exhibits a task, metric, or regime where HPC *beats*
BPTT, or even where its locality/energy properties translate into a measurable
win. A reviewer's first question is "why would I use this instead of BPTT?"
and the paper's honest answer is "for hypothetical analog hardware" — which it
does not build or simulate at the device level.
- **Remedy.** Deliver at least one of: (a) a continual-learning / few-shot
  result where the finite-ε (PC-equilibrium) regime beats BPTT — the
  "prospective configuration" advantage the paper itself cites (Song et al.,
  2024) but never tests; (b) a concrete energy/locality accounting on a
  plausible analog substrate; or (c) a memory/throughput advantage on digital
  hardware. Absent any of these, the contribution is "RHEL also works for PC,"
  which is thin.

### B2. Synthetic-only evaluation
Every experiment uses bespoke sinusoidal toy data. There is no standard
benchmark (sequential MNIST/CIFAR, the UEA suite or PPG-DaLiA used by the RHEL
paper, any language/vision task). Scale (depth, width, sequence length,
parameter count) is small and unreported in the main text.
- **Remedy.** Reproduce on at least the RHEL paper's own benchmarks (UEA
  classification, PPG-DaLiA regression) so HPC is comparable to a published
  baseline, and add one image-as-sequence task. Report model sizes, sequence
  lengths, wall-clock, and seeds in the main text.

### B3. The "no weight transport" claim does not hold for the whole network
The locality argument applies to the recurrent PC core, but the **readout head
is trained by ordinary backprop with weight transport** (the paper says so in
§3 and the appendix). So the network as a whole is not weight-transport-free,
and the strongest framing of benefit #1 is overstated.
- **Remedy.** Either (a) make the readout local too (e.g. a local classifier /
  local error on the top level) and show end-to-end locality, or (b) scope the
  claim precisely throughout ("the recurrent credit assignment is
  transport-free; the readout is not") and quantify what fraction of parameters
  and of the credit-assignment path is local.

---

## Major issues

### M1. Novelty over RHEL is narrow and should be argued, not assumed
Proposition 1 is a direct substitution of the PC potential into RHEL's
estimator (Thm 3.2); the centered-vs-one-sided O(ε²) result is standard
finite-difference calculus plus the known EqProp connection. The genuinely new
content is "PC is a good Hamiltonian to choose." That may be enough, but the
paper must *argue* the novelty explicitly and not let a reviewer conclude it is
an application note.
- **Remedy.** Sharpen the conceptual claim: PC is the model class where RHEL's
  ∇_θH becomes the canonical local plasticity signal, closing RHEL's own
  open problem (their App. A.3.2, "still requires explicit ∇_θH"). State this
  as a contribution and support it (e.g. show RHEL on a *non*-PC Hamiltonian
  does *not* yield a transport-free local rule, making PC special).

### M2. The unsupervised benefit rests on a weak proxy
The representation probe saturates at 100% even for a *random* reservoir, so it
cannot show that learning improves representations; the paper falls back to
reconstruction MSE. But low reconstruction MSE on smooth sinusoids is a low bar,
and the dissipative-PC baseline is compared only on the (saturated/at-chance)
probe, not on reconstruction. The claim "HPC does useful unsupervised learning"
is therefore under-supported.
- **Remedy.** Use a task where the probe is *not* saturated by a random
  reservoir (so learning demonstrably helps a downstream task), and compare
  HPC's unsupervised representations head-to-head with (i) classic dissipative
  PC and (ii) a backprop autoencoder, on the *same* downstream metric. Show
  HPC is at least competitive, ideally on real data.

### M3. No analysis of the conservative inference dynamics
HPC replaces PC's relaxation with undamped oscillation, but the paper never
analyzes whether this "inference" is meaningful: the leapfrog PC dynamics
oscillate and do not settle, and the readout mean-pools over the oscillation.
Stability (does the symplectic map stay bounded for the learned, possibly
stiff, potential?), the role of the step δ, and what the latents actually
represent are all unexamined. The strong-leak collapse reported in §6 suggests
the dynamics are delicate.
- **Remedy.** Add a stability/again-energy-conservation analysis of the
  integrator on the learned potential; characterize the inference quality
  (e.g. does the conservative pass reach the PC fixed point in any averaged
  sense?); ablate δ and α.

### M4. Missing local-learning baselines
The paper positions HPC against backprop and (briefly) dissipative PC, but
omits the obvious neighbors: feedback alignment, target propagation, and
Equilibrium Propagation as *trained* baselines (not just an FA gradient-angle
probe). Without them, the locality contribution floats free of its field.
- **Remedy.** Train FA, EqProp, and dissipative PC on the same tasks; show
  where HPC sits (it should match BPTT while they may not).

### M5. The feedback-alignment "control" is a strawman as presented
Figure (C)/§5.2 shows that randomizing backprop's backward weights misaligns
the *raw* gradient (cosine −0.24). But feedback alignment is famous precisely
because training still works despite this misalignment. The panel therefore
shows backprop *uses* transported weights, not that transport is *necessary* or
that RHEL is *better*. The rhetorical weight placed on −0.24 is more than it
can bear.
- **Remedy.** Reframe as "backprop's gradient is a function of transported
  weights; RHEL's is not" (a structural statement) and drop any implication of
  superiority; or run FA as a trained baseline and compare final accuracy.

---

## Minor issues

- **m1. Seeds and error bars.** 2–3 seeds throughout; several headline numbers
  (96.3% vs 95.8%) have overlapping error bars and are described as "matching"
  — fine, but state the non-significance explicitly and add seeds.
- **m2. γ must be re-grid per model.** The loss-rescaling sweet spot differs
  (10² for HPC vs 10⁴ for SSMs). This is a usability wart inherited from RHEL;
  acknowledge it as a hyperparameter sensitivity, not a footnote.
- **m3. Not online.** "Forward-only" is claimed, but inputs are replayed in
  reverse during the echo, so it is not online; the abstract/intro could be
  read as stronger than warranted. Already noted in §6 — surface it earlier.
- **m4. Reconstruction metric scale.** Panel A mixes "predict-mean 0.368" and
  per-element MSE; make the normalization explicit so 0.015 is interpretable.
- **m5. Biological-plausibility framing.** The "self-learning machine" framing
  is aspirational; no spiking, no Dale's law, no locality-in-time analysis of
  the echo replay (which arguably needs a buffered reverse pass — not obviously
  biological). Temper or substantiate.
- **m6. Theorem citation.** Theorem 1 is stated "informal" and the separability
  / time-reversal conditions for the *PC* Hamiltonian should be verified in an
  appendix lemma, not asserted, since the leak term and tanh nonlinearity must
  be checked against the assumptions.

## Questions for the authors
1. Is there *any* task where HPC beats BPTT? If not, what is the paper's claim
   beyond "RHEL applies to PC"?
2. What do the conservative latents represent, and does the undamped inference
   ever help versus hurt relative to dissipative PC?
3. How does the method scale with depth/sequence length, and what breaks first
   (stability, underflow, the weak-leak requirement)?
4. End to end, what fraction of parameters are updated by a local,
   transport-free rule once the readout is included?

## What is already solid (keep)
- The construction and Proposition 1 are clean; the locality decomposition is
  verified convincingly (cosine 1.0, ~1e-8).
- The centered vs one-sided O(ε²)/O(ε) result and the EqProp bridge are correct
  and a nice touch.
- The limitations section is unusually honest; lean into that credibility by
  pre-empting B1/M2 rather than letting reviewers raise them.

## Suggested revision priority
1. B1 (a real advantage) and B2 (real benchmarks) — without these it does not
   clear the bar.
2. B3/M5 (scope the locality claim honestly) and M4 (local-learning baselines).
3. M2/M3 (strengthen the unsupervised story; analyze the dynamics).
4. The minors.
