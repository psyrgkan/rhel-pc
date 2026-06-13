"""Synthetic sequence tasks.

The UEA / PPG-DaLiA hosts used in the paper are not reachable from this
environment, so training parity between RHEL and BPTT (Tables 1-2 of the
paper) is assessed on a synthetic long-range classification task instead.

Delayed-cue classification: each sequence has length `seq_len`; a
class-specific sinusoidal cue occupies only the first `cue_len` steps and
the remainder is pure noise. Because the model readout mean-pools over
time, the recurrent state must carry the cue information across the whole
sequence -- a long-range memory requirement in miniature.
"""

import numpy as np


def delayed_cue_dataset(n, seq_len=500, cue_len=20, n_classes=4, channels=4,
                        noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    freqs = 1.0 + np.arange(n_classes)          # class-specific cue frequency
    labels = rng.integers(0, n_classes, size=n)
    t = np.arange(cue_len)
    x = rng.normal(0.0, noise, size=(n, seq_len, channels)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=(n, channels))
    for i in range(n):
        cue = np.sin(2 * np.pi * freqs[labels[i]] * t[:, None] / cue_len + phase[i])
        x[i, :cue_len] += cue.astype(np.float32)
    y = np.eye(n_classes, dtype=np.float32)[labels]
    return x, y


def make_splits(n_train=2000, n_val=400, n_test=400, seed=0, **kw):
    x, y = delayed_cue_dataset(n_train + n_val + n_test, seed=seed, **kw)
    return ((x[:n_train], y[:n_train]),
            (x[n_train:n_train + n_val], y[n_train:n_train + n_val]),
            (x[n_train + n_val:], y[n_train + n_val:]))
