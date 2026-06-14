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


def smooth_waveform_dataset(n, seq_len=60, channels=4, n_classes=4, noise=0.05,
                            seed=0):
    """Clean, reconstructable sequences for generative / autoencoding tasks.

    Each sample is a class-specific low-frequency multi-channel sinusoid with
    random per-channel phase and amplitude, plus small observation noise. The
    structured (predictable) part dominates, so reconstruction error is a
    meaningful learning signal — unlike a noise-dominated task where a random
    reservoir already saturates any downstream probe.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, seq_len)
    freqs = 1.0 + np.arange(n_classes)
    labels = rng.integers(0, n_classes, n)
    x = np.zeros((n, seq_len, channels), np.float32)
    for i in range(n):
        phase = rng.uniform(0, 2 * np.pi, channels)
        amp = rng.uniform(0.7, 1.0, channels)
        x[i] = (amp * np.sin(2 * np.pi * freqs[labels[i]] * t[:, None] + phase)).astype(np.float32)
    x += rng.normal(0, noise, x.shape).astype(np.float32)
    y = np.eye(n_classes, dtype=np.float32)[labels]
    return x, y


def make_smooth_splits(n_train=1800, n_val=300, n_test=400, seed=0, **kw):
    parts = [smooth_waveform_dataset(n, seed=seed + i, **kw)
             for i, n in enumerate((n_train, n_val, n_test))]
    return tuple(parts)
