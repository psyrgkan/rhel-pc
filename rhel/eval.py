"""Evaluation utilities: a frozen-feature linear probe.

Representation quality is measured the standard self-supervised way: freeze
the learned features, fit a linear (multinomial logistic) classifier on top,
and report test accuracy. The probe is the only thing that sees labels.
"""

import jax
import jax.numpy as jnp
import optax


def standardize(train_X, *others, eps=1e-6):
    mu = train_X.mean(0, keepdims=True)
    sd = train_X.std(0, keepdims=True) + eps
    return tuple((X - mu) / sd for X in (train_X, *others))


def linear_probe(train_X, train_y, test_X, test_y, key,
                 steps=600, lr=0.05, weight_decay=1e-3):
    """Fit multinomial logistic regression on frozen features.

    train_y / test_y are one-hot. Returns test accuracy.
    """
    train_X, test_X = standardize(train_X, test_X)
    n_feat, n_cls = train_X.shape[1], train_y.shape[1]
    params = {"w": jnp.zeros((n_cls, n_feat)), "b": jnp.zeros((n_cls,))}
    opt = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = opt.init(params)

    def loss_fn(p, X, y):
        logits = X @ p["w"].T + p["b"]
        return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits, -1), -1))

    @jax.jit
    def step(p, opt_state):
        loss, g = jax.value_and_grad(loss_fn)(p, train_X, train_y)
        updates, opt_state = opt.update(g, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    for _ in range(steps):
        params, opt_state, _ = step(params, opt_state)

    logits = test_X @ params["w"].T + params["b"]
    return float(jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(test_y, -1)))
