from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def inner_prod(a, b):
    return jnp.sum(a * b, axis=-1)


@partial(jax.jit, static_argnums=(1,))
def argtopk(a, k: int):
    assert a.ndim == 1
    i = jnp.argpartition(a, -k)[-k:]
    v = a[i]
    j = jnp.argsort(-v)
    return i[j]  # (k,)


@jax.jit
def take_topk(a, indices):
    return a[indices]
