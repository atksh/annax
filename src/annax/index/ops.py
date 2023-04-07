from functools import partial

import jax
import jax.numpy as np


@jax.jit
def inner_prod(a, b):
    return np.sum(a * b, axis=-1)


@partial(jax.jit, static_argnums=(1,))
def argtopk(a, k: int):
    assert a.ndim == 1
    i = np.argpartition(a, -k)[-k:]
    v = a[i]
    j = np.argsort(-v)
    return i[j]  # (k,)
