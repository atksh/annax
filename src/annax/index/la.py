from functools import partial

import jax
import jax.numpy as np
from jax import Array

from ..utils import pvmap


@jax.jit
def inner_prod(a, b):
    return np.sum(a * b, axis=-1)


def parallel_inner_prod(query, data):
    f = pvmap(partial(inner_prod, b=data))
    return f(query)


@partial(jax.jit, static_argnums=(1,))
def argtopk(a, k: int):
    assert a.ndim == 1
    i = np.argpartition(a, -k)[-k:]
    v = a[i]
    j = np.argsort(-v)
    return i[j]  # (k,)


def parallel_argtopk(a, k: int):
    f = pvmap(partial(argtopk, k=k))
    return f(a)
