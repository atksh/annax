from functools import partial

import jax
import jax.numpy as np
from jax import Array

from ..utils import pvmap


def inner_prod(a, b):
    return np.sum(a * b, axis=-1)


@jax.jit
def parallel_inner_prod(query, data):
    f = pvmap(inner_prod, in_axes=(0, None), out_axes=0)
    return f(query, data)


@partial(jax.jit, static_argnums=(1,))
def argtopk(a, k: int):
    assert a.ndim == 1
    i = np.argpartition(a, -k)[-k:]
    v = a[i]
    j = np.argsort(v)
    return i[j]


@partial(jax.jit, static_argnums=(1,))
def parallel_argtopk(a, k: int):
    f = pvmap(argtopk, in_axes=0, out_axes=0)
    return f(a, k)
