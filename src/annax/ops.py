from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


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
    return a.take(indices, axis=0)


@partial(jax.jit, static_argnums=(1,))
def nonzero_idx(mask: Array, max_cluster_size: int) -> Array:
    """Find the indices of nonzero elements in a mask.

    Args:
        mask (Array): a mask with shape (n,)
        max_cluster_size (int): maximum number of nonzero elements

    Note:
        sum(mask) <= max_cluster_size

    Returns:
        Array: indices of nonzero elements
    """
    out = jnp.full((max_cluster_size,), -1, dtype=jnp.int32)
    init_val = {"i": 0, "j": 0, "out": out}

    def cond(val):
        return val["i"] < mask.shape[0]

    def body(val):
        i = val["i"]
        j = val["j"]
        out = val["out"]
        j = jax.lax.cond(mask[i], lambda _: j + 1, lambda _: j, None)
        out = jax.lax.cond(mask[i], lambda _: out.at[j].set(i), lambda _: out, None)
        return {"i": i + 1, "j": j, "out": out}

    val = jax.lax.while_loop(cond, body, init_val)
    return val["out"]
