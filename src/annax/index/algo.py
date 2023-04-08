from typing import Tuple

import jax
import jax.numpy as np
from jax import Array

from ..pfunc import k_pmap, left_pjit
from .ops import argtopk, inner_prod, take_topk

__all__ = ["naive_search"]

parallel_inner_prod = left_pjit(inner_prod)
parallel_argtopk = k_pmap(argtopk)
parallel_take_topk = jax.vmap(take_topk, in_axes=(0, 0), out_axes=0)


def naive_search(data: Array, query: Array, k: int = 1) -> Tuple[Array, Array]:
    prods = parallel_inner_prod(data, query)  # (n, m)
    indices = parallel_argtopk(prods, k)  # (n, k)
    dists = parallel_take_topk(prods, indices)  # (n, k)
    return indices, dists
