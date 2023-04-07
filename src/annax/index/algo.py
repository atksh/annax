from typing import Tuple

import jax.numpy as np
from jax import Array

from ..pfunc import k_pmap, left_pjit
from .ops import argtopk, inner_prod

parallel_inner_prod = left_pjit(inner_prod)
parallel_argtopk = k_pmap(argtopk)


def naive_search(data: Array, query: Array, k: int = 1) -> Tuple[Array, Array]:
    prods = parallel_inner_prod(data, query)  # (n, m)
    indices = parallel_argtopk(prods, k)  # (n, k)
    dists = prods[np.arange(indices.shape[0])[:, None], indices]  # (n, k)
    return indices, dists
