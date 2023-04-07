from functools import partial
from typing import Tuple

import jax
import jax.numpy as np
from jax import Array

from .la import parallel_argtopk, parallel_inner_prod


@partial(jax.jit, static_argnums=(1,))
def naive_search(data: Array, query: Array, k: int = 1) -> Tuple[Array, Array]:
    prods = parallel_inner_prod(query, data)  # (n, m)
    indices = parallel_argtopk(prods, k=k)  # (n, k)
    dists = prods[np.arange(indices.shape[0])[:, None], indices]  # (n, k)
    return indices, dists
