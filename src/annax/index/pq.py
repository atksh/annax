from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ..pfunc import pmap
from .kmeans import find_assignments, kmeans


def pq(data: Array, k: int, n_iter: int = 100, batch_size: int = 1024, momentum: float = 0.7, seed: int = 42) -> Array:
    assert data.ndim == 3
    f = partial(kmeans, k=k, n_iter=n_iter, batch_size=batch_size, momentum=momentum, seed=seed)
    return pmap(f)(data)


class ProductQuantizer:
    k: int = 256

    def __init__(self, data: Array, sub_dim: int = 8) -> None:
        self.sub_dim = sub_dim
        self.data = self._pad_data(data)

    def _pad_data(self, data: Array) -> Array:
        d = data.shape[1]
        pad = self.sub_dim - (d % self.sub_dim)
        if pad == self.sub_dim:
            return data
        return jnp.pad(data, ((0, 0), (0, pad)), mode="constant", constant_values=0)

    def fit(self) -> Array:
        n, d = self.data.shape
        s = self.sub_dim
        data = self.data.reshape(n, d // s, s)
        data = data.transpose(1, 0, 2)  # (d // s, n, s)
        return pq(data, self.k)
