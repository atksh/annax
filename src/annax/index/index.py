from typing import Dict, Optional, Tuple

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from .algo import naive_search


class BaseIndex:
    def __init__(self, data: ArrayLike, *, dtype: np.dtype = np.float32) -> None:
        self._dtype = dtype
        self._data = self._asarray(data)
        self._meta = self._build(self.data)

    @property
    def data(self) -> Array:
        return self._data

    @property
    def meta(self) -> Optional[Dict[str, Array]]:
        return self._meta

    def _build(self, data: Array) -> Optional[Dict[str, Array]]:
        return None

    def _asarray(self, array: ArrayLike) -> Array:
        return np.asarray(array, dtype=self.dtype)

    def search(self, query: ArrayLike, *, k: int = 1) -> Tuple[Array, Array]:
        """Search for the k nearest neighbors of the query points.

        Args:
            query (ArrayLike): batch of query points with shape (n, d)
            k (int, optional): number of neighbors to search for. Defaults to 1.

        Returns:
            Tuple[Array, Array]: indices and (approx.) inner prod. of the k (approx.) nearest neighbors. Both have shape (n, k).
        """

        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        elif len(query.shape) > 2:
            raise ValueError(f"Query array must be 1- or 2-dimensional, got {len(query.shape)} dimensions")
        return self._search(self._asarray(query), k=k)

    def _search(self, query: Array, *, k: int = 1) -> Array:
        raise NotImplementedError


class Index(BaseIndex):
    def _search(self, query: Array, *, k: int = 1) -> Array:
        return naive_search(self.data, query=query, k=k)
