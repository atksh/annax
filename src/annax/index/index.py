from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .algo import ivf_search, naive_search, pq_search
from .kmeans import find_assignments, kmeans
from .pq import calc_prod_table, encode, pq, prep


class BaseIndex:
    def __init__(self, data: ArrayLike, *, dtype: jnp.dtype = jnp.float32) -> None:
        self._dtype = dtype
        self._meta = self._build(data)

    @property
    def meta(self) -> Optional[Dict[str, Array]]:
        return self._meta

    def _build(self, data: Array) -> Optional[Dict[str, Array]]:
        return {"data": self._asarray(data)}

    def _asarray(self, array: ArrayLike) -> Array:
        return jnp.asarray(array, dtype=self._dtype)

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
        return naive_search(self.meta["data"], query=query, k=k)


class IndexPQ(BaseIndex):
    def __init__(
        self,
        data: ArrayLike,
        *,
        sub_dim: int = 8,
        batch_size: int = 8192,
        n_iter: int = 10_000,
        k: int = 256,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.sub_dim = sub_dim
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.k = k
        super().__init__(data, dtype=dtype)

    def _build(self, data: Array) -> Dict[str, Array]:
        codebooks = pq(data, self.sub_dim, n_iter=self.n_iter, batch_size=self.batch_size, k=self.k)
        prod_tables = calc_prod_table(codebooks)
        encoded_data = encode(prep(data, self.sub_dim), codebooks)
        return {"codebooks": codebooks, "prod_tables": prod_tables, "encoded_data": encoded_data}

    def _search(self, query: Array, *, k: int = 1) -> Array:
        query = encode(prep(query, self.sub_dim), self.meta["codebooks"])
        return pq_search(self.meta["encoded_data"], query, self.meta["prod_tables"], k=k)


class IndexIVF(BaseIndex):
    def __init__(
        self,
        data: ArrayLike,
        *,
        nlist: int = 100,
        nprobe: int = 3,
        batch_size: int = 8192,
        n_iter: int = 10_000,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.batch_size = batch_size
        self.n_iter = n_iter
        super().__init__(data, dtype=dtype)

    def _build(self, data: Array) -> Dict[str, Array]:
        codebook = kmeans(data, self.nlist, n_iter=self.n_iter, batch_size=self.batch_size)
        data_clusters = find_assignments(data, codebook)
        max_cluster_size = int(jnp.max(jnp.bincount(data_clusters)))
        return {
            "codebook": codebook,
            "data_clusters": data_clusters,
            "data": data,
            "max_cluster_size": max_cluster_size,
        }

    def _search(self, query: Array, *, k: int = 1) -> Array:
        return ivf_search(
            self.meta["data"],
            query,
            self.meta["data_clusters"],
            self.meta["codebook"],
            self.meta["max_cluster_size"],
            self.nprobe,
            k=k,
        )
