from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .kmeans import find_assignments, kmeans
from .pfunc import pmap, pmap_zip


@partial(jax.jit, static_argnums=(1,))
def prep(data: Array, sub_dim: int) -> Array:
    n, d = data.shape
    pad = (sub_dim - (d % sub_dim)) % sub_dim
    data = jnp.pad(data, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    data = data.reshape(n, -1, sub_dim)
    data = data.transpose(1, 0, 2)  # (d // s, n, s)
    return data


def pq(
    data: Array,
    sub_dim: int,
    k: int,
    n_iter: int = 100,
    batch_size: int = 1024,
    momentum: float = 0.7,
    seed: int = 42,
) -> Array:
    data = prep(data, sub_dim)
    f = partial(kmeans, k=k, n_iter=n_iter, batch_size=batch_size, momentum=momentum, seed=seed)
    codebooks = pmap(f)(data)  # (d // s, k, s)
    return codebooks


class EncodeP:
    def __init__(self):
        def f(x: Array, codebook: Array) -> Array:
            ret = find_assignments(x, codebook)
            return jax.lax.convert_element_type(ret, jnp.uint8)

        self.f = pmap_zip(f)

    def __call__(self, data: Array, codebooks: Array) -> Array:
        return self.f(data, codebooks).T


class DecodeP:
    def __init__(self):
        def f(code: Array, codebook: Array) -> Array:
            one_hot = jax.nn.one_hot(code, codebook.shape[0], dtype=codebook.dtype)
            return jnp.dot(one_hot, codebook)

        self.f = pmap_zip(f)

    def __call__(self, data: Array, codebooks: Array) -> Array:
        return self.f(data, codebooks)


p_encode = EncodeP()
p_decode = DecodeP()


def encode(data: Array, codebooks: Array) -> Array:
    ret = p_encode(data, codebooks)  # (d // s, n)
    return ret


def decode(codes: Array, codebooks: Array) -> Array:
    ret = p_decode(codes, codebooks)  # (d // s, n, s)
    return ret


@jax.jit
def calc_prod_table(codebooks: Array) -> Array:
    """Calculate dot product table for each codebook.

    Args:
        codebooks (Array): (d // s, k, s)

    Returns:
        Array: (d // s, k, k)
    """
    return jnp.einsum("nid,njd->nij", codebooks, codebooks)


@jax.jit
def lookup_prod_table(table: Array, x: Array, y: Array) -> Array:
    """Lookup dot product table for given data.

    Args:
        table (Array): (d // s, k, k)
        x (Array): encoded codes. (n, d // s)
        y (Array): encoded codes. (m, d // s)

    Returns:
        Array: (n, m)
    """
    x = x.T
    y = y.T

    @jax.jit
    def f(t, i, j):
        """Lookup dot product table for given data.

        Args:
            t (Array): (k, k)
            i (Array): encoded codes. (n,)
            j (Array): encoded codes. (m,)

        Returns:
            Array: (n, m)
        """
        return t[i].take(j, axis=1)

    sub_dist = jax.vmap(f, in_axes=(0, 0, 0))(table, x, y)  # (d // s, n, m)
    return sub_dist.sum(axis=0)


class ProductQuantizer:
    k: int = 256

    def __init__(
        self, data: Array, sub_dim: int = 8, *, batch_size: int = 8192, n_iter: int = 1000
    ) -> None:
        self.dim = data.shape[1]
        self.data = data
        self.sub_dim = sub_dim
        self.batch_size = min(batch_size, data.shape[0]) if batch_size > 0 else data.shape[0]
        self.n_iter = n_iter

        self.codebooks = None
        self.prod_table = None

    def prep(self, data: Array) -> Array:
        return prep(data, self.sub_dim)

    def fit(self) -> Tuple[Array, Array]:
        self.codebooks = pq(
            self.data, self.sub_dim, self.k, n_iter=self.n_iter, batch_size=self.batch_size
        )
        self.prod_table = calc_prod_table(self.codebooks)
        return self.codebooks, self.prod_table

    def compute_codes(self, x: Array) -> Array:
        """Compute codes for given data.

        Args:
            x (Array): (n, d)

        Returns:
            Array: (n, d // s)
        """
        x = self.prep(x)  # (d // s, n, s)
        ret = encode(x, self.codebooks)  # (d // s, n)
        return ret

    def decode(self, codes: Array) -> Array:
        """Decode codes to data.

        Args:
            codes (Array): (n, d // s)

        Returns:
            Array: (n, d)
        """
        ret = decode(codes.T, self.codebooks)  # (d // s, n, s)
        ret = ret.transpose(1, 0, 2)  # (n, d // s, s)
        ret = ret.reshape(ret.shape[0], -1)  # (n, d)
        return ret[:, : self.dim]

    def aprod(self, codes_x: Array, codes_y: Array) -> Array:
        """Approximate dot product between two code vectors.

        Args:
            codes_x (Array): (n, d // s)
            codes_y (Array): (m, d // s)

        Returns:
            Array: (n, m)
        """
        assert codes_x.shape[1] == codes_y.shape[1]
        assert self.prod_table is not None
        return lookup_prod_table(self.prod_table, codes_x, codes_y)
