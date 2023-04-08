from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from ..pfunc import pmap, pmap_zip
from .kmeans import find_assignments, kmeans


@partial(jax.jit, static_argnums=(1,))
def _prep(data: Array, sub_dim: int) -> Array:
    n, d = data.shape
    pad = (sub_dim - (d % sub_dim)) % sub_dim
    data = jnp.pad(data, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    data = data.reshape(n, -1, sub_dim)
    data = data.transpose(1, 0, 2)  # (d // s, n, s)
    return data


def pq(
    data: Array, sub_dim: int, k: int, n_iter: int = 100, batch_size: int = 1024, momentum: float = 0.7, seed: int = 42
) -> Array:
    data = _prep(data, sub_dim)
    f = partial(kmeans, k=k, n_iter=n_iter, batch_size=batch_size, momentum=momentum, seed=seed)
    codebooks = pmap(f)(data)  # (d // s, k, s)
    return codebooks


class EncodeP:
    def __init__(self):
        def f(x: Array, codebook: Array) -> Array:
            return find_assignments(x, codebook).astype(jnp.uint8)

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


class ProductQuantizer:
    k: int = 256

    def __init__(self, data: Array, sub_dim: int = 8, *, batch_size: int = 8192, n_iter: int = 1000) -> None:
        self.dim = data.shape[1]
        self.data = data
        self.sub_dim = sub_dim
        self.codebooks = None
        self.batch_size = min(batch_size, data.shape[0]) if batch_size > 0 else data.shape[0]
        self.n_iter = n_iter

    def _prep(self, data: Array) -> Array:
        return _prep(data, self.sub_dim)

    def fit(self) -> None:
        self.codebooks = pq(self.data, self.sub_dim, self.k, n_iter=self.n_iter, batch_size=self.batch_size)
        return self.codebooks

    def compute_codes(self, x: Array) -> Array:
        x = self._prep(x)  # (d // s, n, s)
        ret = encode(x, self.codebooks)  # (d // s, n)
        return ret

    def decode(self, codes: Array) -> Array:
        ret = decode(codes.T, self.codebooks)  # (d // s, n, s)
        ret = ret.transpose(1, 0, 2)  # (n, d // s, s)
        ret = ret.reshape(ret.shape[0], -1)  # (n, d)
        return ret[:, : self.dim]
