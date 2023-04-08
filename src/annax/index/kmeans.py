from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .ops import inner_prod


@jax.jit
def find_assignments(data: Array, codebook: Array) -> Array:
    """Find the nearest codebook vector for each data point.

    Args:
        data (Array): data points with shape (n, d)
        codebook (Array): codebook vectors with shape (k, d)

    Returns:
        Array: indices of the nearest codebook vector for each data point
    """
    data_sq = jnp.sum(data**2, axis=-1)[:, None]  # (n, 1)
    codebook_sq = jnp.sum(codebook**2, axis=-1)[None, :]  # (1, k)
    prods = jax.vmap(inner_prod, in_axes=(0, None))(data, codebook)  # (n, k)
    dists = data_sq + codebook_sq - 2 * prods  # (n, k)
    return jnp.argmin(dists, axis=-1)  # (n,)


@partial(jax.jit, static_argnums=(3,))
def update_codebook(data: Array, assignments: Array, codebook: Array, momentum: float) -> Array:
    """Update the codebook vectors.

    Args:
        data (Array): data points with shape (n, d)
        assignments (Array): indices of the nearest codebook vector for each data point
        codebook (Array): codebook vectors with shape (k, d)
        momentum (float): momentum for the update

    Returns:
        Array: updated codebook vectors with shape (k, d)
    """
    k = codebook.shape[0]
    one_hot = jax.nn.one_hot(assignments, k, dtype=data.dtype)  # (n, k)
    sums = jnp.dot(one_hot.T, data)  # (k, d)
    counts = jnp.sum(one_hot, axis=0)  # (k,)
    mask = counts == 0
    counts = jnp.where(mask, 1, counts)[:, None]  # avoid division by zero
    new_codebook = sums / counts
    return momentum * codebook + (1 - momentum) * new_codebook


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def kmeans(
    data: Array, k: int, n_iter: int = 100, batch_size: int = 1024, momentum: float = 0.7, seed: int = 42
) -> Array:
    n = data.shape[0]

    prng_key = jax.random.PRNGKey(seed)
    idx = jnp.arange(n)
    idx = jax.random.permutation(prng_key, idx)[:k]
    codebook = data[idx].copy()

    @jax.jit
    def g(prng_key: Array, codebook: Array) -> Array:
        idx = jax.random.permutation(prng_key, jnp.arange(n))[:batch_size]
        one_hot = jax.nn.one_hot(idx, n, dtype=data.dtype)
        batch = jnp.dot(one_hot, data)
        assignments = find_assignments(batch, codebook)
        return update_codebook(batch, assignments, codebook, momentum)

    def f(i: int, codebook: Array) -> Array:
        prng_key = jax.random.PRNGKey(i + seed + 1)
        return g(prng_key, codebook)

    codebook = jax.lax.fori_loop(0, n_iter, f, codebook)
    assignments = find_assignments(data, codebook)
    one_hot = jax.nn.one_hot(assignments, k, dtype=data.dtype)
    count = one_hot.sum(axis=0)
    count = jnp.where(count == 0, 1, count)
    sums = jnp.dot(one_hot.T, data)
    return sums / count[:, None]


class KMeans:
    def __init__(
        self,
        data: Array,
        k: int,
        *,
        batch_size: int = 1024,
        momentum: float = 0.7,
        n_iter: int = 100,
        seed: int = 0,
    ) -> None:
        self.data = data
        self.k = k
        self.n_iter = n_iter
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.seed = seed

        self.batch_size = min(self.n, batch_size) if batch_size > 0 else data.shape[0]
        self.momentum = momentum

    def fit(self) -> Array:
        """Fit the codebook vectors.

        Returns:
            Array: codebook vectors with shape (k, d)
        """
        self.codebook = kmeans(self.data, self.k, self.n_iter, self.batch_size, self.momentum, self.seed)
        return self.codebook
