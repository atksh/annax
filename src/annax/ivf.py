from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .ops import argtopk, inner_prod, nonzero_idx
from .pq import lookup_prod_table


@partial(jax.jit, static_argnums=(2,))
def find_assignments_topk(data: Array, codebook: Array, k: int) -> Array:
    """Find the top-k nearest codebook vectors for each data point.

    Args:
        data (Array): data points with shape (n, d)
        codebook (Array): codebook vectors with shape (m, d)
        k (int): number of nearest codebook vectors to find

    Returns:
        Array: indices of the nearest codebook vector for each data point
    """
    data_sq = jnp.sum(data**2, axis=-1)[:, None]  # (n, 1)
    codebook_sq = jnp.sum(codebook**2, axis=-1)[None, :]  # (1, m)
    prods = jax.vmap(inner_prod, in_axes=(0, None))(data, codebook)  # (n, m)
    dists = data_sq + codebook_sq - 2 * prods  # (n, m)
    return jax.vmap(partial(argtopk, k=k))(-dists)  # (n, k)


@partial(jax.jit, static_argnums=(5,))
def find_assignments_topk_pq(
    data: Array, codebook: Array, encoded_data: Array, encoded_codebook: Array, prod_tables: Array, k: int
) -> Array:
    """Find the top-k nearest codebook vectors for each data point.

    Args:
        data (Array): data points with shape (n, d)
        codebook (Array): codebook vectors with shape (m, d)
        encoded_data (Array): encoded data points with shape (n, s)
        encoded_codebook (Array): encoded codebook vectors with shape (m, s)
        prod_tables (Array): product tables with shape (d // s, k, k)
        k (int): number of nearest codebook vectors to find

    Returns:
        Array: indices of the nearest codebook vector for each data point
    """
    data_sq = jnp.sum(data**2, axis=-1)[:, None]  # (n, 1)
    codebook_sq = jnp.sum(codebook**2, axis=-1)[None, :]  # (1, m)
    prods = lookup_prod_table(prod_tables, encoded_data, encoded_codebook)  # (n, m)
    dists = data_sq + codebook_sq - 2 * prods  # (n, m)
    return jax.vmap(partial(argtopk, k=k))(-dists)  # (n, k)


@partial(jax.jit, static_argnums=(4, 5))
def single_ivf_search(
    data: Array, data_cluster: Array, query: Array, near_index: Array, max_cluster_size: int, k: int
) -> Tuple[Array, Array]:
    """Find the nearest neighbor for each data point.

    Args:
        data (Array): data points with shape (n, d)
        data_cluster (Array): indices of the nearest codebook vector for each data point with shape (n,)
        query (Array): a query point with shape (d,)
        near_index (Array): index of the nearest codebook vector for the query point with shape (nprobe,)
        max_cluster_size (int): maximum number of data points in a cluster
        k (int): number of nearest neighbors to find

    Returns:
        Array: indices of the nearest neighbor for each data point
    """
    assert query.ndim == 1

    def f(i):
        idx = nonzero_idx(data_cluster == i, max_cluster_size)  # (count,)
        subset_data = data.take(idx, axis=0)  # (?, d)
        prods = jax.vmap(inner_prod, in_axes=(0, None))(subset_data, query)  # (?,)
        # pad prods with -1e5
        prods = jnp.pad(prods, (k, 0), constant_values=-1e5)
        ret = argtopk(prods, k=k) - k  # (k,)
        return jnp.where(ret < 0, idx[0], idx[ret])

    indices = jax.vmap(f)(near_index).reshape(-1)  # (nprobe * k)
    prods = jax.vmap(inner_prod, in_axes=(0, None))(data[indices], query)  # (nprobe * k,)
    idx = argtopk(prods, k=k)  # (k,)
    return indices[idx], prods[idx]  # (k,), (k,)


@partial(jax.jit, static_argnums=(5, 6))
def single_ivf_search_pq(
    encoded_data: Array,
    prod_tables: Array,
    data_cluster: Array,
    encoded_query: Array,
    near_index: Array,
    max_cluster_size: int,
    k: int,
) -> Tuple[Array, Array]:
    """Find the nearest neighbor for each data point.

    Args:
        encoded_data (Array): data points with shape (n, d)
        prod_tables (Array): product tables with shape (d // s, k, k)
        data_cluster (Array): indices of the nearest codebook vector for each data point with shape (n,)
        encoded_query (Array): a query point with shape (d,)
        near_index (Array): index of the nearest codebook vector for the query point with shape (nprobe,)
        max_cluster_size (int): maximum number of data points in a cluster
        k (int): number of nearest neighbors to find

    Returns:
        Array: indices of the nearest neighbor for each data point
    """
    assert encoded_query.ndim == 1

    def f(i):
        idx = nonzero_idx(data_cluster == i, max_cluster_size)  # (count,)
        subset_data = encoded_data.take(idx, axis=0)  # (?, d)
        prods = lookup_prod_table(prod_tables, subset_data, encoded_query)  # (?,)
        # pad prods with -1e5
        prods = jnp.pad(prods, (k, 0), constant_values=-1e5)
        ret = argtopk(prods, k=k) - k  # (k,)
        return jnp.where(ret < 0, idx[0], idx[ret])

    indices = jax.vmap(f)(near_index).reshape(-1)  # (nprobe * k)
    prods = lookup_prod_table(prod_tables, encoded_data[indices], encoded_query)  # (nprobe * k,)
    idx = argtopk(prods, k=k)  # (k,)
    return indices[idx], prods[idx]  # (k,), (k,)
