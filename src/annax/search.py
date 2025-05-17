from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .ivf import (
    find_assignments_topk,
    single_ivf_search,
    single_ivf_search_pq,
)
from .ops import argtopk, inner_prod, take_topk
from .pfunc import k_pmap, left_pjit, pmap_tuple
from .pq import lookup_prod_table

parallel_inner_prod = left_pjit(inner_prod)
parallel_argtopk = k_pmap(argtopk)
parallel_take_topk = jax.vmap(take_topk, in_axes=(0, 0), out_axes=0)


def naive_search(data: Array, query: Array, k: int = 1) -> Tuple[Array, Array]:
    prods = parallel_inner_prod(data, query)  # (n, m)
    indices = parallel_argtopk(prods, k)  # (n, k)
    dists = parallel_take_topk(prods, indices)  # (n, k)
    return indices, dists


def pq_search(
    encoded_data: Array, encoded_query: Array, prod_tables: Array, k: int = 1
) -> Tuple[Array, Array]:
    prods = lookup_prod_table(prod_tables, encoded_query, encoded_data)  # (n, m)
    indices = parallel_argtopk(prods, k)  # (n, k)
    dists = parallel_take_topk(prods, indices)  # (n, k)
    return indices, dists


def ivf_search(
    data: Array,
    query: Array,
    data_clusters: Array,
    codebook: Array,
    max_cluster_size: int,
    nprobe: int = 1,
    k: int = 1,
) -> Tuple[Array, Array]:
    near_indices = find_assignments_topk(query, codebook, nprobe)  # (m, nprobe)

    def f(i):
        return single_ivf_search(
            data, data_clusters, query[i], near_indices[i], max_cluster_size, k
        )

    indices, dists = pmap_tuple(f)(jnp.arange(query.shape[0]))  # (m, k)
    return indices, dists


def ivfpq_search(
    encoded_data: Array,
    prod_tables: Array,
    data_clusters: Array,
    codebooks: Array,
    query: Array,
    codebook: Array,
    max_cluster_size: int,
    nprobe: int = 1,
    k: int = 1,
) -> Tuple[Array, Array]:
    """IVF search using residual product quantization.

    Args:
        encoded_data: PQ encoded residual data with shape (n, d // s).
        prod_tables: pre-computed product tables for the PQ codebooks.
        data_clusters: cluster assignment index for each data point.
        codebooks: PQ codebooks used for residual encoding.
        query: query points with shape (m, d).
        codebook: k-means centroids for IVF with shape (nlist, d).
        max_cluster_size: maximum size of a cluster.
        nprobe: number of clusters to probe.
        k: number of neighbors to return.

    Returns:
        Tuple containing indices and approximate inner products of the top-k
        nearest neighbors for each query.
    """

    near_indices = find_assignments_topk(query, codebook, nprobe)  # (m, nprobe)

    sub_dim = codebooks.shape[-1]
    residuals = query[:, None, :] - codebook[near_indices]  # (m, nprobe, d)
    flat = residuals.reshape(-1, residuals.shape[-1])  # (m * nprobe, d)
    encoded_query = encode(prep(flat, sub_dim), codebooks).T
    encoded_query = encoded_query.reshape(query.shape[0], nprobe, -1)  # (m, nprobe, d // s)

    def f(i):
        def g(j):
            return single_ivf_search_pq(
                encoded_data,
                prod_tables,
                data_clusters,
                encoded_query[i, j],
                near_indices[i, j],
                max_cluster_size,
                k,
            )

        idx, dst = jax.vmap(g)(jnp.arange(nprobe))  # (nprobe, k)
        idx = idx.reshape(-1)
        dst = dst.reshape(-1)
        top = argtopk(dst, k)  # (k,)
        return idx[top], dst[top]

    indices, dists = pmap_tuple(f)(jnp.arange(query.shape[0]))  # (m, k)
    return indices, dists
