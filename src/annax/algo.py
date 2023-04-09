import os

os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .ivf import find_assignments_topk, find_assignments_topk_pq, single_ivf_search, single_ivf_search_pq
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


def pq_search(encoded_data: Array, encoded_query: Array, prod_tables: Array, k: int = 1) -> Tuple[Array, Array]:
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
        return single_ivf_search(data, data_clusters, query[i], near_indices[i], max_cluster_size, k)

    indices, dists = pmap_tuple(f)(jnp.arange(query.shape[0]))  # (m, k)
    return indices, dists


def ivfpq_search(
    encoded_data: Array,
    encoded_query: Array,
    encoded_codebook: Array,
    prod_tables: Array,
    data_clusters: Array,
    query: Array,
    codebook: Array,
    max_cluster_size: int,
    nprobe: int = 1,
    k: int = 1,
) -> Tuple[Array, Array]:
    near_indices = find_assignments_topk_pq(query, codebook, encoded_query, encoded_codebook, prod_tables, nprobe)

    def f(i):
        return single_ivf_search_pq(
            encoded_data, prod_tables, data_clusters, encoded_query[i], near_indices[i], max_cluster_size, k
        )

    indices, dists = pmap_tuple(f)(jnp.arange(encoded_query.shape[0]))  # (m, k)
    return indices, dists


def ivfpq_search(
    encoded_data: Array,
    encoded_query: Array,
    encoded_codebook: Array,
    prod_tables: Array,
    data_clusters: Array,
    query: Array,
    codebook: Array,
    max_cluster_size: int,
    nprobe: int = 1,
    k: int = 1,
) -> Tuple[Array, Array]:
    near_indices = find_assignments_topk_pq(query, codebook, encoded_query, encoded_codebook, prod_tables, nprobe)

    def f(i):
        return single_ivf_search_pq(
            encoded_data, prod_tables, data_clusters, encoded_query[i], near_indices[i], max_cluster_size, k
        )

    indices, dists = jax.vmap(f)(jnp.arange(encoded_query.shape[0]))  # (m, k)
    return indices, dists
