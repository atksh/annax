import os

import pytest

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=7'

import jax
import jax.numpy as jnp
import numpy as np

import annax


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.float16])
def test_index(dtype):
    np.random.seed(0)
    data = np.random.random((1000, 128))

    # Create an Annax index with the default configuration
    index = annax.Index(data, dtype=dtype)

    # Query for the 10 nearest neighbors of a random vector
    query = np.random.random((10, 128))
    neighbors, distances = index.search(query, k=3)
    assert neighbors.shape == (10,3)
    assert distances.shape == (10,3)

    neighbors, distances = index.search(data[:10], k=5)
    assert np.all(neighbors[:, 0] == jnp.arange(10))
    gold = np.sum(data[:10].astype(dtype) ** 2, axis=1)
    assert np.all(gold > distances[:, 1])