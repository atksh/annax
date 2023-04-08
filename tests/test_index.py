import os

import pytest

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np

from annax.index.index import Index, PQIndex


@pytest.mark.parametrize("klass", [Index, PQIndex])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.float16])
def test_index(klass, dtype):
    np.random.seed(0)
    data = np.random.random((1000, 128))

    # Create an Annax index with the default configuration
    index = klass(data, dtype=dtype)

    # Query for the 10 nearest neighbors of a random vector
    query = np.random.random((10, 128))
    neighbors, distances = index.search(query, k=3)
    assert neighbors.shape == (10, 3)
    assert distances.shape == (10, 3)

    neighbors, distances = index.search(data[:10], k=5)
    assert np.all(neighbors[:, 0] == jnp.arange(10))
    gold = np.sum(data[:10].astype(dtype) ** 2, axis=1)
    assert np.all(gold > distances[:, 1])
