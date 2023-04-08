import jax
import jax.numpy as jnp
import numpy as np
import pytest

from annax.index.index import Index, IndexIVF, IndexPQ


@pytest.mark.parametrize("klass", [Index, IndexIVF, IndexPQ])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.bfloat16])
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
