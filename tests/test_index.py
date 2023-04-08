import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from annax.index.index import Index, IndexIVF, IndexIVFPQ, IndexPQ


@pytest.mark.parametrize("klass", [Index, IndexIVF, IndexPQ, IndexIVFPQ])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.bfloat16])
def test_index(klass, dtype):
    np.random.seed(0)
    data = np.random.random((1000, 128))

    # Create an Annax index with the default configuration
    index = klass(data, dtype=dtype)
    with io.BytesIO() as f:
        index.dump(f)
        f.seek(0)
        index = klass.load(f)

    # Query for the 10 nearest neighbors of a random vector
    query = np.random.random((10, 128))
    neighbors, distances = index.search(query, k=3)
    assert neighbors.shape == (10, 3)
    assert distances.shape == (10, 3)

    neighbors, distances = index.search(data[:10], k=5)
    assert np.mean(neighbors[:, 0] == jnp.arange(10)) >= 0.75
    gold = np.sum(data[:10].astype(dtype) ** 2, axis=1)
    assert np.mean(gold > distances[:, 1]) > 0.75
