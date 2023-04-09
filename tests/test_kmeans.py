import jax
import numpy as np
import pytest

from annax.kmeans import KMeans, find_assignments


def test_kmeans():
    np.random.seed(0)
    data = np.random.random((100_000, 4)).astype(np.float32)
    kmeans = KMeans(data, k=100, n_iter=100)
    codebook = kmeans.fit()
    assignments = find_assignments(data, codebook)
    one_hot = jax.nn.one_hot(assignments, kmeans.k, dtype=data.dtype)
    count = one_hot.sum(axis=0)
    assert np.all(count > 100)
