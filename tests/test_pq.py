import jax
import numpy as np
import pytest

from annax.index.pq import ProductQuantizer


def test_pq():
    np.random.seed(0)
    data = np.random.random((100_000, 128)).astype(np.float32)
    pq = ProductQuantizer(data, sub_dim=8)
    pq.fit()
    x = np.random.random((10, 128)).astype(np.float32)
    codes = pq.compute_codes(x)
    xa = pq.decode(codes)
    diff = xa - x
    assert np.all(np.abs(diff) / np.abs(x).max() < 0.5)
