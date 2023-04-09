import jax
import numpy as np
import pytest

from annax.pq import ProductQuantizer


def test_pq():
    np.random.seed(0)
    data = np.random.random((1000, 33)).astype(np.float32)
    pq = ProductQuantizer(data, sub_dim=2)
    pq.fit()
    x = np.random.random((10, data.shape[1])).astype(np.float32)
    codes = pq.compute_codes(x)
    xa = pq.decode(codes)
    diff = xa - x
    assert np.all(np.abs(diff) / np.abs(x).max() < 0.1)
    ad = pq.aprod(codes[:6], codes[6:])
    d = jax.numpy.einsum("id,jd->ij", x[:6], x[6:])
    diff = ad - d
    assert np.all(np.abs(diff) / np.abs(d).max() < 0.1)
