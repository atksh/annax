import jax
import numpy as np
import pytest

from annax.index.pq import ProductQuantizer


def test_pq():
    np.random.seed(0)
    data = np.random.random((100_000, 8)).astype(np.float32)
    pq = ProductQuantizer(data, sub_dim=2)
    ret = pq.fit()
    print(ret.shape)
