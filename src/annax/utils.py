import jax
import jax.numpy as np


def pvmap(f, in_axes=0, out_axes=0):
    """Parallel vmap"""
    vec_f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    num_cores = jax.local_device_count()

    def pvec_f(x):
        n = x.shape[0]
        thre = (n // num_cores) * num_cores
        xb, xr = x[:thre], x[thre:]
        xb = xb.reshape(num_cores, n // num_cores, *x.shape[1:])
        yb = jax.pmap(vec_f)(xb)
        yr = vec_f(xr)
        yb = yb.reshape(-1, *yr.shape[1:])
        yb = jax.device_put(yb, device=yr.device_buffer.device())
        return np.concatenate([yb, yr], axis=0)

    return pvec_f
