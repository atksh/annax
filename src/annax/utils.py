import jax


def pvmap(f, in_axes=0, out_axes=0):
    """Parallel vmap"""
    vec_f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    num_cores = jax.local_device_count()
    if num_cores == 1:
        return vec_f
    else:

        @jax.jit
        def pvec_f(x):
            n = x.shape[0]
            thre = (n // num_cores) * num_cores
            xb, xr = x[:thre], x[thre:]
            xb = xb.reshape(num_cores, -1, *xb.shape[1:])
            yb = jax.pmap(vec_f)(xb)
            yr = vec_f(xr)
            return jax.device_put(np.concatenate([yb, yr], axis=0))

        return pvec_f
