import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec


def left_pjit(f):
    vec_f = jax.vmap(f, in_axes=(0, None), out_axes=0)
    num_cores = jax.local_device_count()
    in_shardings = (PartitionSpec("data", "dim"), PartitionSpec("query", "dim"))
    out_shardings = PartitionSpec("data", "query")

    shard_names = ("data", "query", "dim")
    mesh_shape = (num_cores, 1, 1)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape=mesh_shape)

    func = pjit(vec_f, in_shardings=in_shardings, out_shardings=out_shardings)

    def pjit_f(x, y):
        n = x.shape[0]
        thre = (n // num_cores) * num_cores
        xb, xr = x[:thre], x[thre:]
        yr = vec_f(xr, y)
        with Mesh(device_mesh, shard_names):
            yb = func(xb, y)
            yb = jax.device_get(yb)
        out = jnp.concatenate([yb, yr], axis=0).T
        return out

    return pjit_f


def k_pmap(f):
    vec_f = jax.vmap(f, in_axes=(0, None), out_axes=0)
    num_cores = jax.local_device_count()

    def func(x, k):
        n = x.shape[0]
        thre = (n // num_cores) * num_cores
        xb, xr = x[:thre], x[thre:]
        yr = vec_f(xr, k)
        xb = xb.reshape(num_cores, -1, *xb.shape[1:])
        yb = jax.pmap(lambda z: vec_f(z, k))(xb)
        yb = yb.reshape(-1, *yr.shape[1:])
        return jnp.concatenate([yb, yr], axis=0)

    return func


def pmap(f):
    vec_f = jax.vmap(f, in_axes=0, out_axes=0)
    num_cores = jax.local_device_count()

    def func(x):
        n = x.shape[0]
        thre = (n // num_cores) * num_cores
        xb, xr = x[:thre], x[thre:]
        yr = vec_f(xr)
        xb = xb.reshape(num_cores, -1, *xb.shape[1:])
        yb = jax.pmap(vec_f)(xb)
        yb = yb.reshape(-1, *yr.shape[1:])
        return jnp.concatenate([yb, yr], axis=0)

    return func


def pmap_zip(f):
    vec_f = jax.vmap(f, in_axes=(0, 0), out_axes=0)
    num_cores = jax.local_device_count()

    def func(x1, x2):
        n = x1.shape[0]
        assert x2.shape[0] == n
        thre = (n // num_cores) * num_cores
        x1b, x1r = x1[:thre], x1[thre:]
        x2b, x2r = x2[:thre], x2[thre:]
        yr = vec_f(x1r, x2r)
        x1b = x1b.reshape(num_cores, -1, *x1b.shape[1:])
        x2b = x2b.reshape(num_cores, -1, *x2b.shape[1:])
        yb = jax.pmap(vec_f)(x1b, x2b)
        yb = yb.reshape(-1, *yr.shape[1:])
        return jnp.concatenate([yb, yr], axis=0)

    return func
