import jax
import os
import dask
import jax.random as jr
import jax.numpy as jnp
from time import time
from itertools import repeat

from dask.distributed import Client

cpu_device, *_ = jax.devices("cpu")
cpu_count = os.cpu_count()
client = Client(n_workers=cpu_count-1)
client


KEY = jax.random.PRNGKey(314)
KEY = jax.device_put(KEY, cpu_device)
N = 5000

def f(key, x):
    mat = jr.normal(key, (N, N))
    return jnp.max(mat * mat * x)

def fm(el):
    key, x = el
    mat = jr.normal(key, (N, N))
    return jnp.max(mat * mat * x)

def serial(key, xs):
    ys = [f(key, x) for x in xs]
    return ys


def parallel(key, xs, nproc):
    with Pool(processes=nproc) as pool:
        ys = pool.starmap(f, zip(repeat(key), xs))
    pool.close()
    pool.join()
    return ys

print('cpu count = ', cpu_count)
nproc = cpu_count - 1

xs = jnp.arange(50)
xs = jax.device_put(xs, cpu_device)
init_time = time()
y1 = serial(KEY, xs)
end_time = time()
print(f"Serial Time elapsed: {end_time - init_time:.2f}s")

init_time = time()
input_vals = [(key, x) for key, x in zip(repeat(KEY), xs)]
res_lazy = client.map(fm, input_vals)
y2 = client.gather(res_lazy)
end_time = time()
print(f"Serial Time elapsed: {end_time - init_time:.2f}s")

print([(y1v - y2v).item() for y1v, y2v in zip(y1, y2)])