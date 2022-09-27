import jax
import os
import dask
import jax.random as jr
import jax.numpy as jnp
from time import time
from itertools import repeat
import numpy as np

from dask.distributed import Client

cpu_device, *_ = jax.devices("cpu")
cpu_count = os.cpu_count()
print('cpu count = ', cpu_count)
client = Client(n_workers=cpu_count-1)
client


KEY = jax.random.PRNGKey(314)
KEY = jax.device_put(KEY, cpu_device)
N = 5000

def f(el):
    key, x = el
    mat = jr.normal(key, (N, N))
    return jnp.max(mat * mat * x)

xs = jnp.arange(50)
xs = jax.device_put(xs, cpu_device)
input_vals = [(key, x) for key, x in zip(repeat(KEY), xs)]

init_time = time()
y1 = [f(v) for v in input_vals]
end_time = time()
print(f"Serial Time elapsed: {end_time - init_time:.2f}s")

init_time = time()
res_lazy = client.map(f, input_vals)
y2 = client.gather(res_lazy)
end_time = time()
print(f"Serial Time elapsed: {end_time - init_time:.2f}s")

print(np.allclose(y1, y2))

#print([(y1v - y2v).item() for y1v, y2v in zip(y1, y2)])