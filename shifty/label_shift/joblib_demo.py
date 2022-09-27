import jax
import os
import dask
import jax.random as jr
import jax.numpy as jnp
from time import time
from itertools import repeat
import numpy as np

from joblib import Parallel, delayed, parallel_backend

# Limit jax multi-threading
# https://github.com/google/jax/issues/743

# https://joblib.readthedocs.io/en/latest/parallel.html#avoiding-over-subscription-of-cpu-resources

cpu_device, *_ = jax.devices("cpu")
cpu_count = os.cpu_count()
print('cpu count = ', cpu_count)

def f(arg):
    key, x = arg
    N = 6000
    mat = jr.normal(key, (N, N))
    return jnp.max(mat * mat * x)

KEY = jr.PRNGKey(42)
#KEY = jax.device_put(KEY, cpu_device)

args = list(zip(repeat(KEY), np.arange(200)))

init_time = time()
out1 = [f(arg) for arg in args]
out1 = np.array(out1)
end_time = time()
print(f"Serial Time elapsed: {end_time - init_time:.2f}s")

init_time = time()
out2 = Parallel(n_jobs=10, prefer="threads", verbose=1)(delayed(f)(arg) for arg in args)
#out2 = Parallel(n_jobs=-2, verbose=1)(delayed(f)(arg) for arg in args)
out2 = np.array(out2)
end_time = time()
print(f"Parallel Time elapsed: {end_time - init_time:.2f}s")

assert jnp.allclose(out1, out2)
