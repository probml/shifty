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

#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
#os.environ["XLA_FLAGS"] = "--intra_op_parallelism_threads=1"
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=90"

cpu_device, *_ = jax.devices("cpu")
cpu_count = os.cpu_count()
print('cpu count = ', cpu_count)

def serial_map(f, args):
    init_time = time()
    out1 = [f(arg) for arg in args]
    end_time = time()
    return out1, end_time - init_time

def parallel_map(f, args, n_jobs):
    init_time = time()
    out2 = Parallel(n_jobs=n_jobs, prefer="threads", verbose=1)(delayed(f)(arg) for arg in args)
    end_time = time()
    return out2, end_time - init_time

def f(arg):
    key, x = arg
    N = 6000
    mat = jr.normal(key, (N, N))
    return jnp.max(mat * mat * x)

KEY = jr.PRNGKey(42)
#KEY = jax.device_put(KEY, cpu_device)
args = list(zip(repeat(KEY), np.arange(200)))

out_serial, time_serial = serial_map(f, args)
print('Serial time', time_serial)

ntrials = [1, 10, 100, 500, 1000]
njobs_list = [1, 10, 20, 40]
times_parallel = []
n = len(njobs_list)
for i in range(n):
    njobs = njobs_list[i]
    out_parallel, time_parallel = parallel_map(f, args, njobs)
    assert jnp.allclose(jnp.array(out_serial), jnp.array(out_parallel))
    times_parallel.append(time_parallel)


print('Parallel times', times_parallel)


