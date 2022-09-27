from multiprocessing import Pool, TimeoutError
import time
import os
import numpy as np
from time import time
import jax.numpy as jnp
import jax
import jax.random as jr
from itertools import repeat

KEY = jr.PRNGKey(42)
N = 200
#MAT = jnp.reshape(np.arange(N*N), (N, N))
MAT = jr.normal(KEY, (N, N))

def f(key, x):
    mat = jr.normal(key, (N, N))
    return jnp.max(mat * mat * x)

#def f(x):
    #return jnp.max(MAT * MAT * x)


def serial(key, xs):
    ys = [f(key, x) for x in xs]
    return ys

def parallel(key, xs, nproc):
    with Pool(processes=nproc) as pool:
        ys = pool.starmap(f, zip(repeat(key), xs))
    pool.close()
    pool.join()
    return ys

def main(): 
    cpu_count = os.cpu_count()
    print('cpu count = ', cpu_count)
    nproc = cpu_count - 1

    xs = np.arange(10)
    init_time = time()
    y1 = serial(KEY, xs)
    end_time = time()
    print(f"Serial Time elapsed: {end_time - init_time:.2f}s")

    init_time = time()
    y2 = parallel(KEY, xs, nproc)
    end_time = time()
    print(f"Parallel Time elapsed: {end_time - init_time:.2f}s")

    assert np.allclose(y1, y2)
    #print('result' , y1)

if __name__ == '__main__':
    main()