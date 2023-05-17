# %%
import cProfile
import multiprocessing
import sys
import time
from typing import Iterable


def getsizeof_in(var, targetUnit):
    """
    
    """
    from sys import getsizeof
    shifts_to = {
        'Kb': 7,
        'Mb': 17,
        'Gb': 27,
        'KB': 10,
        'MB': 20,
        'GB': 30,
        'TB': 40,
        'PB': 50,
    } 
    return getsizeof(var)/float(1<<shifts_to[targetUnit])
    

import platform
platform.machine()
platform.version()
platform.platform()
platform.uname()
platform.system()
platform.processor()
import platform,re,uuid,json,psutil,logging


def get_system_info(mac_address: bool = False) -> dict:
    info = {}
    info['platform'] = platform.system()
    info['platform-release'] = platform.release()
    info['platform-version'] = platform.version()
    info['architecture'] = platform.machine()

    if mac_address:
        info['mac-address'] = ':'.join(
            re.findall('..', '%012x' % uuid.getnode()))
    
    info['processor'] = platform.processor()
    info['cpu_freq'] = str(psutil.cpu_freq())
    info['cpu_count'] = str(psutil.cpu_count())
    
    info['ram'] = str(
        round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
    return info


# import multiprocessing
# import cProfile
# import time
# def worker(num):
#     time.sleep(3)
#     print 'Worker:', num

# if __name__ == '__main__':
#     for i in range(5):
#         p = multiprocessing.Process(target=worker, args=(i,))
#         cProfile.run('p.start()', 'prof%d.prof' %i)
# %%

get_system_info()
# %%  # benchmark

import timeit

import numpy as np
import matplotlib.pyplot as plt


def iterative_mul(
        n: int = int(1e4), a: float = 11, b: float = 13) -> float:
    for _ in range(n):
        a *= b
    return a


def iterative_matmul(
        n: int = int(1e4), mat_shape = (2, 2)) -> np.ndarray:
    mat1 = np.random.randint(0, 100, np.prod(mat_shape)).reshape(mat_shape)
    mat2 = np.random.randint(0, 100, np.prod(mat_shape)).reshape(mat_shape)
    for _ in range(n):
        mat1 = np.matmul(mat1, mat2)
    return mat1


def measureit(
        f: callable,
        iterations: Iterable[int] = [1, 10, 50, 100]) -> np.ndarray:
    return {n_i: timeit.timeit(f, number=100) for n_i in iterations}


def plot_measure_dict(
        d: dict,
        xlabel: str = "# iterations",
        ylabel: str = "t[s]",
        ax: plt.Axes = None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(list(d.keys()), list(d.values()))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

m1 = measureit(iterative_mul)
plot_measure_dict(m1)
# %%

import cProfile
import re
cProfile.run('print("ciao")')

# %%

pr = cProfile.Profile()
for i in range(5):
    print(pr.calibrate(10))

# %%

import multiprocessing
import cProfile
import time
def worker(num):
    time.sleep(3)
    print ('Worker:', num)

if __name__ == '__main__':
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        cProfile.run('p.start()', 'prof%d.prof' %i)
# %%
