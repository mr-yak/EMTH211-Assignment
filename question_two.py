# only need 10 numbers to see curve effectively
# Timings are pretty consistent each run after implementing averaging
# other application does affect outcome (1000x1000 = .006 vs .010 (s) while opening app)

import time
from display import *
import os
import pickle
from tqdm import tqdm

os.environ["NUMBA_CPU_NAME"] = "generic"

import numpy as np
from numba import njit

RECALCULATE = False

def main():
    print("#" * 25 + "QUESTION TWO" + "#" * 25)
    solvey(np.array([[1.]]), np.array([1.])) # jit compile step
    k = 50
    if RECALCULATE:
        rng = np.random.default_rng(seed=42)
        ns = np.arange(2, 1001, 2, dtype = int)
        times = np.empty((len(ns), k))
        for i, n in tqdm(enumerate(ns.tolist()), desc = "Time Crunching"):
            times[i] = calculate_timings(n, rng, k)
        with open("times.pkl", "wb") as file:
            pickle.dump([times, ns], file)
    print("\n")
    timing_plot(k)

def calculate_timings(n, rng, k):
    times = np.empty(k)
    for l in range(k):
        A = rng.random((n, n))  # random matrix
        b = rng.random(n)
        t0 = time.perf_counter()
        solvey(A, b) # only solving within times
        times[l] = (time.perf_counter() - t0)
    return times

@njit
def solvey(A,b):
    return np.linalg.solve(A, b)

if __name__ == "__main__":
    main()
