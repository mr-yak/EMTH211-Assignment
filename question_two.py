# only need 10 numbers to see curve effectively
# Timings are pretty consistent each run after implementing averaging
# other application does affect outcome (1000x1000 = .006 vs .010 (s) while opening app)

import time
import numpy as np
from display import *
import os
import pickle
from tqdm import tqdm

os.environ["NUMBA_CPU_NAME"] = "generic"
from numba import njit, objmode

RECALCULATE = False

def main():
    print("#" * 25 + "QUESTION TWO" + "#" * 25)
    if RECALCULATE:
        ns = np.linspace(2, 1000, 999, dtype = int)
        times = np.zeros(len(ns))
        for i, n in tqdm(enumerate(ns.tolist()), desc = "Time Crunching"):
            avg_times =  np.zeros(10)
            for j in range(10):
                avg_times[j] = calculate_timings(n, times)
            times[i] = np.average(avg_times)
        with open("times.pkl", "wb") as file:
            pickle.dump([times, ns], file)
    print("\n")


    timing_plot()

@njit
def calculate_timings(n, times):
    A = np.random.rand(n, n) #random matrix
    b = np.random.rand(n) # random vector
    with objmode(t0='f8'):
        t0 = time.time()
    np.linalg.solve(A, b) # only solving within times
    with objmode(t1='f8'):
        t1 = time.time()
    delta_time = t1 - t0 # time takes
    return delta_time

"""
c)
theory and practice are similar but not identical, actual times from 2 - 999 are slightly greater on average
reasons as follows
1. flop_count != 2/3n**3, it also has n**2 & n factors
2. random variance in background applications/system noise
3.using n=10k, actual time trends to theoretical time
"""

if __name__ == "__main__":
    main()
