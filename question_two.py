# only need 10 numbers to see curve effectivley
# Timings are pretty consistant each run after implementing averaging
# other application does effect outcome (1000x1000 = .006 vs .010 (s) while opening app)

import time
import numpy as np
from display import *

def main():
    times = np.array([])
    avg = []
    scaling_factor = []
    ns = np.linspace(2, 1000, 10, dtype = int)
    for n in ns:
        A = np.random.rand(n, n) #random matrix
        b = np.random.rand(n) # random vector
        t0 = time.time() #start
        np.linalg.solve(A, b) # only solving within times
        t1 = time.time() #end
        times = np.append(times, t1 - t0) # time takes
    timing_plot(ns, times)

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
