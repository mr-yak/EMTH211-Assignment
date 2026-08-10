# only need 10 numbers to see curve effectivley
# Timings are pretty consistant each run after implementing averaging
# other application does effect outcome (1000x1000 = .006 vs .010 (s) while opening app)

import time
import matplotlib.pyplot as plt
import numpy as np

times = []
ns = []
avg = []
scaling_factor = []
theoretical_time = []
sizes = np.linspace(2, 1000, 10, dtype = int)



for n in sizes:
    A = np.random.rand(n, n) #random matrix
    b = np.random.rand(n) # random vector

    t0 = time.time() #start
    np.linalg.solve(A, b) # only solving within times
    t1 = time.time() #end
    times.append(t1 - t0) # time takes
    ns.append(n)

"""
b)
"""
# calculate theoretical using (count = 2/3*n**3)
theoretical_time = [times[-1] * (n / sizes[-1])**3 for n in sizes]

plt.title("curve of time vs nxn matrix size")
plt.ylabel("time")
plt.xlabel("matrix size")
plt.plot(ns, times, label = 'actual')
plt.plot(ns, theoretical_time, label ='theoretical')
plt.legend()
plt.show()


"""
c)
theory and practice are similar but not identical, actual times from 2 - 999 are slightly greater on average
reasons as follows
1. flop_count != 2/3n**3, it also has n**2 & n factors
2. random variance in background applications/system noise
3.using n=10k, actual time trends to theoretical time
"""
