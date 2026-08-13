"""
only part A done, will get onto part B later in the week, could you give it a quick check over make sure im not reading anything wrong?

also i saw you changed up Q2, if you want to do the same here go ahead, im just a bit lost with that so cant really help there
"""
from display import *
from tqdm import tqdm
import numpy as np
import pickle
import os

os.environ["NUMBA_CPU_NAME"] = "generic"

from numba import *

RECALCULATE = False

@njit
def sor(A, b, x0, w, max_iter =1000 , tol =1e-8 ):
    error = np.inf
    xk = x0.copy()
    n = len(x0)
    i = 0
    diag = np.diag(A)
    wd = w / diag
    while error > tol and i < max_iter:
        i += 1
        x_k1 = np.empty(n)
        for j in range(n):
            x_k1[j] = (
                (1-w) * xk[j]
                +  wd[j] *(
                    - np.dot(A[j, :j], x_k1[:j])
                    - np.dot(A[j, j+1:], xk[j+1:])
                    + b[j]
                )
            )
        error = np.linalg.norm((x_k1 - xk), ord = np.inf)/np.linalg.norm(x_k1, ord = np.inf)
        xk[:] = x_k1
    if error > tol:
        return xk, 1001
    return xk, i

def good_matrix(n , d ):
    rng = np.random.default_rng(seed = 211)
    B = rng.uniform(size =(n, n))
    return B.T @ B + d*n*np.eye(n)

def _calculate_d_and_w():
    b = np.ones(20)
    x0 = b.copy()
    ds = np.arange(0.01, 2.01, 0.01)
    ws = np.linspace(0.01, 1.99, 300)
    X = np.zeros((len(ds), len(ws)))
    Y = np.zeros((len(ds), len(ws)))
    Z = np.zeros((len(ds), len(ws)))
    for i, d in tqdm(enumerate(ds.tolist()), desc = "Data Crunching..."):
        Y[i] = np.full((len(ws),), d)
        A = good_matrix(20, d)
        iters = np.zeros(len(ws))
        for j, w in enumerate(ws.tolist()):
            x, num_iter = sor(A, b, x0, w, max_iter=1000)
            iters[j] = num_iter
        X[i] = ws
        Z[i] = iters
    with open("x.pkl", "wb") as file:
        pickle.dump(X, file)
    with open("y.pkl", "wb") as file:
        pickle.dump(Y, file)
    with open("z.pkl", "wb") as file:
        pickle.dump(Z, file)

def main():
    #a i)
    A = good_matrix(20, .5)
    b = np.ones(20)
    x0 = b.copy()

    ws = np.arange(0.01, 1.995, 0.005) #397 makes sure gs is included
    iters = np.array([])
    for i in ws:
        x, num_iter = sor(A, b, x0, i, max_iter=1000)
        iters = np.append(iters, num_iter)
    #ii / iii)
    gs_iter = iters[np.where(np.isclose(ws, 1.0))[0]][0]
    gs_iters = np.full(len(ws), gs_iter)
    ws_gt_gs = ws[iters <= gs_iters]
    best_w = ws[np.argmin(iters)]
    sor_iteration_plot(ws, iters, gs_iters, ws_gt_gs)
    width = 50
    print("#" * width)
    print(f"GS takes {gs_iter:.0f} iterations to converge")
    print(f"SOR is better that GS when {ws_gt_gs[0]:.3f} < ω < {ws_gt_gs[-1]:.3f}")
    print(f"SOR is best when ω = {best_w:.3f}, taking {iters[np.argmin(iters)]:.0f} iterations to converge")
    print("#" * width)
    #b)

    if RECALCULATE:
        #only do this calculation once, very expensive, writes to files
        _calculate_d_and_w()

    plot_surface()

if __name__ == "__main__":
    main()




