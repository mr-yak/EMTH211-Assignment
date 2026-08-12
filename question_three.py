"""
only part A done, will get onto part B later in the week, could you give it a quick check over make sure im not reading anything wrong?

also i saw you changed up Q2, if you want to do the same here go ahead, im just a bit lost with that so cant really help there
"""
import numpy as np
from question_one import *
from display import *
'''
def forwardSub(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    return y
'''
#changing the gauss seidel stuff from learn, unsure if the equation is right
"""
def sor(A , b , x0 , w , max_iter =1000 , tol =1e-8 ): 
    n = x0.shape[0]
    x = x0.copy()
    results = []
    converged = False
    for k in range(max_iter):
        x_new = np.empty_like(x)
        for i in range(n):
                x_new[i] =(1-w) * x[i] + w * (b[i] - (A[i,:i].dot(x_new[:i])) - (A[i,i+1:].dot(x[i+1:])))/A[i,i]        
        results.append([k+1,*x_new])
        err = np.linalg.norm(x_new - x,np.inf)/np.linalg.norm(x_new,np.inf)
        if err < tol:
            converged = True
            break
        x = x_new.copy()
    return np.array(results), converged
"""
def sor(A, b, x0, w, max_iter =1000 , tol =1e-8 ):
    error = np.inf
    xk = x0.copy()
    i = 0
    while error > tol and i <= max_iter:
        i += 1
        x_k1 = np.array([])
        for j in range(len(xk)):
            x_k1 = np.append(
                x_k1,
                (1-w)*xk[j]
                + (w/A[j,j])*(
                    - np.dot(A[j, :j], x_k1[:j])
                    - np.dot(A[j, j+1:], xk[j+1:])
                    + b[j]
                )
            )
        error = inf_norm(x_k1 - xk)/inf_norm(x_k1)
        xk = x_k1
    if i >= max_iter:
        return xk, 1000
    else:
        return xk, i

#from learn page 
def gauss_seidel(A, b, x0, max_iter=100, tol=1e-3): # adding in to compare to SOR (a. iii)

    P = np.tril(A)
    solver = forwardSub

    Q = P - A
    converged = False
    x = x0.copy()
    results = []
    for k in range(max_iter):
        rhs = Q @ x + b
        x_new = solver(P,rhs)
        results.append([k+1,*x_new])
        err = np.linalg.norm(x_new - x,np.inf)/np.linalg.norm(x_new,np.inf)
        if err < tol:
            converged = True
            break
        x = x_new.copy()
    return np.array(results), k

def good_matrix(n , d ):
    rng = np.random.default_rng(seed = 211)
    B = rng.uniform(size =(n, n))
    return B.T @ B + d*n*np.eye(n)

def main():
    """
    a. i)
    """
    n = 20
    d = .5
    A = good_matrix(n, d)
    b = np.ones(n)
    x0 = b.copy()

    ws = np.arange(0.01, 1.995, 0.005) #397 makes sure gs is included
    iters = np.array([])
    for i in ws:
        x, num_iter = sor(A, b, x0, i, max_iter=1000)
        iters = np.append(iters, num_iter)
    """
    a. ii)
    """
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
if __name__ == "__main__":
    main()




