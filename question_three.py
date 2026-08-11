import numpy as np
import scipy
from question_one import inf_norm

def good_matrix(n, d):
    rng = np.random.default_rng(seed = 211)
    B = rng.uniform(size = (n, n))
    return B.T @ B + d*n*np.eye(n)

def sor(A, b, x0, w, max_iter =1000 , tol =1e-8 ):
    error = inf_norm(x0)
    xk = x0.copy()
    i = 0
    while error < tol or i <= max_iter:
        i += 1
        x_k1 = xk.copy()
        for j, x in np.ndenumerate(xk):
            j = j[0]
            print(A[j,:j-2], x_k1[:j-2], j)
            x_k1 = np.append(x_k1, (1-w)*x + w/A[j,j] - A[j,:j-2] * x_k1[:j-2] - A[j,:i-2] * x_k1[:j-2] + b[j])
        error = inf_norm(x_k1 - xk)/inf_norm(x_k1)
        xk = x_k1
    if i == max_iter:
        return xk, -1
    else:
        return xk, i

def main():
    pass

if __name__ == "__main__":
    main()