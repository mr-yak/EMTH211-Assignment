import numpy as np
import scipy
from display import *
from tests import numpy_2_latex


def myLU(A, dtype = np.float64):
    """ Takes a nxn numpy array and computes its
    LU decomposition with partial pivoting"""
    A.astype(dtype)
    n, m = A.shape
    if n != m:
        raise ValueError('Matrix must be square')
    L = np.zeros((n, n), dtype= dtype)
    P = np.eye(n, dtype = dtype)
    U = A.copy()
    rows = np.arange(n)
    row_swaps = np.empty((0, 2))
    for col in range(n-1):
        pivot_row = rows[np.abs(U[rows,col]).argmax()]
        pivot = (U[:, col])[pivot_row]
        if pivot != 0:
            row_swaps = np.vstack([row_swaps, np.array([col, pivot_row])])
            rows = rows[rows != pivot_row]
            factors = U[rows, col] / pivot
            L[rows, col] = factors
            U[rows] -= factors[:, np.newaxis] * U[pivot_row]
    row_swaps = np.vstack([row_swaps, np.array([n-1, rows[0]])])
    row_swaps = np.astype(row_swaps, int)
    U[row_swaps.T[0]] = U[row_swaps.T[1]]
    P[row_swaps.T[0]] = P[row_swaps.T[1]]
    L = P @ L + np.eye(n)
    return P, L, U

def forwardSub(L, b, dtype = np.float64):
    """Given a nxn lower triangular 2D array L and a
       1D array b with length n, solves Lx = b with
       forward substitution"""
    L, b = L.astype(dtype), b.astype(dtype)
    n, m = L.shape
    if n != m:
        raise ValueError('Matrix must be square')
    x = np.zeros(n, dtype = dtype)
    for row in range(n):
        x[row] = (b[row] - np.dot(L[row,:row], x[:row]))/ L[row,row]
    return np.array(x)

def backSub(U, b, dtype = np.float64):
    """Given a nxn upper triangular 2D array U and a
       1D array b with length n, solves Ux = b with
       back substitution"""
    n, m = U.shape
    if n != m:
        raise ValueError('Matrix must be square')
    x = np.zeros(n, dtype = dtype)
    for row in range(n -1, -1, -1):
        x[row] = ((b[row] - np.dot(U[row,row+1:], x[row+1:]))/ U[row, row])
    return x

def solve_system(A, b, k=1, dtype = np.float64 ):
    """ Solves the system A**k @ x = b for nxn A """
    if k <= 0:
        return False
    A, b = A.astype(dtype), b.astype(dtype)
    P, L, U = myLU(A, dtype)
    x = b.copy()
    for inter in range(k):
        x_til = forwardSub(L, P @ x)
        x = backSub(U, x_til)
    return x

def inf_norm(v):
    return max(np.abs(v))

def main():
    print("#" * 25 + "QUESTION ONE" + "#" * 25)
    A = np.array([
        [10.0, -4.0, -3.9, -5.9, 0.5],
        [1.0, -0.4, 2.4, -2.0, 3.7],
        [5.5, -2.2, 2.4, -2.0, 3.7],
        [9.9, -6.9, 5.0, 0.7, 6.2],
        [-7.5, 3.5, 5.9, 8.5, -6.2]
    ])
    P, L, U = myLU(A)
    print(f"\nP = \n{P}\n")
    print(f"\nL = \n{L}\n")
    print(f"\nU = \n{U}\n")
    B = scipy.linalg.orth(np.random.rand(5, 5))
    B = B.astype(np.longdouble)
    x_true = np.ones(5, dtype=np.longdouble)
    errors = np.array([], dtype=np.longdouble)
    ks = np.arange(1, 201)
    for k in ks:
        b = np.linalg.matrix_power(B, k) @ x_true
        x_approx = solve_system(B, b, k, dtype = np.longdouble)
        errors = np.append(errors, inf_norm(x_true - x_approx)/inf_norm(x_true))
    power_plot(ks, errors)
    ns = np.arange(2, 16)
    residuals = np.zeros(ns.size)
    errors = np.zeros(ns.size)
    conds = np.zeros(ns.size)
    for i,n in np.ndenumerate(ns):
        C = scipy.linalg.hilbert(n)
        x = np.ones(n)
        d = C @ x
        x_tilde = solve_system(C, d)
        d_tilde = C @ x_tilde
        residuals[i] = scipy.linalg.norm(d - d_tilde)/ scipy.linalg.norm(d)
        errors[i] = np.linalg.norm(x - x_tilde) / np.linalg.norm(x)
        conds[i] = np.linalg.cond(C)
    hilbert_table(ns, residuals, errors, conds)

if __name__ == "__main__":
    main()


