import numpy as np
import scipy
from tests import *
from display import *

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
        pivot_row = np.abs(U[rows,col]).argmax()
        pivot = (U[rows, col])[pivot_row]
        row_swaps = np.vstack([row_swaps, np.array([col, pivot_row])])
        rows = rows[rows != pivot_row]
        for row in rows:
            L[row, col] = U[row, col] / pivot
            U[row] = U[row] - (U[row, col] / pivot) * U[pivot_row]
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

