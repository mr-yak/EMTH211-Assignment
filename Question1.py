import numpy as np

def myRowEchelon(A):
    """ Takes a 3x3 matrix and uses Gaussian
    elimination to find its row echelon form . """
    # Your code here
    n, m = A.shape
    if n != m:
        raise ValueError('Matrix must be square')
    for col in range(n-1):
        k = 0
        while A[col + k, col] == 0:
            k += 1
        A[col + k], A[col] = A[col], A[col + k]
        for row in range(col + 1, m):
            A[row] = A[row] - (A[row, col] / A[col, col]) * A[col]
    return A

def myLU(A):
    """ Takes a nxn numpy array and computes its
    LU decomposition without partial pivoting
    and assuming that no row swaps are required. """
    n, m = A.shape
    if n != m:
        raise ValueError('Matrix must be square')
    L = np.eye(n)
    U = A.copy()
    for col in range(n-1):
        k = 0
        while A[col + k, col] == 0:
            k += 1
        A[col + k], A[col] = A[col], A[col + k]
        for row in range(col + 1 + k, m):
            L[row, col] = U[row, col] / U[col, col]
            U[row] = U[row] - (U[row, col] / U[col, col]) * U[col]

    return L, U

def forwardSub(L, b):
    """Given a 3x3 lower triangular 2D array L and a
       1D array b with length 3, solves Lx = b with
       forward substitution"""
    n, m = A.shape
    if n != m:
        raise ValueError('Matrix must be square')
    x = []
    for row in range(n):
        x.append((b[row] - sum([L[row,i] * x_i for i, x_i in enumerate(x)]))/ L[row,row])
    return np.array(x)

def backSub(U, b):
    """Given a 3x3 upper triangular 2D array U and a
       1D array b with length 3, solves Ux = b with
       back substitution"""
    n, m = A.shape
    if n != m:
        raise ValueError('Matrix must be square')
    x = []
    for row in range(n):
        x.append((b[n-row-1] - sum([U[n-row-1,n-i-1] * x_i for i, x_i in enumerate(x)]))/ U[n-row-1,n-row-1])
    x = np.flip(np.array(x))
    return x

def solve_system(A , k , b ):
    """ Solves the system A**k @ x = b for nxn A """
    # Your code here
    if k <= 0:
        return False
    L, U = myLU(A)
    c_vec = b.copy()
    for inter in range(k):
        x_til = forwardSub(L, c_vec)
        x = backSub(U, x_til)
        c_vec = x.copy()
    return x


if __name__ == "__main__":
    np.random.seed(seed = 211)
    A = np.random.rand(3, 3)
    b = np.ones(3)
    k = 5
    x = solve_system(A , k , b)
    print (f"{x = }")
    print (np.allclose(np.linalg.matrix_power(A , k) @ x, b))
