import numpy as np

def solve_system_test(solve_system):
    np.random.seed(seed=211)
    A = np.random.rand(3, 3)
    b = np.ones(3)
    k = 5
    x = solve_system(A, b, k)
    print(f"{x = }")
    print(np.allclose(np.linalg.matrix_power(A, k) @ x, b))


def LU_partial_pivot_test(mylu):
    A = np.array([[0, 2, -2, 2], [2, -1, 1, 0], [-2, 2, 0, -3], [-3, 3, 0, 0]])
    P, L, U = mylu(A)
    print(f"P_T = \n {P.T}")
    print(f"P = \n{P}")
    print(f"L = \n{L}")
    print(f"U = \n{U}")
    print(f"A = \n{A}")
    print(f"P^-1LU = \n{P.T @ L @ U}")


def myRowEchelon(A):
    """ Takes a nxn matrix and uses Gaussian
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