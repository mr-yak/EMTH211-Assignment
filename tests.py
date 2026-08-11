import numpy as np
from question_one import *
from question_two import *
from question_three import *

def solve_system_test():
    np.random.seed(seed=211)
    A = np.random.rand(3, 3)
    b = np.ones(3)
    k = 5
    x = solve_system(A, b, k)
    print(f"{x = }")
    print(np.allclose(np.linalg.matrix_power(A, k) @ x, b))


def LU_partial_pivot_test():
    #A = np.array([[0, 2, -2, 2], [2, -1, 1, 0], [-2, 2, 0, -3], [-3, 3, 0, 0]])
    A = np.random.rand(5, 5)
    P, L, U = myLU(A)
    print(f"P_T = \n {P.T}")
    print(f"P = \n{P}")
    print(f"L = \n{L}")
    print(f"U = \n{U}")
    print(f"A = \n{A}")
    print(f"P^-1LU = \n{P.T @ L @ U}")

def so_iterator_test(n, w):
    A = good_matrix(n, 100)
    b = np.ones(n)
    print(f"A = \n{A}")
    print(f"b = \n{b}")
    x = np.linalg.solve(A, b)
    print(f"x = \n{x}")
    x, i = sor(A, b, np.ones(n), w, max_iter =1000 , tol =1e-8 )
    print(f"x_so = \n{x}")
    print(f"iters = \n{i}")

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