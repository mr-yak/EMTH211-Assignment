import numpy as np

def solve_system_test(solve_system):
    np.random.seed(seed=211)
    A = np.random.rand(3, 3)
    b = np.ones(3)
    k = 5
    x = solve_system(A, k, b)
    print(f"{x = }")
    print(np.allclose(np.linalg.matrix_power(A, k) @ x, b))


def LU_partial_pivot_test(mylu):
    A = np.array([[0, 2, -2, 2], [2, -1, 1, 0], [-2, 2, 0, -3], [-3, 3, 0, 0]])
    P_T, L, U = mylu(A)
    print(f"P_T = \n {P_T}")
    print(f"P = \n{P_T.T}")
    print(f"L = \n{L}")
    print(f"U = \n{U}")