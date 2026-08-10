from question_one import *

question = "Q1a Q1b1 Q1b2"

def main():
    if "Q1a" in question:
        A = np.array([
            [10.0, -4.0, -3.9, -5.9, 0.5],
            [1.0, -0.4, 2.4, -2.0, 3.7],
            [5.5, -2.2, 2.4, -2.0, 3.7],
            [9.9, -6.9, 5.0, 0.7, 6.2],
            [-7.5, 3.5, 5.9, 8.5, -6.2]
        ])
        P, L, U = myLU(A)
        print(f"P = \n{P}")
        print(f"L = \n{L}")
        print(f"U = \n{U}")
    if "Q1b1" in question:
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
    if "Q1b2" in question:
        ns = np.arange(2, 15)
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


def tests():
    #LU_partial_pivot_test(myLU)
    #solve_system_test(solve_system)
    pass

if __name__ == "__main__":
    #tests()
    main()