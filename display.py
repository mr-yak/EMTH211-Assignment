import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def V(v):
    return r"\mathbf{\underline{" + v + r"}}"

def power_plot(ks, errors):
    ax = plt.axes()
    ax.plot(ks, errors, label = "Relative Error", color = "r")
    ax.set_xlabel(r"k solving $\mathit{A}^k" + V(r"x") +" = " + V(r"b") + r"$")
    ax.set_ylabel(r"Error using $\infty$-norm $\frac{||" + V("x") + "-" + V(r"\tilde{x}") + r"||_{\infty}}{||" + V("x") + r"||_{\infty}}$")
    ax.set_title("Error vs k solving 5x5 orthogonal linear system")

def hilbert_table(ns, resids, errors, conds):
    df = pd.DataFrame({
        "n": ns,
        "Relative Residual": resids,
        "Relative Error": errors,
        "Condition Number": conds,
    })
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(df.to_string(index = False))

def timing_plot(ns, times):
    """
    b)
    """
    # calculate theoretical using (count = 2/3*n**3)
    theoretical_time = times[-1] * (ns / ns[-1]) ** 3
    ax_2 = plt.axes()
    ax_2.set_title("Time vs Matrix Size between $n = 1$ and $n = 1000")
    ax_2.set_ylabel("Time [s]")
    ax_2.set_xlabel("Matrix Size (n)")
    ax_2.plot(ns, times, label='actual')
    ax_2.plot(ns, theoretical_time, label='theoretical')
    ax_2.legend()

def sor_iteration_plot(ws, iters, gs_iters, ws_gt_gs):
    fig, (ax_3, ax_4) = plt.subplots(1, 2, figsize=(12, 6))
    ax_4.set_yscale('log')
    ax_4.set_ylabel('Number of iterations for SOR to converge (log scale)')
    ax_3.set_ylabel('Number of iterations for SOR to converge')
    fig.suptitle(r"Number of iterations to converge vs $\omega$ using Successive Over Relaxation")
    def _plot(ax_n):
        ax_n.set_xlabel(r'$\omega$ value')
        ax_n.plot( ws, gs_iters, label='Gauss-Siedel Iterations', color='orange')
        ax_n.plot(ws, iters, color="red", label='SOR Iterations')
        mask = iters <= gs_iters
        ax_n.fill_between(ws_gt_gs, gs_iters[mask], iters[mask], facecolor="none", hatch="xxx", edgecolor="orange")
        ax_n.legend(loc="upper right")
    _plot(ax_3)
    _plot(ax_4)