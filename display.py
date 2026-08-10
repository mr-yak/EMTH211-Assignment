import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def V(v):
    return r"\mathbf{\underline{" + v + r"}}"

def power_plot(ks, errors):
    ax = plt.axes()
    ax.plot(ks, errors, label = "Relative Error")
    ax.set_xlabel(r"k solving $\mathit{A}^k" + V(r"x") +" = " + V(r"b") + r"$")
    ax.set_ylabel(r"Error using $\infty$-norm $\frac{||" + V("x") + "-" + V(r"\tilde{x}") + r"||_{\infty}}{||" + V("x") + r"||_{\infty}}$")
    ax.set_title("Error vs k solving 5x5 orthogonal linear system")
    log_ax = ax.twinx()
    log_ax.set_ylabel(r"log scale")
    log_ax.set_yscale("log")
    log_ax.plot(ks, errors, label="Relative Error - Log Scale")

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
