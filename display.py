import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from great_tables import GT
from unicodedata import decimal


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
    plt.show()

def hilbert_table(ns, resids, errors, conds):
    df = pd.DataFrame({
        "n": ns,
        "Relative Residual": resids,
        "Relative Error": errors,
        "Condition Number": conds,
    })
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(df.to_string(index = False))

