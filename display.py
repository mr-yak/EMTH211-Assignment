import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np


def V(v):
    return r"\mathbf{\underline{" + v + r"}}"

def power_plot(ks, errors):
    fig1 = plt.figure(1, figsize = (6.4, 4.8))
    ax_1 = fig1.add_subplot(111)
    ax_1.plot(ks, errors, label = "Relative Error", color = "r")
    ax_1.set_xlabel(r"k solving $\mathit{A}^k" + V(r"x") +" = " + V(r"b") + r"$")
    ax_1.set_ylabel(r"Error using $\infty$-norm $\frac{||" + V("x") + "-" + V(r"\tilde{x}") + r"||_{\infty}}{||" + V("x") + r"||_{\infty}}$")
    ax_1.set_title("Error vs k solving 5x5 orthogonal linear system")
    fig1.tight_layout()

def hilbert_table(ns, resids, errors, conds):
    df = pd.DataFrame({
        "n": ns,
        "Relative Residual": resids,
        "Relative Error": errors,
        "Condition Number": conds,
    })
    print("\n")
    print("#" * 10 + "HILBERT MATRIX ERRORS" + "#" * 10)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(df.to_string(index = False))
    print("\n")
    df.style.format(to_scientific_latex).to_latex(
        "hilbert_table.tex",
        )

def to_scientific_latex(val):
    if val == 0:
        return "$0$"
    # Format to standard scientific string
    s = "{:.2e}".format(val)
    mantissa, exponent = s.split('e')
    # Clean up the sign and leading zeros from the exponent
    exponent = int(exponent)
    if exponent <= 2 and exponent >= -2:
        return f"{float(mantissa) * (10 ** exponent)}"
    return f"${mantissa} \\times 10^{{{exponent}}}$"

def timing_plot(k):
    fig2 = plt.figure(2, figsize = (6.4, 4.8))
    ax_2 = fig2.add_subplot(111)
    with open('times.pkl', 'rb') as file:
        data = pickle.load(file)  # ds
    ns = data[1]
    time_data = data[0]
    times = np.empty(len(ns))
    for i, time in enumerate(time_data.tolist()):
        times[i] = np.mean(time)

    # calculate theoretical using (count = 2/3*n**3)
    theoretical_time = np.median(times[-1]) * (ns / ns[-1]) ** 3
    ax_2.set_title(r"Time vs Matrix Size between $n = 1$ and $n = 1000$" + "\n" +f"averaged over {k} solves per " + r"$n$")
    ax_2.set_ylabel("Time [s]")
    ax_2.set_xlabel("Matrix Size (n)")
    ax_2.plot(ns, times, label='actual')
    ax_2.plot(ns, theoretical_time, label='theoretical')
    ax_2.legend()
    fig2.tight_layout()

def sor_iteration_plot(ws, iters, gs_iters, ws_gt_gs):
    fig3 = plt.figure(3, figsize = (9.6, 4.8))
    ax_3 = fig3.add_subplot(121)
    ax_4 = fig3.add_subplot(122)
    ax_4.set_yscale('log')
    ax_4.set_ylabel('Iteration Count (log scale)')
    ax_3.set_ylabel('Iteration Count')
    fig3.suptitle(r'Number of iterations for SOR to converge vs $\omega$')
    def _plot(ax_n):
        ax_n.set_xlabel(r'$\omega$ value')
        ax_n.plot( ws, gs_iters, label='Gauss-Seidel Iterations', color='orange')
        ax_n.plot(ws, iters, color="red", label='SOR Iterations')
        mask = iters <= gs_iters
        ax_n.fill_between(ws_gt_gs, gs_iters[mask], iters[mask], facecolor="none", hatch="xxx", edgecolor="orange")
        ax_n.legend(loc="upper right")
    _plot(ax_3)
    _plot(ax_4)
    fig3.tight_layout()

def plot_surface(from_file = True):
    if from_file:
        with open('X.pkl', 'rb') as file:
            X = pickle.load(file) # ds
        with open('Y.pkl', 'rb') as file:
            Y = pickle.load(file) # ws
        with open('Z.pkl', 'rb') as file:
            Z = pickle.load(file)
        with open('w_min.pkl', 'rb') as file:
            w_min = pickle.load(file)
        with open('GS.pkl', 'rb') as file:
            gs = pickle.load(file)
    gs = gs.T
    gs_iters = gs[1]
    w_min = w_min.T
    w_min_X = w_min[0]
    w_min_Y = w_min[1]
    w_min_Z = w_min[2]
    fig4 = plt.figure(4, figsize = (9.6, 6.8))
    ax_5 = plt.subplot2grid((2,3), (0,0), fig=fig4)
    ax_3d = plt.subplot2grid((2,3), (0,1), rowspan=2, colspan=2, projection='3d', fig=fig4)
    ax_6 = plt.subplot2grid((2,3), (1,0), fig=fig4)

    norm = plt.Normalize(Z.min(), Z.max())
    face_rgb = plt.colormaps["coolwarm"](norm(Z))
    face_rgb[..., 3] = 0.5
    ax_3d.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=face_rgb,
                        linewidth = 0, antialiased=True, shade=True)
    ax_3d.contour(X,Y,Z, linewidths = 3, zorder = 20, alpha = 1, levels = 20, cmap = plt.get_cmap("coolwarm"))
    ax_3d.yaxis.set_inverted(True)
    ax_3d.plot(w_min_X, w_min_Y, w_min_Z, label=r"Optimum $\omega$", linewidth=2, color='indigo', alpha=1, zorder = 20)
    ax_3d.legend(loc="upper right")
    ax_3d.set(xlabel=r'SOR $\omega$ parameter', ylabel=r'Diagonal dominance $d$',  zlabel='Iteration count')
    ax_3d.set_title("Iteration count until convergence of SOR" + "\n" r"algorithm vs $\omega$ and diagonal dominance $d$")
    ax_5.plot(w_min_Y, w_min_X)
    ax_5.set_ylabel(r'Optimal $\omega$ value')
    ax_5.set_xlabel(r'Diagonal dominance $d$ value')
    ax_5.set_title(r"Optimum $\omega$ for each diagonal dominance $d$ value")
    ax_6.plot(w_min_Y, (gs_iters/w_min_X))
    ax_6.set_title(r"Iteration ratio for each diagonal dominance $d$ value")
    ax_6.set_xlabel(r'Diagonal dominance $d$ value')
    ax_6.set_ylabel('Iteration Ratio')
    fig4.tight_layout()
