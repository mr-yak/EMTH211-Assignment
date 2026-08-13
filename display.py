import matplotlib.pyplot as plt
import pandas as pd
import pickle
from matplotlib import cbook
from matplotlib.colors import LightSource


plt.rcParams['figure.figsize'] = (10, 15)
fig = plt.figure()

def V(v):
    return r"\mathbf{\underline{" + v + r"}}"

def power_plot(ks, errors):
    ax = plt.subplot2grid((4, 2), (0,0))
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
    ax_2 = plt.subplot2grid((4, 2), (0,1))
    ax_2.set_title("Time vs Matrix Size between $n = 1$ and $n = 1000")
    ax_2.set_ylabel("Time [s]")
    ax_2.set_xlabel("Matrix Size (n)")
    ax_2.plot(ns, times, label='actual')
    ax_2.plot(ns, theoretical_time, label='theoretical')
    ax_2.legend()

def sor_iteration_plot(ws, iters, gs_iters, ws_gt_gs):
    ax_3 = plt.subplot2grid((4, 2), (1,0))
    ax_4 = plt.subplot2grid((4, 2), (1,1))
    ax_4.set_yscale('log')
    ax_4.set_ylabel('Number of iterations for SOR to converge (log scale)')
    ax_3.set_ylabel('Number of iterations for SOR to converge')
    def _plot(ax_n):
        ax_n.set_xlabel(r'$\omega$ value')
        ax_n.plot( ws, gs_iters, label='Gauss-Seidel Iterations', color='orange')
        ax_n.plot(ws, iters, color="red", label='SOR Iterations')
        mask = iters <= gs_iters
        ax_n.fill_between(ws_gt_gs, gs_iters[mask], iters[mask], facecolor="none", hatch="xxx", edgecolor="orange")
        ax_n.legend(loc="upper right")
    _plot(ax_3)
    _plot(ax_4)

def plot_surface(from_file = True):
    if from_file:
        with open('X.pkl', 'rb') as file:
            X = pickle.load(file) # ds
        with open('Y.pkl', 'rb') as file:
            Y = pickle.load(file) # ws
        with open('Z.pkl', 'rb') as file:
            Z = pickle.load(file)

    ax_3d = plt.subplot2grid((4, 2), (2,0), rowspan=2, colspan=2, projection='3d')
    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(Z, cmap=plt.colormaps["coolwarm"], vert_exag=0.1, blend_mode='soft')
    face_rgb = rgb.copy()
    face_rgb[..., 3] = 0.2
    surf = ax_3d.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=face_rgb,
                       linewidth=1, antialiased=False, shade=False)
    ax_3d.yaxis.set_inverted(True)
    ax_3d.set(xlabel=r'SOR $\omega$ parameter', ylabel=r'Diagonal dominance $d$',  zlabel='Iteration count')