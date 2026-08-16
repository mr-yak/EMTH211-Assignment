import question_one
import question_two
import question_three
from tests import *
import matplotlib

question = "Q1 Q2 Q3"
fig_names = ["error_plot", "timing_plot", "SOR_vs_omega", "SOR_vs_omega_and_d"]

def main():
    np.set_printoptions(precision = 5)
    matplotlib.rcParams['savefig.format'] = 'svg'
    if "Q1" in question:
        question_one.main()
    if "Q2" in question:
        question_two.main()
    if "Q3" in question:
        question_three.main()
    for i in plt.get_fignums():
        if i == 4:
            matplotlib.rcParams['savefig.format'] = 'png'
        plt.figure(i)
        plt.savefig(f'plots/{fig_names[i-1]}')
        plt.close()
    plt.show()


def tests():
    #LU_partial_pivot_test()
    #solve_system_test()
    #so_iterator_test(5, 1.3)
    pass

if __name__ == "__main__":
    tests()
    main()