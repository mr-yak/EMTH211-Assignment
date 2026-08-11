import question_one
import question_two
import question_three
import matplotlib.pyplot as plt
from tests import *

question = "Q1 Q_2"

def main():
    np.set_printoptions(precision = 2)
    if "Q1" in question:
        question_one.main()
    if "Q2" in question:
        question_two.main()
    if "Q3" in question:
        question_three.main()
    plt.show()


def tests():
    #LU_partial_pivot_test()
    #solve_system_test()
    so_iterator_test(5, 3)
    pass

if __name__ == "__main__":
    tests()
    main()