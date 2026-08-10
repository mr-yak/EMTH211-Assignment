import question_one
import question_two
import matplotlib.pyplot as plt
from tests import *

question = "Q_1 Q2"

def main():
    if "Q1" in question:
        question_one.main()
    if "Q2" in question:
        question_two.main()
    plt.show()


def tests():
    #LU_partial_pivot_test(myLU)
    #solve_system_test(solve_system)
    pass

if __name__ == "__main__":
    #tests()
    main()