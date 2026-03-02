"""
Exercise 5: solve A X = B using LUdecomp and LUsolve, and compute |A|.
"""

import numpy as np

from input import read_AX_equals_B
from LU_Decomposition_Method import LUdecomp, LUsolve


if __name__ == "__main__":
    # Read A (n x n) and B (n x m) for the system A X = B.
    A, B = read_AX_equals_B()

    print("\nMatrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    LUA = LUdecomp(A.copy())

    # Determinant of A from U (stored on the diagonal of LUA)
    det_A = float(np.prod(np.diag(LUA)))

    # Solve A x_j = b_j for each column b_j of B
    solutions = []
    for j in range(B.shape[1]):
        b_col = B[:, j].copy()
        x_col = LUsolve(LUA.copy(), b_col)
        solutions.append(x_col)

    X = np.column_stack(solutions)

    print("\nSolution matrix X to A X = B:")
    print(X)
    print("\n|A| (determinant of A):")
    print(det_A)

