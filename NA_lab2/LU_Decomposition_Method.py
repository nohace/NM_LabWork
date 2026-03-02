"""
Simple in-place LU decomposition method using LUdecomp and LUsolve.
"""

import numpy as np

from input import read_system


def LUdecomp(A: np.ndarray) -> np.ndarray:
    """
    Perform LU decomposition on matrix A in-place.
    The matrix A is modified to store L and U such that A = L * U.
    """
    n = len(A)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if A[i, k] != 0.0:
                lam = A[i, k] / A[k, k]
                A[i, k + 1 : n] -= lam * A[k, k + 1 : n]
                A[i, k] = lam
    return A


def LUsolve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(A)

    # Forward substitution
    for k in range(1, n):
        b[k] = b[k] - np.dot(A[k, 0:k], b[0:k])

    # Back substitution
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(A[k, k + 1 : n], b[k + 1 : n])) / A[k, k]

    return b


if __name__ == "__main__":
    A, b = read_system()

    print("\nMatrix A:")
    print(A)
    print("\nVector b:")
    print(b)

    LUA = LUdecomp(A.copy())
    x = LUsolve(LUA, b.copy())

    print("\nSolution x:")
    print(x)

