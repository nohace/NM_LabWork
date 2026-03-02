"""
Doolittle's LU Decomposition Method.
"""

import numpy as np

from input import read_system


def doolittle_lu_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    U = np.zeros_like(A, dtype=float)

    for i in range(n):
        # Upper triangular matrix U
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        # Lower triangular matrix L
        for j in range(i, n):
            if i == j:
                L[i, j] = 1.0
            else:
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def backward_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return x


def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    L, U = doolittle_lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


if __name__ == "__main__":
    A, b = read_system()

    print("\nMatrix A:")
    print(A)
    print("\nVector b:")
    print(b)

    L, U = doolittle_lu_decomposition(A)
    print("\nL (Lower-triangular matrix):")
    print(L)
    print("\nU (Upper-triangular matrix):")
    print(U)

    x = solve_system(A, b)
    print("\nSolution x:")
    print(x)

