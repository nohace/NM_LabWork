'''
L = choleski(A).
Choleski decomposition: [L][L]^T = [A].
'''

from numpy import dot
from math import sqrt
import numpy as np

from input import read_system


def choleski(A):



    n = len(A)
    for k in range(n):
        try:
            A[k, k] = sqrt(A[k, k] - dot(A[k, 0:k], A[k, 0:k]))
        except ValueError:
            raise ValueError("Matrix is not positive-definite")

        for i in range(k + 1, n):
            A[i, k] = (A[i, k] - dot(A[i, 0:k], A[k, 0:k])) / A[k, k]

    # Zeroing out the upper triangle
    for k in range(1, n):
        A[0:k, k] = 0.0

    return A


def solve_choleski(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
        A = L L^T
        L y = b  (forward substitution)
        L^T x = y (back substitution)
    """
    L = choleski(A.copy())
    n = L.shape[0]

    # Forward substitution to solve L y = b
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - dot(L[i, :i], y[:i])) / L[i, i]

    # Back substitution to solve L^T x = y
    x = np.zeros_like(b, dtype=float)
    for i in reversed(range(n)):
        x[i] = (y[i] - dot(L[i + 1 :, i], x[i + 1 :])) / L[i, i]

    return x


if __name__ == "__main__":
    # Read A and b from the shared input module
    A, b = read_system()

    print("\nMatrix A:")
    print(A)
    print("\nVector b:")
    print(b)



    x = solve_choleski(A, b)

    print("\nSolution x to A x = b:")
    print(x)


