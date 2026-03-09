

import numpy as np
from math import sqrt

from input import read_system


def gauss_seidel(A: np.ndarray,
                 b: np.ndarray,
                 x0: np.ndarray | None = None,
                 tol: float = 1.0e-9,
                 max_iter: int = 500,
                 use_adaptive_omega: bool = True) -> tuple[np.ndarray, int, float]:
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)

    # Initial guess
    if x0 is None:
        if np.any(np.diag(A) == 0.0):
            raise ValueError("Zero on diagonal, cannot form initial guess b_i / A_ii.")
        x = b / np.diag(A)
    else:
        x = np.asarray(x0, dtype=float).copy()

    omega = 1.0
    k_relax = 10  # index for relaxation
    p = 1        # parameter in relaxation
    dx1 = None

    for it in range(1, max_iter + 1):
        x_old = x.copy()

        # Standard Gauss-Seidel sweep with relaxation
        for i in range(n):

            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j]
            x_i_new = (b[i] - sigma) / A[i, i]
            x[i] = omega * x_i_new + (1.0 - omega) * x[i]

        dx = sqrt(float(np.dot(x - x_old, x - x_old)))

        if dx < tol:
            return x, it, omega

        if use_adaptive_omega:
            if it == k_relax:
                dx1 = dx
            if it == k_relax + p and dx1 is not None and dx < dx1:
                dx2 = dx
                ratio = (dx2 / dx1) ** (1.0 / p)
                if ratio < 1.0:
                    omega = 2.0 / (1.0 + sqrt(1.0 - ratio))

    print("Gauss-Seidel failed to converge within the maximum number of iterations.")
    return x, max_iter + 1, omega


if __name__ == "__main__":
    A, b = read_system()

    print("\nMatrix A:")
    print(A)
    print("\nVector b:")
    print(b)

    x, num_iter, omega = gauss_seidel(A, b)

    print("\nApproximate solution x:")
    print(x)
    print("\nNumber of iterations:")
    print(num_iter)
    print("\nFinal relaxation factor omega:")
    print(omega)

