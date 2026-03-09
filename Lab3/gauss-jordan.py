

import numpy as np

from input import read_system, read_matrix
from fractions import Fraction


def _as_fraction_str(x: float, max_den: int = 10000) -> str:
    if np.isfinite(x) and abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return str(Fraction(x).limit_denominator(max_den))


def format_fractions(arr: np.ndarray, max_den: int = 10000) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.empty(arr.shape, dtype=object)
    it = np.nditer(arr, flags=["multi_index"])
    for v in it:
        out[it.multi_index] = _as_fraction_str(float(v), max_den=max_den)
    return out


def gauss_jordan_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(A)
    a = A.astype(float).copy()
    c = b.astype(float).copy()

    for k in range(n):
        # Partial pivoting
        p = np.argmax(np.abs(a[k:, k])) + k
        if a[p, k] == 0.0:
            raise ValueError("Matrix is singular.")
        if p != k:
            a[[k, p]] = a[[p, k]]
            c[[k, p]] = c[[p, k]]

        # Make the pivot 1
        piv = a[k, k]
        a[k, k:] = a[k, k:] / piv
        c[k] = c[k] / piv

        # Eliminate the k-th column in all other rows
        for i in range(n):
            if i == k:
                continue
            lam = a[i, k]
            if lam != 0.0:
                a[i, k:] = a[i, k:] - lam * a[k, k:]
                c[i] = c[i] - lam * c[k]

    return c


def gauss_jordan_inverse(A: np.ndarray) -> np.ndarray:
    n = len(A)
    a = A.astype(float).copy()
    inv = np.eye(n, dtype=float)

    for k in range(n):
        # Partial pivoting
        p = np.argmax(np.abs(a[k:, k])) + k
        if a[p, k] == 0.0:
            raise ValueError("Matrix is singular.")
        if p != k:
            a[[k, p]] = a[[p, k]]
            inv[[k, p]] = inv[[p, k]]

        # Make the pivot 1
        piv = a[k, k]
        a[k, :] = a[k, :] / piv
        inv[k, :] = inv[k, :] / piv

        # Eliminate the k-th column in all other rows
        for i in range(n):
            if i == k:
                continue
            lam = a[i, k]
            if lam != 0.0:
                a[i, :] = a[i, :] - lam * a[k, :]
                inv[i, :] = inv[i, :] - lam * inv[k, :]

    return inv


if __name__ == "__main__":
    mode = input("(1) solve Ax=b, (2) invert A").strip()

    if mode == "2":
        A = read_matrix()
        print("\nMatrix A:")
        print(A)
        A_inv = gauss_jordan_inverse(A)
        print("\nA^{-1}:")
        print(A_inv)
        print("\nA^{-1} (fractions):")
        print(format_fractions(A_inv))
    else:
        A, b = read_system()
        print("\nMatrix A:")
        print(A)
        print("\nVector b:")
        print(b)
        x = gauss_jordan_solve(A, b)
        print("\nSolution x:")
        print(x)
        print("\nSolution x (fractions):")
        print(format_fractions(x))

