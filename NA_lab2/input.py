import numpy as np


def _read_n() -> int:
    """Helper to read the matrix/system size n."""
    while True:
        try:
            return int(input("Enter matrix size n (for an n x n matrix A): "))
        except ValueError:
            print("Please enter an integer for n.")


def read_matrix() -> np.ndarray:
    """
    Read a square matrix A from the keyboard.

    - First, ask for n (size of the matrix).
    - Then, for each of the n rows, ask for n numbers separated by spaces.

    Returns
    -------
    A : np.ndarray of shape (n, n)
    """
    n = _read_n()

    print("\nEnter the matrix A row by row.")
    print("For each row, enter", n, "numbers separated by spaces.")
    rows = []
    for i in range(n):
        while True:
            try:
                row_str = input(f"Row {i + 1} of A: ")
                row_vals = [float(x) for x in row_str.split()]
                if len(row_vals) != n:
                    print(f"Please enter exactly {n} numbers.")
                    continue
                rows.append(row_vals)
                break
            except ValueError:
                print("Could not parse numbers, please try again.")

    A = np.array(rows, dtype=float)
    return A


def read_system() -> tuple[np.ndarray, np.ndarray]:
    """
    Read a linear system A x = b from the keyboard.

    - First, ask for n (size of the system).
    - Then, read an n x n matrix A (same format as read_matrix).
    - Finally, read an n-vector b.

    Returns
    -------
    A : np.ndarray of shape (n, n)
    b : np.ndarray of shape (n,)
    """
    n = _read_n()

    print("\nEnter the coefficient matrix A row by row.")
    print("For each row, enter", n, "numbers separated by spaces.")
    rows = []
    for i in range(n):
        while True:
            try:
                row_str = input(f"Row {i + 1} of A: ")
                row_vals = [float(x) for x in row_str.split()]
                if len(row_vals) != n:
                    print(f"Please enter exactly {n} numbers.")
                    continue
                rows.append(row_vals)
                break
            except ValueError:
                print("Could not parse numbers, please try again.")

    A = np.array(rows, dtype=float)

    print("\nEnter the right-hand side vector b (", n, "numbers separated by spaces):")
    while True:
        try:
            b_str = input("b: ")
            b_vals = [float(x) for x in b_str.split()]
            if len(b_vals) != n:
                print(f"Please enter exactly {n} numbers.")
                continue
            b = np.array(b_vals, dtype=float)
            break
        except ValueError:
            print("Could not parse numbers, please try again.")

    return A, b
































def read_AX_equals_B() -> tuple[np.ndarray, np.ndarray]:
    """
    Read matrices A and B for a system A X = B.

    - First, ask for n (size of A: n x n).
    - Then, ask for m (number of columns of B).
    - Read A as an n x n matrix.
    - Read B as an n x m matrix.
    """
    n = _read_n()

    while True:
        try:
            m = int(input("Enter number of columns in B: "))
            if m <= 0:
                print("Number of columns must be positive.")
                continue
            break
        except ValueError:
            print("Please enter an integer for the number of columns.")

    print("\nEnter the matrix A row by row.")
    print("For each row, enter", n, "numbers separated by spaces.")
    rows_A = []
    for i in range(n):
        while True:
            try:
                row_str = input(f"Row {i + 1} of A: ")
                row_vals = [float(x) for x in row_str.split()]
                if len(row_vals) != n:
                    print(f"Please enter exactly {n} numbers.")
                    continue
                rows_A.append(row_vals)
                break
            except ValueError:
                print("Could not parse numbers, please try again.")

    A = np.array(rows_A, dtype=float)

    print("\nEnter the matrix B row by row.")
    print("For each row, enter", m, "numbers separated by spaces.")
    rows_B = []
    for i in range(n):
        while True:
            try:
                row_str = input(f"Row {i + 1} of B: ")
                row_vals = [float(x) for x in row_str.split()]
                if len(row_vals) != m:
                    print(f"Please enter exactly {m} numbers.")
                    continue
                rows_B.append(row_vals)
                break
            except ValueError:
                print("Could not parse numbers, please try again.")

    B = np.array(rows_B, dtype=float)
    return A, B


if __name__ == "__main__":
    # Simple manual test: read a system and print it.
    A, b = read_system()
    print("\nYou entered matrix A:")
    print(A)
    print("\nYou entered vector b:")
    print(b)


