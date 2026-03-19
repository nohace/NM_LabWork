import os
from typing import List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt


def ensure_figures_dir(path: str = "figures") -> str:
	"""
	Ensure the figures directory exists and return its absolute path.
	"""
	abs_path = os.path.abspath(path)
	os.makedirs(abs_path, exist_ok=True)
	return abs_path


# -----------------------------
# Interpolation: Lagrange (from scratch)
# -----------------------------
def lagrange_interpolate(x_nodes: List[float], y_nodes: List[float], x_eval: np.ndarray) -> np.ndarray:
	"""
	Evaluate the Lagrange interpolating polynomial defined by (x_nodes, y_nodes) at x_eval.
	This implementation computes values directly without forming coefficients.
	"""
	x_nodes = np.asarray(x_nodes, dtype=float)
	y_nodes = np.asarray(y_nodes, dtype=float)
	x_eval = np.asarray(x_eval, dtype=float)

	n = len(x_nodes)
	if n != len(y_nodes):
		raise ValueError("x_nodes and y_nodes must be the same length.")

	y_eval = np.zeros_like(x_eval, dtype=float)
	for i in range(n):
		# Compute L_i(x)
		den = 1.0
		for j in range(n):
			if i != j:
				den *= (x_nodes[i] - x_nodes[j])
		if den == 0:
			raise ValueError("Duplicate x_nodes are not allowed.")
		li = np.ones_like(x_eval, dtype=float)
		for j in range(n):
			if i != j:
				li *= (x_eval - x_nodes[j])
		y_eval += y_nodes[i] * (li / den)
	return y_eval


# -----------------------------
# Interpolation: Newton Divided Differences (from scratch)
# -----------------------------
def newton_divided_differences(x_nodes: List[float], y_nodes: List[float]) -> np.ndarray:
	"""
	Compute Newton's divided differences table coefficients a_0, a_1, ..., a_{n-1}.
	Returns the first row of the Newton table which are the coefficients for the nested form.
	"""
	x = np.asarray(x_nodes, dtype=float)
	y = np.asarray(y_nodes, dtype=float)
	n = len(x)
	if n != len(y):
		raise ValueError("x_nodes and y_nodes must be the same length.")
	if n == 0:
		return np.array([])

	# Copy y into a working array for in-place divided differences
	coef = y.astype(float).copy()
	for j in range(1, n):
		for i in range(n - 1, j - 1, -1):
			den = x[i] - x[i - j]
			if den == 0:
				raise ValueError("Duplicate x_nodes are not allowed.")
			coef[i] = (coef[i] - coef[i - 1]) / den
	return coef  # coef[i] holds a_i for i in [0..n-1] when paired with x in nested evaluation


def newton_evaluate(x_nodes: List[float], coef: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
	"""
	Evaluate the Newton interpolating polynomial with divided differences 'coef'
	at points x_eval using nested multiplication (Horner-like scheme).
	"""
	x = np.asarray(x_nodes, dtype=float)
	coef = np.asarray(coef, dtype=float)
	x_eval = np.asarray(x_eval, dtype=float)
	n = len(coef)
	if n == 0:
		return np.zeros_like(x_eval, dtype=float)

	y_eval = np.zeros_like(x_eval, dtype=float) + coef[n - 1]
	for k in range(n - 2, -1, -1):
		y_eval = y_eval * (x_eval - x[k]) + coef[k]
	return y_eval


# -----------------------------
# Interpolation: Neville's Algorithm (from scratch)
# -----------------------------
def neville_interpolate(x_nodes: List[float], y_nodes: List[float], x_eval: np.ndarray) -> np.ndarray:
	"""
	Evaluate interpolation at x_eval using Neville's algorithm.
	This is O(n^2) per x_eval; vectorized by looping over query points.
	"""
	x = np.asarray(x_nodes, dtype=float)
	y = np.asarray(y_nodes, dtype=float)
	x_eval = np.asarray(x_eval, dtype=float)
	n = len(x)
	if n != len(y):
		raise ValueError("x_nodes and y_nodes must be the same length.")

	def neville_single(xq: float) -> float:
		p = y.astype(float).copy()
		for m in range(1, n):
			for i in range(0, n - m):
				den = x[i + m] - x[i]
				if den == 0:
					raise ValueError("Duplicate x_nodes are not allowed.")
				p[i] = ((xq - x[i]) * p[i + 1] - (xq - x[i + m]) * p[i]) / den
		return p[0]

	return np.array([neville_single(xq) for xq in np.atleast_1d(x_eval)], dtype=float)


# -----------------------------
# Natural Cubic Spline (from scratch)
# -----------------------------
def natural_cubic_spline_coefficients(x_nodes: List[float], y_nodes: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Compute natural cubic spline piecewise coefficients for intervals [x_i, x_{i+1}].
	Returns arrays a, b, c, d for i in [0..n-2] such that:
	    S_i(x) = a[i] + b[i]*(x - x_i) + c[i]*(x - x_i)^2 + d[i]*(x - x_i)^3, x in [x_i, x_{i+1}]
	"""
	x = np.asarray(x_nodes, dtype=float)
	y = np.asarray(y_nodes, dtype=float)
	n = len(x)
	if n < 2 or n != len(y):
		raise ValueError("x_nodes and y_nodes must have same length and at least 2 points.")
	if np.any(np.diff(x) <= 0):
		raise ValueError("x_nodes must be strictly increasing.")

	h = np.diff(x)
	# Build tridiagonal system for c (second derivative scaled)
	alpha = np.zeros(n)
	for i in range(1, n - 1):
		alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1])

	l = np.ones(n)
	mu = np.zeros(n)
	z = np.zeros(n)
	# Natural spline boundary conditions: c0 = cn-1 = 0 achieved via this forward elimination
	for i in range(1, n - 1):
		l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
		if l[i] == 0:
			raise ValueError("Degenerate system encountered in spline construction.")
		mu[i] = h[i] / l[i]
		z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

	l[n - 1] = 1.0
	z[n - 1] = 0.0

	c = np.zeros(n)
	b = np.zeros(n - 1)
	d = np.zeros(n - 1)
	a = y[:-1].copy()

	# Back substitution
	for j in range(n - 2, -1, -1):
		c[j] = z[j] - mu[j] * c[j + 1]
		b[j] = ((y[j + 1] - y[j]) / h[j]) - (h[j] * (2.0 * c[j] + c[j + 1]) / 3.0)
		d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

	return a, b, c[:-1], d  # c returned for intervals (exclude the last point)


def evaluate_natural_cubic_spline(x_nodes: List[float], coeffs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], x_eval: np.ndarray) -> np.ndarray:
	"""
	Evaluate the natural cubic spline specified by coefficients at x_eval.
	"""
	x = np.asarray(x_nodes, dtype=float)
	a, b, c, d = coeffs
	x_eval = np.asarray(x_eval, dtype=float)
	n = len(x)

	def eval_single(xq: float) -> float:
		if xq <= x[0]:
			i = 0
		elif xq >= x[n - 1]:
			i = n - 2
		else:
			i = np.searchsorted(x, xq) - 1
			i = max(0, min(i, n - 2))
		dx = xq - x[i]
		return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx

	return np.array([eval_single(xq) for xq in np.atleast_1d(x_eval)], dtype=float)


# -----------------------------
# Least Squares Polynomial Fit (manual normal equations using numpy.linalg)
# -----------------------------
def least_squares_poly_fit(x_nodes: List[float], y_nodes: List[float], degree: int) -> np.ndarray:
	"""
	Fit a polynomial of given degree (<= len(x_nodes)-1 usually) via least squares using
	normal equations. Returns coefficients in descending powers (compatible with np.polyval).
	"""
	x = np.asarray(x_nodes, dtype=float)
	y = np.asarray(y_nodes, dtype=float)
	if degree < 0:
		raise ValueError("degree must be non-negative.")
	# Vandermonde with descending powers for compatibility with polyval
	V = np.vander(x, N=degree + 1, increasing=False)
	# Normal equations: (V^T V) c = V^T y
	normal_mat = V.T @ V
	normal_rhs = V.T @ y
	coeffs = np.linalg.solve(normal_mat, normal_rhs)
	return coeffs


# -----------------------------
# Utility printing
# -----------------------------
def print_header(title: str) -> None:
	print("\n" + "=" * 72)
	print(title)
	print("=" * 72)


def main() -> None:
	# Data
	depth_m = np.array([0.0, 50.0, 100.0, 150.0, 200.0], dtype=float)
	temp_c = np.array([25.0, 22.0, 20.0, 18.0, 16.0], dtype=float)
	fig_dir = ensure_figures_dir("figures")

	# Subset for "hand-written" style verification: three consecutive points
	x3 = np.array([50.0, 100.0, 150.0], dtype=float)
	y3 = np.array([22.0, 20.0, 18.0], dtype=float)
	test_points = np.array([50.0, 75.0, 100.0, 125.0, 150.0], dtype=float)

	print_header("Problem B8: Oceanographic Data Analysis - Interpolation and Fitting")
	print("Depth (m):", depth_m.tolist())
	print("Temperature (C):", temp_c.tolist())

	# -----------------------------
	# Verify on 3-point subset: Lagrange, Newton, Neville
	# -----------------------------
	print_header("Verification on 3-point subset: (50,22), (100,20), (150,18)")
	y_lagr_3 = lagrange_interpolate(x3, y3, test_points)
	coef_newton_3 = newton_divided_differences(x3, y3)
	y_newt_3 = newton_evaluate(x3, coef_newton_3, test_points)
	y_nev_3 = neville_interpolate(x3, y3, test_points)

	print("Test points:", test_points.tolist())
	print("Lagrange values:", np.round(y_lagr_3, 6).tolist())
	print("Newton values:", np.round(y_newt_3, 6).tolist())
	print("Neville values:", np.round(y_nev_3, 6).tolist())
	print("Max pairwise difference (consistency check):",
	      float(np.max(np.abs(y_lagr_3 - y_newt_3))))

	# Plot the subset interpolation
	xx_subset = np.linspace(50.0, 150.0, 201)
	yy_subset = lagrange_interpolate(x3, y3, xx_subset)
	plt.figure(figsize=(7, 4.5))
	plt.plot(x3, y3, "o", label="Data (subset)")
	plt.plot(xx_subset, yy_subset, "-", label="3-pt Interpolant (deg 2)")
	plt.title("3-Point Interpolation (Subset)")
	plt.xlabel("Depth (m)")
	plt.ylabel("Temperature (°C)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	subset_fig = os.path.join(fig_dir, "subset_3pt_interpolation.png")
	plt.savefig(subset_fig, bbox_inches="tight", dpi=150)
	plt.close()
	print(f"Saved subset plot to: {subset_fig}")

	# -----------------------------
	# Apply interpolation to full dataset
	# -----------------------------
	print_header("Interpolation on Full Dataset (5 points)")
	xx = np.linspace(depth_m[0], depth_m[-1], 401)

	# Lagrange on full set
	yy_lagr = lagrange_interpolate(depth_m, temp_c, xx)
	# Newton on full set
	coef_newton = newton_divided_differences(depth_m, temp_c)
	yy_newt = newton_evaluate(depth_m, coef_newton, xx)
	# Neville on full set (for a few sample points to reduce O(n^2)*N cost)
	sample_pts = np.array([25.0, 75.0, 125.0, 175.0], dtype=float)
	yy_nev_samples = neville_interpolate(depth_m, temp_c, sample_pts)

	print("Sample points for Neville (full set):", sample_pts.tolist())
	print("Neville sample values:", np.round(yy_nev_samples, 6).tolist())
	print("Max |Lagrange - Newton| over dense grid:",
	      float(np.max(np.abs(yy_lagr - yy_newt))))

	plt.figure(figsize=(7.5, 4.5))
	plt.plot(depth_m, temp_c, "ko", label="Data (5 pts)")
	plt.plot(xx, yy_lagr, "-", label="Lagrange (deg 4)")
	plt.plot(xx, yy_newt, "--", label="Newton (deg 4)")
	plt.title("Polynomial Interpolation on Full Dataset")
	plt.xlabel("Depth (m)")
	plt.ylabel("Temperature (°C)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	full_poly_fig = os.path.join(fig_dir, "full_polynomial_interpolation.png")
	plt.savefig(full_poly_fig, bbox_inches="tight", dpi=150)
	plt.close()
	print(f"Saved full polynomial interpolation plot to: {full_poly_fig}")

	# -----------------------------
	# Natural Cubic Spline
	# -----------------------------
	print_header("Natural Cubic Spline on Full Dataset")
	spline_coeffs = natural_cubic_spline_coefficients(depth_m, temp_c)
	yy_spline = evaluate_natural_cubic_spline(depth_m, spline_coeffs, xx)

	plt.figure(figsize=(7.5, 4.5))
	plt.plot(depth_m, temp_c, "ko", label="Data (5 pts)")
	plt.plot(xx, yy_spline, "-", label="Natural Cubic Spline")
	plt.title("Natural Cubic Spline Interpolation")
	plt.xlabel("Depth (m)")
	plt.ylabel("Temperature (°C)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	spline_fig = os.path.join(fig_dir, "spline_interpolation.png")
	plt.savefig(spline_fig, bbox_inches="tight", dpi=150)
	plt.close()
	print(f"Saved spline plot to: {spline_fig}")

	# -----------------------------
	# Least Squares Polynomial Fit (with noisy dataset comparison)
	# -----------------------------
	print_header("Least Squares Fit and Noisy Dataset Comparison")
	# Construct a slightly noisy dataset (reproducible)
	rng = np.random.default_rng(seed=42)
	noise = rng.normal(loc=0.0, scale=0.25, size=depth_m.shape)  # ~0.25 C noise
	temp_noisy = temp_c + noise
	print("Noisy temperatures (°C):", np.round(temp_noisy, 3).tolist())

	# Choose a modest degree for smoothing (e.g., quadratic)
	deg = 2
	coeffs_ls_clean = least_squares_poly_fit(depth_m, temp_c, degree=deg)
	coeffs_ls_noisy = least_squares_poly_fit(depth_m, temp_noisy, degree=deg)
	yy_ls_clean = np.polyval(coeffs_ls_clean, xx)
	yy_ls_noisy = np.polyval(coeffs_ls_noisy, xx)

	print(f"Least-squares degree {deg} coefficients (clean):", np.round(coeffs_ls_clean, 6).tolist())
	print(f"Least-squares degree {deg} coefficients (noisy):", np.round(coeffs_ls_noisy, 6).tolist())

	# Compare exact interpolation vs least squares (noisy)
	plt.figure(figsize=(8.0, 4.8))
	plt.plot(depth_m, temp_c, "ko", label="Original Data")
	plt.plot(depth_m, temp_noisy, "s", color="#cc4444", label="Noisy Data")
	plt.plot(xx, yy_lagr, "-", label="Exact Interpolant (deg 4)")
	plt.plot(xx, yy_spline, "--", label="Cubic Spline")
	plt.plot(xx, yy_ls_noisy, "-", color="#1f77b4", label=f"Least Squares (deg {deg}, noisy)")
	plt.title("Exact Interpolation vs Least-Squares Smoothing (Noisy Data)")
	plt.xlabel("Depth (m)")
	plt.ylabel("Temperature (°C)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	compare_fig = os.path.join(fig_dir, "interpolation_vs_least_squares_noisy.png")
	plt.savefig(compare_fig, bbox_inches="tight", dpi=150)
	plt.close()
	print(f"Saved comparison plot to: {compare_fig}")

	# -----------------------------
	# Small numerical report at select depths
	# -----------------------------
	print_header("Numerical Results at Selected Depths")
	report_depths = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0])
	report = {
		"Lagrange_deg4": lagrange_interpolate(depth_m, temp_c, report_depths),
		"Newton_deg4": newton_evaluate(depth_m, coef_newton, report_depths),
		"Spline": evaluate_natural_cubic_spline(depth_m, spline_coeffs, report_depths),
		"LS_quadratic_clean": np.polyval(coeffs_ls_clean, report_depths),
		"LS_quadratic_noisy": np.polyval(coeffs_ls_noisy, report_depths),
	}
	for name, vals in report.items():
		print(f"{name}: {np.round(vals, 4).tolist()}")

	print_header("Done")
	print(f"Figures saved in: {fig_dir}")


if __name__ == "__main__":
	main()

