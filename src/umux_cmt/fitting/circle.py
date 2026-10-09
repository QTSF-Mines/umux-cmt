"""Algebraic circle fit of complex ``S21`` data.

In the complex plane a notch resonance traces a circle (Probst et al. 2015,
Khalil et al. 2012). The circle is fitted first, without any resonator model,
to find its centre and radius; those seed the phase fit and the full notch fit
in :mod:`umux_cmt.fitting.notch`.

The fit minimises the algebraic distance of the points to the circle
``A (x**2 + y**2) + B x + C y + D = 0`` under the constraint
``B**2 + C**2 - 4 A D = 1`` (Chernov & Lesort 2005), which turns the problem
into a 4x4 generalised eigenvalue problem; the eigenvector of the smallest
non-negative eigenvalue is the circle. The data are scaled to unit mean radius
before the eigenproblem so that the matrices are well conditioned for any
transmission level. For points that lie exactly on a circle (simulated,
noise-free data) the eigenvalue of the fit is zero up to round-off and may
come out slightly negative; such values are accepted within a relative
tolerance, which changes nothing for measured data, where that eigenvalue is
positive.
"""
import numpy as np
from scipy.linalg import eig


def fit_circle(x_s, y_s):
    """Fit a circle to the points ``(x_s, y_s)``.

    Parameters
    ----------
    x_s, y_s : ndarray
        Real and imaginary parts of the data, one point per element.

    Returns
    -------
    (xc, yc, r) : tuple of float
        Centre and radius of the fitted circle.

    Raises
    ------
    ValueError
        If the generalised eigenproblem has no non-negative eigenvalue, which
        happens for degenerate (collinear or single-point) data.
    """
    scale = np.mean(np.sqrt(x_s**2 + y_s**2))

    if scale < 1e-9:
        scale = 1.0
    x = x_s / scale
    y = y_s / scale

    n = len(x)
    w = x**2 + y**2

    Mw = np.sum(w)
    Mx = np.sum(x)
    My = np.sum(y)
    Mww = np.sum(w**2)
    Mxw = np.sum(x * w)
    Myw = np.sum(y * w)
    Mxx = np.sum(x**2)
    Myy = np.sum(y**2)
    Mxy = np.sum(x * y)

    M = np.array([
        [Mww, Mxw, Myw, Mw],
        [Mxw, Mxx, Mxy, Mx],
        [Myw, Mxy, Myy, My],
        [Mw,  Mx,  My,  n]
    ])

    B = np.array([
        [0, 0, 0, -2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-2, 0, 0, 0]
    ])

    eigenvalues, eigenvectors = eig(M, B)

    n_min = -1
    min_eig_val = np.inf
    zero = -1e-9 * np.max(np.abs(eigenvalues.real))      # round-off of an exactly-zero eigenvalue

    for i, val in enumerate(eigenvalues):
        if val.real > zero and val.real < min_eig_val:
            min_eig_val = val.real
            n_min = i

    if n_min == -1:
        raise ValueError("Circle fit failed. No positive eigenvalue found.")

    params = eigenvectors[:, n_min].real
    A, B_param, C_param, D_param = params

    xc = -B_param / (2 * A)
    yc = -C_param / (2 * A)

    r = np.sqrt(B_param**2 + C_param**2 - 4 * A * D_param) / (2 * np.abs(A))

    xc_s = xc * scale
    yc_s = yc * scale
    r_s = r * scale

    return xc_s, yc_s, r_s


def circle_points(xc, yc, r, n=100):
    """Points on the circle ``(xc, yc, r)`` for plotting.

    Parameters
    ----------
    xc, yc : float
        Centre of the circle.
    r : float
        Radius.
    n : int, optional
        Number of points; the first and the last coincide.

    Returns
    -------
    (xs, ys) : tuple of ndarray
        ``n`` points around the full circle.
    """
    theta = np.linspace(0, 2 * np.pi, n)
    xs = xc + r * np.cos(theta)
    ys = yc + r * np.sin(theta)
    return xs, ys


__all__ = ["fit_circle", "circle_points"]
