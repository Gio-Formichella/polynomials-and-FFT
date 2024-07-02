import numpy as np
from numpy.linalg import inv


# Evaluating the polynomial for n distinct points achieves the point representation of the same polynomial.
def evaluation(pol: np.array, x):
    """
    Using Horner's rule for polynomial evaluation. Complexity: O(n)
    :param pol: polynomial in coefficient form as numpy array
    :param x: point of evaluation
    :return: evaluation of pol at x
    """
    size = pol.size
    if size == 0:
        return 0
    if size == 1:
        return pol[0]

    s = x * pol[0]
    i = 1
    while i < size - 1:
        s += pol[i]
        s *= x
        i += 1

    s += pol[i]
    return s


# Note: a O(n^2) algorithm exists that uses Lagrange's formula, I wasn't able to find a way to simply implement it
def interpolation(pol: list) -> np.array:
    """
    Complexity: O(n^3)
    :param pol: point representation of the polynomial
    :return: coefficient representation of the same polynomial
    """
    if not pol:
        raise ValueError("No points given")

    x = [point[0] for point in pol]
    y = [point[1] for point in pol]
    y = np.array(y).reshape(-1, 1)  # column vector
    v = np.vander(x)

    a = np.dot(inv(v), y).reshape(1, -1)
    return a
