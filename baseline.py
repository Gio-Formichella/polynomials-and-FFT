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


def coef_sum(pol1: np.array, pol2: np.array) -> np.array:
    """
    Complexity: O(n)
    :param pol1: polynomial in coefficient form as numpy array
    :param pol2: polynomial in coefficient form as numpy array
    :return: sum polynomial in coefficient form as numpy array
    """

    i = 0
    j = 0
    poly_sum = []

    while i < pol1.size and j < pol2.size:
        poly_sum.insert(0, pol1[-i - 1] + pol2[-j - 1])
        i += 1
        j += 1

    while i < pol1.size:
        poly_sum.insert(0, pol1[-i - 1])
        i += 1

    while j < pol2.size:
        poly_sum.insert(0, pol2[-j - 1])
        j += 1

    return np.array(poly_sum)


def coef_mul(pol1: np.array, pol2: np.array) -> np.array:
    """
    Also called convolution. Complexity: O(n^2)
    :param pol1: polynomial in coefficient form as numpy array
    :param pol2: polynomial in coefficient form as numpy array
    :return: sum polynomial in coefficient form as numpy array
    """

    mul_poly_size = pol1.size + pol2.size - 1
    mul_poly = np.zeros(mul_poly_size, dtype=complex)

    for j in range(1, mul_poly_size + 1):
        for k in range(1, j + 1):
            if k <= pol1.size and j - k + 1 <= pol2.size:
                mul_poly[-j] += pol1[-k] * pol2[- j + k - 1]

    return mul_poly


def point_sum(pol1: list, pol2: list) -> list:
    """
    Note: The two polynomial in point representation must use the same points, an exception is raised otherwise

    Complexity: O(n)
    :param pol1: polynomial in point form
    :param pol2: polynomial in point form
    :return: sum polynomial in point form
    """

    if len(pol1) != len(pol2):
        raise ValueError("Different number of points given")

    sum_poly = []  # Stores result
    for i in range(len(pol1)):
        if pol1[i][0] != pol2[i][0]:
            raise ValueError("Different points given")
        sum_poly.append((pol1[i][0], pol1[i][1] + pol2[i][1]))

    return sum_poly


def point_mul(pol1: list, pol2: list) -> list:
    """
    Note: The two polynomial in point representation must use the same points, an exception is raised otherwise.
    The number of points must be double the highest degree.

    Complexity: O(n)
    :param pol1: polynomial in extensive point form (double the points)
    :param pol2: polynomial in extensive point form (double the points)
    :return: mul polynomial in point form
    """

    if len(pol1) != len(pol2):
        raise ValueError("Different number of points given")

    mul_poly = []  # Stores result
    for i in range(len(pol1)):
        if pol1[i][0] != pol2[i][0]:
            raise ValueError("Different points given")
        mul_poly.append((pol1[i][0], pol1[i][1] * pol2[i][1]))

    return mul_poly
