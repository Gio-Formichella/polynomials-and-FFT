import numpy as np


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
                mul_poly[-j] += pol1[-k]*pol2[- j + k - 1]

    return mul_poly


def point_sum(pol1: list, pol2: list) -> list:
    """
    Note: The two polynomial in point representation must use the same points, an exception is run otherwise

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
