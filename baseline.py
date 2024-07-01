import numpy as np


def baseline_sum(pol1: np.array, pol2: np.array) -> np.array:
    """
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
