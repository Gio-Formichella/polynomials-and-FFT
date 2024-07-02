import numpy as np


def evaluation(pol: np.array, x):
    """
    Using Horner's rule for polynomial evaluation
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
