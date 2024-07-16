import math

import numpy as np


def recursive_fft(a: np.array, inverse=False, root=True) -> np.array:
    """
    Fast Fourier Transformation of polynomial, can be used to calculate both the direct and inverse DFT.
    Complexity: O(n lg(n))

    Note: length of a must be a power of 2, to solve constraint add 0 in front of a
    :param a: polynomial
    :param inverse: direct or inverse DFT flag
    :param root: used to check for root function in recursive calls
    :return: DFT(a) in point form
    """
    n = len(a)

    if n == 1:
        return a

    half_size = n // 2

    omega_n = np.exp(2 * np.pi * 1j / n)  # twiddle factor
    if inverse:
        omega_n **= -1

    omega = 1

    even_a = a[0::2]
    odd_a = a[1::2]
    even_y = recursive_fft(even_a, inverse, False)
    odd_y = recursive_fft(odd_a, inverse, False)

    y = np.empty(n, dtype=complex)
    for k in range(half_size):
        y[k] = even_y[k] + omega * odd_y[k]
        y[k + half_size] = even_y[k] - omega * odd_y[k]
        omega = omega * omega_n

    if inverse and root:
        y /= n
    return y


def fft_poly_mul(pol1: np.array, pol2: np.array) -> np.array:
    """
    Complexity: O(n lg(n))
    :param pol1: coefficient form polynomial
    :param pol2: coefficient form polynomial
    :return: multiplication polynomial in coefficient form
    """
    l1 = len(pol1)
    l2 = len(pol2)

    # Null polynomial
    if l1 == 0:
        return np.zeros(l2, dtype=pol1.dtype)
    if l2 == 0:
        return np.zeros(l1, dtype=pol2.dtype)

    max_length = l1 + l2 - 1
    m = 2 ** (math.ceil(math.log2(max_length)))

    pol1 = np.pad(pol1, (0, m - l1))
    pol2 = np.pad(pol2, (0, m - l2))

    y1 = recursive_fft(pol1)
    y2 = recursive_fft(pol2)

    y = y1 * y2

    # Inverse DFT
    a = recursive_fft(y, inverse=True)

    # Trimming the result to remove padding
    return a[:max_length]
