import concurrent.futures
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


def iterative_fft(a: np.array, inverse=False) -> np.array:
    """
    Iterative Fast Fourier Transformation of polynomial. Runs butterflies in parallel

    Complexity: O(n lg(n))
    :param a: polynomial in coefficient form
    :param inverse: direct or inverse DFT flag
    :return: DFT(a) in point form
    """
    n = len(a)
    a = bit_reverse_copy(a)

    for s in range(1, int(np.log2(n)) + 1):
        m = 2 ** s
        omega_m = np.exp(2 * np.pi * 1j / m)
        if inverse:
            omega_m **= -1

        def butterfly(k):
            omega = 1
            for j in range(m // 2):
                t = omega * a[k + j + m // 2]
                u = a[k + j]
                a[k + j] = u + t
                a[k + j + m // 2] = u - t
                omega = omega * omega_m

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(butterfly, range(0, n, m))

    if inverse:
        a = a / n
    return a


def bit_reverse_copy(a: np.array) -> np.array:
    """

    :param a: polynomial
    :return: reordered array based on index bits reverse
    """
    n = len(a)
    bit_size = int(np.log2(n))
    reversed_a = np.zeros(n, dtype=a.dtype)

    for i in range(n):
        reversed_index = reverse_bits(i, bit_size)
        reversed_a[reversed_index] = a[i]

    return reversed_a


def reverse_bits(n, bit_size):
    """

    :param n: integer
    :param bit_size:  bits used for integer representation
    :return: reversed bit integer
    """
    reverse_n = 0
    for i in range(bit_size):
        reverse_n = (reverse_n << 1) | (n & 1)
        n >>= 1

    return reverse_n
