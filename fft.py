import numpy as np


def recursive_fft(a: np.array):
    """
    Fast Fourier Transformation of polynomial

    Note: length of a must be a power of 2, 0 in front can be added to solve constraint
    :param a: polynomial
    :return: DFT(a)
    """
    n = len(a)

    if n == 1:
        return a

    # n must be a power of 2
    half_size = int(n / 2)

    omega_n = np.exp(2 * np.pi * 1j / n)
    omega = 1

    even_a = a[1::2]
    odd_a = a[::2]
    even_y = recursive_fft(even_a)
    odd_y = recursive_fft(odd_a)

    y = np.empty(n)
    for k in range(half_size):
        y[k] = even_y[k] + odd_y[k]
        y[k + half_size] = even_y[k] - omega * odd_y[k]
        omega = omega * omega_n

    return y
