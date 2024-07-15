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
