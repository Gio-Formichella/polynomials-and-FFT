import unittest

import numpy.testing as npt

from fft import *


class MyTestCase(unittest.TestCase):
    def test_recursive_fft(self):
        # direct fft
        pol1 = np.array([1, 1, 1, 1])  # n = 4 = 2^2
        r = np.array([4, 0, 0, 0])
        npt.assert_almost_equal(recursive_fft(pol1), r)

        # inverse fft
        points = np.array([4, 0, 0, 0])
        r = np.array([1, 1, 1, 1])
        npt.assert_almost_equal(recursive_fft(points, True), r)

        #
        pol2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        points = recursive_fft(pol2)
        npt.assert_almost_equal(pol2, recursive_fft(points, True))

    def test_fft_poly_mul(self):
        # ITERATIVE FFT USAGE

        # testing different degree polynomials
        pol1 = np.array([4, 3, 2, 1])
        pol2 = np.array([1, 4, 1])
        r1 = np.array([4.0, 19.0, 18.0, 12.0, 6.0, 1.0])

        npt.assert_allclose(r1, fft_poly_mul(pol1, pol2))

        # testing null polynomial
        pol3 = np.array([])
        r2 = np.zeros(pol1.size)

        npt.assert_array_equal(r2, fft_poly_mul(pol1, pol3))

        # testing complex numbers
        pol4 = np.array([1 + 1j, 4])
        pol5 = np.array([1, 2 + 2j])
        r3 = np.array([1 + 1j, 4 + 4j, 8 + 8j])
        npt.assert_allclose(r3, fft_poly_mul(pol4, pol5))

        # RECURSIVE FFT USAGE

        # testing different degree polynomials
        pol1 = np.array([4, 3, 2, 1])
        pol2 = np.array([1, 4, 1])
        r1 = np.array([4.0, 19.0, 18.0, 12.0, 6.0, 1.0])

        npt.assert_allclose(r1, fft_poly_mul(pol1, pol2, "recursive"))

        # testing null polynomial
        pol3 = np.array([])
        r2 = np.zeros(pol1.size)

        npt.assert_array_equal(r2, fft_poly_mul(pol1, pol3, "recursive"))

        # testing complex numbers
        pol4 = np.array([1 + 1j, 4])
        pol5 = np.array([1, 2 + 2j])
        r3 = np.array([1 + 1j, 4 + 4j, 8 + 8j])
        npt.assert_allclose(r3, fft_poly_mul(pol4, pol5, "recursive"))

    def test_iterative_fft(self):
        # direct fft
        pol1 = np.array([1, 1, 1, 1])  # n = 4 = 2^2
        r = np.array([4, 0, 0, 0])
        npt.assert_almost_equal(iterative_fft(pol1), r)

        # inverse fft
        points = np.array([4, 0, 0, 0])
        r = np.array([1, 1, 1, 1])
        npt.assert_almost_equal(iterative_fft(points, True), r)

        pol2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        points = recursive_fft(pol2)
        npt.assert_almost_equal(pol2, iterative_fft(points, True))


if __name__ == '__main__':
    unittest.main()
