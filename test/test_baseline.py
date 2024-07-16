import unittest

import numpy.testing as npt

from baseline import *


class MyTestCase(unittest.TestCase):
    def test_evaluation(self):
        # testing edge cases
        pol1 = np.array([])
        self.assertEqual(evaluation(pol1, 1), 0)
        pol2 = np.array([4])
        self.assertEqual(evaluation(pol2, 1), 4)

        # real coefficient polynomial
        pol3 = np.array([1, 1, 1])
        self.assertEqual(evaluation(pol3, 0), 1)
        self.assertEqual(evaluation(pol3, 1), 3)
        self.assertEqual(evaluation(pol3, 1j), 1j)

        # complex coefficient polynomial
        pol4 = np.array([1 + 1j, 1j])
        self.assertEqual(evaluation(pol4, 2), 2 + 3j)

    def test_interpolation(self):
        # testing edge cases
        self.assertRaises(ValueError, interpolation, [])

        # real points
        pol1 = [(1, 3), (2, 7), (3, 13)]
        r1 = np.array([1.0, 1.0, 1.0]).reshape(1, 3)
        npt.assert_almost_equal(interpolation(pol1), r1)

        # complex points
        pol2 = [(1, 1j), (2j, 2)]
        r2 = np.array([(-4 - 3j) / 5, (4 + 8j) / 5]).reshape(1, 2)
        npt.assert_almost_equal(interpolation(pol2), r2)

    def test_coef_sum(self):
        # testing different degree polynomials
        pol1 = np.array([1, 2, 3, 4])
        pol2 = np.array([5, 6])
        r1 = np.array([1, 2, 8, 10])

        npt.assert_array_equal(r1, coef_sum(pol1, pol2))

        # testing null polynomial
        pol3 = np.array([])
        npt.assert_array_equal(pol1, coef_sum(pol1, pol3))

        # testing complex numbers
        pol4 = np.array([1, 3 + 5j])
        pol5 = np.array([1 + 1j, 1])
        r3 = np.array([2 + 1j, 4 + 5j])
        npt.assert_allclose(r3, coef_sum(pol4, pol5))

    def test_coef_mul(self):
        # testing different degree polynomials
        pol1 = np.array([4, 3, 2, 1])
        pol2 = np.array([1, 4, 1])
        r1 = np.array([4, 19, 18, 12, 6, 1])

        npt.assert_array_equal(r1, coef_mul(pol1, pol2))

        # testing null polynomial
        pol3 = np.array([])
        r2 = np.zeros(pol1.size - 1)

        npt.assert_array_equal(r2, coef_mul(pol1, pol3))

        # testing complex numbers
        pol4 = np.array([1 + 1j, 4])
        pol5 = np.array([1, 2 + 2j])
        r3 = np.array([1 + 1j, 4 + 4j, 8 + 8j])
        npt.assert_allclose(r3, coef_mul(pol4, pol5))

    def test_point_sum(self):
        # invalid input
        self.assertRaises(ValueError, point_sum, [(0, 0)], [])
        self.assertRaises(ValueError, point_sum, [(0, 0)], [1, 1])

        # sum
        pol1 = [(0, 0), (1, 1), (2, 2)]
        pol2 = [(0, 1), (1, 1j), (2, 4)]
        r = [(0, 1), (1, 1 + 1j), (2, 6)]
        self.assertEqual(point_sum(pol1, pol2), r)

    def test_point_mul(self):
        # invalid input
        self.assertRaises(ValueError, point_mul, [(0, 0)], [])
        self.assertRaises(ValueError, point_mul, [(0, 0)], [1, 1])

        # mul
        pol1 = [(0, 1), (1, 2), (2, 3), (3, 4)]
        pol2 = [(0, 2), (1, 1j), (2, 1 + 2j), (3, 0)]
        r = [(0, 2), (1, 2j), (2, 3 + 6j), (3, 0)]
        self.assertEqual(point_mul(pol1, pol2), r)


if __name__ == '__main__':
    unittest.main()
