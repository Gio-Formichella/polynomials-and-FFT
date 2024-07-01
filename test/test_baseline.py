import unittest

import numpy.testing as npt

from baseline import *


class MyTestCase(unittest.TestCase):
    def test_baseline_sum(self):
        # testing different size polynomials
        pol1 = np.array([1, 2, 3, 4])
        pol2 = np.array([5, 6])
        r1 = np.array([1, 2, 8, 10])

        npt.assert_array_equal(r1, baseline_sum(pol1, pol2))

        # testing null polynomial
        pol3 = np.array([])
        npt.assert_array_equal(pol1, baseline_sum(pol1, pol3))

        # testing complex numbers
        pol4 = np.array([1, 3 + 5j])
        pol5 = np.array([1 + 1j, 1])
        r3 = np.array([2 + 1j, 4 + 5j])
        npt.assert_allclose(r3, baseline_sum(pol4, pol5))


if __name__ == '__main__':
    unittest.main()
