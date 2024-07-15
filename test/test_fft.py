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


if __name__ == '__main__':
    unittest.main()
