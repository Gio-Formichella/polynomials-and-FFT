import unittest
from fft import *
import numpy.testing as npt


class MyTestCase(unittest.TestCase):
    def test_recursive_fft(self):
        pol1 = np.array([1, 1, 1, 1])  # n = 4 = 2^2
        r = np.array([4, 0, 0, 0])
        npt.assert_almost_equal(recursive_fft(pol1), r)


if __name__ == '__main__':
    unittest.main()
