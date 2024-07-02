import unittest
import numpy.testing as npt
from representation_change import *


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
        r2 = np.array([(-4 - 3j)/5, (4 + 8j)/5]).reshape(1, 2)
        npt.assert_almost_equal(interpolation(pol2), r2)


if __name__ == '__main__':
    unittest.main()
