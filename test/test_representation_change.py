import unittest

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

        # complex coefficient polynomial
        pol4 = np.array([1 + 1j, 1j])
        self.assertEqual(evaluation(pol4, 2), 2 + 3j)


if __name__ == '__main__':
    unittest.main()
