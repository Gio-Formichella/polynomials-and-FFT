from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from baseline import coef_mul
from fft import fft_poly_mul

max_degree = 2000
repetitions = 5
step = 100

y1 = []  # baseline scores
y2 = []  # iterative fft scores
y3 = []  # recursive fft scores

np.random.seed(0)

for i in range(1, max_degree, step):
    # time sum vars
    s1 = 0
    s2 = 0
    s3 = 0
    for _ in range(repetitions):
        pol1 = np.random.rand(i)
        pol2 = np.random.rand(i)

        start = timer()
        coef_mul(pol1, pol2)
        end = timer()
        s1 += end - start

        # testing iterative fft
        start = timer()
        fft_poly_mul(pol1, pol2, "iterative")
        end = timer()
        s2 += end - start

        # testing recursive fft
        start = timer()
        fft_poly_mul(pol1, pol2, "recursive")
        end = timer()
        s3 += end - start

    y1.append(s1)
    y2.append(s2)
    y3.append(s3)

plt.plot(range(1, max_degree, step), y1, "r", label="Coefficient Multiplication")
plt.plot(range(1, max_degree, step), y2, "b", label="Iterative FFT Multiplication")
plt.plot(range(1, max_degree, step), y3, "g", label="Recursive FFT Multiplication")
plt.xlabel("Polynomial Degree")
plt.ylabel("Time (s)")
plt.legend()
plt.show()
