from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from fft import fft_poly_mul
from sum_and_mul import coef_mul

max_degree = 1000
y1 = []
y2 = []

for i in range(1, max_degree, 10):
    pol1 = np.ones(i)
    pol2 = np.ones(i)

    start = timer()
    coef_mul(pol1, pol2)
    end = timer()
    y1.append(end - start)

    pol1 = np.ones(i)
    pol2 = np.ones(i)
    start = timer()
    fft_poly_mul(pol1, pol2)
    end = timer()
    y2.append(end - start)

plt.plot(y1, "r")
plt.plot(y2, "b")
plt.show()
