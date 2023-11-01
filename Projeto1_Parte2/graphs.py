import numpy as np
import matplotlib.pyplot as plt

a = np.arange(0,11)
b = np.arange(10,-1, -1)
c = np.arange(0,11)
plt.plot(a, b, c)
plt.scatter([4.8], [5.5], c = 'red')
plt.scatter([9.5], [7.5], c = 'green')
plt.show()