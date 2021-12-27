import matplotlib.pyplot as plt
import numpy as np

n = np.arange(1, 101)
y = np.repeat(1 / 2, 100)
plt.plot(n, y)
plt.xlabel("n")
plt.ylabel("Expected sample mean")
plt.show()
