import matplotlib.pyplot as plt
import numpy as np
from scipy import special

t = np.linspace(1, 3, 1000)
markov_1 = np.sqrt(2 / np.pi) / (t ** 1)
markov_2 = 1 / (t ** 2)
markov_3 = 2 * np.sqrt(2 / np.pi) / (t ** 3)
markov_4 = 3 / (t ** 4)
markov_5 = 4 * np.sqrt(2 / np.pi) / (t ** 5)
mills = np.sqrt(2 / np.pi) * np.exp(-(t ** 2) / 2) / t
true_bound = special.erfc(t / np.sqrt(2))

plt.figure(figsize=(12, 8))
plt.plot(t, markov_1, label="markov, k=1")
plt.plot(t, markov_2, label="markov, k=2")
plt.plot(t, markov_3, label="markov, k=3")
plt.plot(t, markov_4, label="markov, k=4")
plt.plot(t, markov_5, label="markov, k=5")
plt.plot(t, mills, label="mill")
plt.plot(t, true_bound, color="black", linestyle="--", label="true bound")
plt.ylabel("Bound")
plt.xlabel("t")
plt.legend()
plt.savefig("04-06.png", bbox_inches="tight")
