import scipy.optimize as opt
import scipy.stats as stats

f = lambda x: 0.95 - stats.norm.cdf((x - 3) / 4) + stats.norm.cdf((-x - 3) / 4)
print(opt.fsolve(f, 9))
