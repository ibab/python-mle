from numpy.random import normal
from numpy import linspace, array, append
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import clock
from pandas import DataFrame

from mle import Normal, var, par, Mix2

x = var("x")
mu1 = par("mu1")
sigma1 = par("sigma1")

model = Normal(x, mu1, sigma1)

N = int(1e6)

data = array(normal(3, 5, N), dtype=[("x", float)])

start = clock()
result = model.fit(data, {"mu1":2, "sigma1":4}, method="BFGS")
print("Fit took {:5.1f} s".format(clock()-start))
print(result)

from mle.tests import kolsmi, chisquare

print("Kol-Smir-Test: {:1.5f}, p-value {:1.2f}".format(*kolsmi(model, result, data)))
print("Chi^2/ndf: {:2.2f}, p-value {:1.2f}".format(*chisquare(model, result, data)))


px = linspace(min(data["x"]), max(data["x"]), 1000)

plt.plot(px, model.pdf(px, **result["x"]))
plt.hist(data["x"], 50, histtype="step", normed=True)
plt.show()
