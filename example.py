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
mu2 = par("mu2")
sigma2 = par("sigma2")
frac = par("frac")

model = Mix2(frac, Normal(x, mu1, sigma1), Normal(x, mu2, sigma2))

N = int(1e5)

data = array(append(normal(3, 1, N), normal(-3, 1, 2*N)), dtype=[("x", float)])

initial = {
    "frac": 0.5,
    "mu1":2,
    "sigma1":4,
    "mu2":2,
    "sigma2":4
}

start = clock()
result = model.fit(data, initial, method="Powell")
print("Fit took {:5.1f} s".format(clock()-start))
print(result)

from mle.tests import kolsmi, chisquare

print("Kol-Smir-Test: {:1.5f}, p-value {:1.2f}".format(*kolsmi(model, result, data)))
print("Chi^2/ndf: {:2.2f}, p-value {:1.2f}".format(*chisquare(model, result, data)))


px = linspace(min(data["x"]), max(data["x"]), 1000)

plt.plot(px, model.pdf(px, **result["x"]))
plt.hist(data["x"], 50, histtype="step", normed=True)
plt.show()
