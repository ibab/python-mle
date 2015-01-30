from numpy.random import normal
from numpy import linspace, float32, append
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import clock

from mle import Normal, var, par, Mix2

x = var("x")
mu1 = par("mu1")
sigma1 = par("sigma1")

model = Normal(x, mu1, sigma1)

data = normal(3, 5, int(1e6))
data = data.astype(float32)

start = clock()
result = model.fit({"x":data}, {"mu1":2, "sigma1":4})
print("Fit took {:5.1f}".format(clock()-start))

print(result)

from mle.tests import kolsmi

print("P-Value of Kolmogorow-Smirnow-Test", kolsmi(model, result, data, 0.05)[0])

mu = result["x"]["mu1"]
sigma = result["x"]["sigma1"]

px = linspace(-22, 28, 1000).astype(float32)

plt.plot(px, norm.pdf(px, mu, sigma))
plt.hist(data, 50, histtype="step", normed=True)
plt.show()
