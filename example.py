
import numpy as np
from mle import *
from time import clock
import operator

def prod(factors):
    return reduce(Join, factors[1:], factors[0])

varnames = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6']

models = []
for name in varnames:
    v = var(name, observed=True, vector=True)
    mu1 = var('mu1_{}'.format(name))
    sigma1 = var('sigma1_{}'.format(name))
    mu2 = var('mu2_{}'.format(name))
    sigma2 = var('sigma2_{}'.format(name))
    theta = var('theta_{}'.format(name))
    models.append(Mix2(theta, Normal(v, mu1, sigma1), Normal(v, mu2, sigma2)))

model = prod(models)

# Generate data
data = dict()
inits = dict()
for name in varnames:
    N = 5e6
    N1 = np.random.binomial(N, 0.4)
    data[name] =  np.append(np.random.normal(-1, 1, N1), np.random.normal(1, 1, N - N1))
    inits['theta_{}'.format(name)] = 0.5
    inits['mu1_{}'.format(name)] = -3
    inits['mu2_{}'.format(name)] = 2
    inits['sigma1_{}'.format(name)] = 2
    inits['sigma2_{}'.format(name)] = 1

result = model.fit(data, inits)

for k, v in result['x'].items():
    print k, v

print(result)

