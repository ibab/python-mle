
import numpy as np
import theano.tensor as T
from numpy import inf, array, ndarray

from .distribution import Model

__all__=['Uniform', 'Normal', 'Mix2']

def alltrue(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret

def bound(logp, *conditions):
    return T.switch(alltrue(conditions), logp, -np.inf)

class Uniform(Model):
    def __init__(self, x, lower, upper, *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)
        self._add_expr('x', x)
        self._add_expr('lower', lower)
        self._add_expr('upper', upper)

        self._logp = T.log(T.switch(T.gt(x, upper), 0, T.switch(T.lt(x, lower), 0, 1/(upper - lower))))
        self._cdf = T.switch(T.gt(x, up), 1, T.switch(T.lt(x, low), 0, (x - low)/(up - low)))

    def sample(self, num, init):
        lower = init[self.lower.name]
        upper = init[self.upper.name]
        return np.random.uniform(lower, upper, num)

class Normal(Model):
    def __init__(self, x, mu, sigma, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self._add_expr('x', x)
        self._add_expr('mu', mu)
        self._add_expr('sigma', sigma)

        self._logp = bound(-(x - mu)**2 / (2 * sigma**2) + T.log(1 / T.sqrt(sigma**2 * 2 * pi)), sigma > 0)
        self._cdf = 0.5 * (1 + T.erf((self.x - self.mu)/(self.sigma*T.sqrt(2))))

    def sample(self, num, init):
        mu = init[self.mu.name]
        sigma = init[self.sigma.name]
        return np.random.normal(mu, sigma, num)

class Mix2(Model):
    def __init__(self, frac, dist1, dist2, *args, **kwargs):
        super(Mix2, self).__init__(*args, **kwargs)
        self._add_expr('frac', frac)
        self._add_submodel('dist1', dist1)
        self._add_submodel('dist2', dist2)
        self._cdf = lambda: self.frac * self.dist1.tcdf() + (1-self.frac) * self.dist2.tcdf()
        self._logp = bound(T.log(frac * T.exp(dist1.logp()) + (1 - frac) * T.exp(dist2.logp())), frac > 0, frac < 1)

    def sample(self, num, init):
        frac = init[self.frac.name]
        N1 = np.random.binomial(num, frac)
        N2 = num - N1
        ret = np.append(self.dist1.sample(N1, init), self.dist2.sample(N2, init))
        np.random.shuffle(ret)
        return ret

