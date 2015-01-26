
import theano.tensor as T
from numpy import inf
from math import pi

def alltrue(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret

def bound(logp, *conditions):
    return T.switch(alltrue(conditions), logp, -inf)

class Distribution:
    def __init__(self):
        pass

class Uniform(Distribution):
    def __init__(self, lower=0, upper=1, *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)
        self.lower = lower
        self.upper = upper

    def logp(self, value):
        upper = self.upper
        lower = self.lower
        return T.log(1 / (upper - lower))

class Normal(Distribution):
    def __init__(self, mu=0, sigma=1, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.mu = mu
        self.sigma = sigma

    def logp(self, value):
        mu = self.mu
        sigma = self.sigma
        ret = -1 / sigma**2 * (value - mu)**2 + T.log(1 / (sigma**2 * 2 * pi)) / 2.
        return bound(ret, sigma > 0)

class Mix2(Distribution):
    def __init__(self, frac, dist1, dist2, *args, **kwargs):
        super(Mix2, self).__init__(*args, **kwargs)
        self.frac = frac
        self.dist1 = dist1
        self.dist2 = dist2

    def logp(self, value):
        frac = self.frac
        dist1 = self.dist1
        dist2 = self.dist2
        return T.log(frac * T.exp(dist1.logp(value)) + (1 - frac) * T.exp(dist2.logp(value)))


