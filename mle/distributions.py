from scipy.optimize import minimize
from theano import function, scan
import theano.tensor as T
from numpy import inf, array, ndarray
from numpy.core.records import recarray
import numpy as np
from math import pi
from collections import OrderedDict as OD
import logging

from .util import hessian_
from .memoize import memoize

__all__ = ['var', 'par', 'Normal', 'Uniform', 'Exponential', 'Power', 'Mix2']

def alltrue(vals):
    ret = 1
    for c in vals:
        ret = ret * (1 * c)
    return ret

def bound(logp, *conditions):
    return T.switch(alltrue(conditions), logp, -inf)

def var(name, label=None):
    return Variable(name, label)

def par(name, label=None, lower=None, upper=None):
    return Parameter(name, label, lower, upper)

class Variable(object):
    def __init__(self, name, label=None):
        self.name = name
        if label:
            self.label = label
        else:
            self.label = name
        self.tvar = T.vector(name)

    def __repr__(self):
        return self.label

    def __rmul__(self, other):
        other * self.tvar

    def __mul__(self, other):
        self.tvar * other

    def __rdiv__(self, other):
        other / self.tvar

    def __div__(self, other):
        self.tvar / other

    def __radd__(self, other):
        other + self.tvar

    def __add__(self, other):
        self.tvar + other

    def __rsub__(self, other):
        other - self.tvar

    def __sub__(self, other):
        self.tvar - other

class Parameter(object):
    def __init__(self, name, label=None, lower=None, upper=None):
        self.name = name
        if label:
            self.label = label
        else:
            self.label = name
        self.lower = lower
        self.upper = upper
        self.tvar = T.scalar(name)

    def __repr__(self):
        return self.label

    def __rmul__(self, other):
        other * self.tvar

    def __mul__(self, other):
        self.tvar * other

    def __rdiv__(self, other):
        other / self.tvar

    def __div__(self, other):
        self.tvar / other

    def __radd__(self, other):
        other + self.tvar

    def __add__(self, other):
        self.tvar + other

    def __rsub__(self, other):
        other - self.tvar

    def __sub__(self, other):
        self.tvar - other

class Distribution(object):
    def __init__(self):
        self.var = OD()
        self.param = OD()
        self.dist = OD()

    def _add_var(self, var):
        self.var[var.name] = var
        return var.tvar

    def _add_param(self, param, enforce_lower=None, enforce_upper=None):
        self.param[param.name] = param
        return param.tvar

    def _add_dist(self, dist, name):
        self.dist[name] = dist
        return dist

    def get_vars(self):
        ret = []
        unique = []
        ret += self.var.values()
        for dist in self.dist.values():
            ret += dist.get_vars()
        for par in ret:
            if par not in unique:
                unique.append(par)
        return unique

    def get_params(self):
        ret = []
        unique = []
        ret += self.param.values()
        for dist in self.dist.values():
            ret += dist.get_params()
        for par in ret:
            if par not in unique:
                unique.append(par)
        return unique

    def get_dists(self):
        return self.dist.values()

    def _get_vars_pars(self):
        variables = list(self.get_vars())
        parameters = list(self.get_params())

        var_args = []
        for var in variables:
            var_args.append(var.tvar)

        par_args = []
        for par in parameters:
            par_args.append(par.tvar)

        return var_args, par_args

    @memoize
    def pdf_compiled(self):
        logging.info('Compiling pdf...')
        vars, pars = self._get_vars_pars()
        return function(vars + pars, T.exp(self.logp()), allow_input_downcast=True)

    @memoize
    def cdf_compiled(self):
        logging.info('Compiling cdf...')
        vars, pars = self._get_vars_pars()
        return function(vars + pars, self.tcdf(), allow_input_downcast=True)

    @memoize
    def logp_compiled(self):
        logging.info('Compiling logp...')
        vars, pars = self._get_vars_pars()
        return function(vars + pars, -T.sum(self.logp()), allow_input_downcast=True)

    @memoize
    def grad_compiled(self):
        logging.info('Compiling grad_logp...')
        vars, pars = self._get_vars_pars()
        return function(vars + pars, T.grad(-T.sum(self.logp()), pars), allow_input_downcast=True)

    @memoize
    def hessian_compiled(self):
        logging.info('Compiling f_hessian...')
        vars, pars = self._get_vars_pars()
        return function(vars + pars, hessian_(-T.sum(self.logp()), pars)[0], allow_input_downcast=True)

    def fit(self, data, init, method="BFGS"):
        variables = list(self.get_vars())
        parameters = list(self.get_params())

        data_args = []
        for var in variables:
            if type(data) is ndarray or type(data) is recarray:
                if var.name not in data.dtype.names:
                    raise ValueError('Random variable {} required by model not found in dataset'.format(var.name))
            else:
                if var.name not in data:
                    raise ValueError('Random variable {} required by model not found in dataset'.format(var.name))
            data_args.append(np.array(data[var.name]))

        x0 = []
        for par in parameters:
            if par.name not in init:
                raise ValueError('No initial value specified for Parameter {}'.format(par.name))
            x0.append(init[par.name])

        obj_func = self.logp_compiled()
        obj_func_grad = self.grad_compiled()
        # obj_func_hessian = self.hessian_compiled()

        # We keep the data fixed while varying the parameters
        def func(pars):
            args = data_args + list(pars)
            return obj_func(*args)

        def func_grad(pars):
            args = data_args + list(pars)
            return np.array(obj_func_grad(*args))

        logging.info('Minimizing negative log-likelihood of model...')
        results = minimize(func, method=method, jac=func_grad, x0=x0)

        if len(parameters) > 1:
            ret = dict()
            for par, val in zip(parameters, results['x']):
                ret[par.name] = val
        else:
            ret = {parameters[0].name: results["x"]}


        results.x = ret

        return results

class Uniform(Distribution):
    def __init__(self, x, lower=0, upper=1, *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)
        self.x = self._add_var(x)
        try:
            self.lower = float(lower)
        except TypeError:
            self.lower = self._add_param(lower)
        try:
            self.upper = float(upper)
        except TypeError:
            self.upper = self._add_param(upper)

        self.pdf = self.pdf_compiled()
        self.cdf = self.cdf_compiled()

    def tcdf(self):
        x = self.x
        low = self.lower
        up = self.upper
        return T.switch(T.gt(x, up), 1, T.switch(T.lt(x, low), 0, (x - low)/(up - low)))

    def sample(self, num, init):
        lower = init[self.lower.name]
        upper = init[self.upper.name]
        return np.random.uniform(lower, upper, num)

    def logp(self):
        x = self.x
        upper = self.upper
        lower = self.lower
        return T.switch(T.gt(x, upper), 0, T.switch(T.lt(x, lower), 0, 1/(upper - lower)))

class Normal(Distribution):
    def __init__(self, x, mu=0, sigma=1, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.x = self._add_var(x)
        self.mu = self._add_param(mu)
        self.sigma = self._add_param(sigma, enforce_lower=0)
        self.pdf = self.pdf_compiled()
        self.cdf = self.cdf_compiled()

    def tcdf(self):
        return 0.5 * (1 + T.erf((self.x - self.mu)/(self.sigma*T.sqrt(2))))

    def sample(self, num, init):
        mu = init[self.mu.name]
        sigma = init[self.sigma.name]
        return np.random.normal(mu, sigma, num)

    def logp(self):
        x = self.x
        mu = self.mu
        sigma = self.sigma
        return bound(-(x - mu)**2 / (2 * sigma**2) + T.log(1 / T.sqrt(sigma**2 * 2 * pi)), sigma > 0)

class Exponential(Distribution):
    def __init__(self, x, beta=1, lower=0, upper=inf, *args, **kwargs):
        super(Exponential, self).__init__(*args, **kwargs)
        self.x = self._add_var(x)
        self.beta = self._add_param(beta)
        try:
            self.lower = float(lower)
        except TypeError:
            self.lower = self._add_param(lower)
        try:
            self.upper = float(upper)
        except TypeError:
            self.upper = self._add_param(upper)

        self.pdf = self.pdf_compiled()
        self.cdf = self.cdf_compiled()

    def tcdf(self):
        x = self.x
        beta = self.beta
        lower = self.lower
        upper = self.upper
        norm = T.exp(-lower/beta) - T.exp(-upper/beta)
        return T.switch(T.lt(x, lower), 0, (T.exp(-lower/beta) - T.exp(-x/beta))/norm)

    def sample(self, num, init):
        beta = init[self.beta.name]
        # FIXME implement bounds
        return np.random.exponential(beta, num)

    def logp(self):
        x = self.x
        beta = self.beta
        upper = self.upper
        lower = self.lower
        norm = 1/(T.exp(-lower/beta) - T.exp(-upper/beta))
        return bound(T.log(norm*T.exp(-x/beta)/beta), beta > 0)

class Power(Distribution):
    def __init__(self, x, alpha=1, lower=1, upper=inf, *args, **kwargs):
        super(Power, self).__init__(*args, **kwargs)
        self.x = self._add_var(x)
        self.alpha = self._add_param(alpha)
        try:
            self.lower = float(lower)
        except TypeError:
            self.lower = self._add_param(lower)
        try:
            self.upper = float(upper)
        except TypeError:
            self.upper = self._add_param(upper)

        self.pdf = self.pdf_compiled()
        self.cdf = self.cdf_compiled()

    def tcdf(self):
        x = self.x
        alpha = self.alpha
        lower = self.lower
        upper = self.upper
        norm = (alpha - 1)/(lower**(1-alpha) - upper**(1-alpha))
        return T.switch(T.lt(x, lower), 0, norm/(alpha - 1) * (lower**(1 - alpha) - x**(1 - alpha)))

    def sample(self, num, init):
        alpha = init[self.alpha.name]
        return np.random.power(alpha, num)

    def logp(self):
        x = self.x
        alpha = self.alpha
        upper = self.upper
        lower = self.lower
        norm = (alpha - 1)/(lower**(1-alpha) - upper**(1-alpha))
        return bound(T.log(norm * x**(-alpha)), alpha > 1)

class Mix2(Distribution):
    def __init__(self, frac, dist1, dist2, *args, **kwargs):
        super(Mix2, self).__init__(*args, **kwargs)
        self.frac = self._add_param(frac)
        self.dist1 = self._add_dist(dist1, 'dist1')
        self.dist2 = self._add_dist(dist2, 'dist2')
        self.tcdf = lambda: self.frac * self.dist1.tcdf() + (1-self.frac) * self.dist2.tcdf()
        self.pdf = self.pdf_compiled()
        self.cdf = self.cdf_compiled()

    def sample(self, num, init):
        frac = init[self.frac.name]
        N1 = np.random.binomial(num, frac)
        N2 = num - N1
        ret = np.append(self.dist1.sample(N1, init), self.dist2.sample(N2, init))
        np.random.shuffle(ret)
        return ret

    def logp(self):
        frac = self.frac
        dist1 = self.dist1
        dist2 = self.dist2
        return bound(T.log(frac * T.exp(dist1.logp()) + (1 - frac) * T.exp(dist2.logp())), frac > 0, frac < 1)

