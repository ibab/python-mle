
from scipy.optimize import minimize
from theano import function, scan
import theano.tensor as T
from numpy import inf, array, ndarray
from numpy.core.records import recarray
import numpy as np
import logging

from .util import hessian_
from .memoize import memoize

__all__ = ['Model']

class Model(object):
    def __init__(self):
        self.exprs = dict()
        self.subdists = dict()

        self._add_compiled_expr('logp', self._logp())
        self._add_compiled_expr('pdf', T.exp(self._logp()))
        self._add_compiled_expr('cdf', self._cdf())
        self._add_compiled_expr('grad_logp', T.grad(self._logp()))
        self._add_compiled_expr('hess_logp', hessian_(self._logp()))

    def get_observed(self):
        pass
    
    def get_params(self):
        pass

    def fit(self, data, init, method='L-BFGS-B'):
        obs = self.get_observed()
        params = self.get_params()

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
        for par in params:
            if par.name not in init:
                raise ValueError('No initial value specified for Parameter {}'.format(par.name))
            x0.append(init[par.name])

        func = lambda pars: self.logp(data_args + list(pars))
        grad_func = lambda pars: self.grad_logp(data_args + list(pars))

        logging.info('Minimizing negative log-likelihood of model...')
        results = minimize(func, method=method, jac=grad_func, x0=x0)

        ret = dict()
        for par, val in zip(parameters, results['x']):
            ret[par.name] = val

        results.x = ret

        return results

    def _add_expr(self, name, expr):
        self.exprs[name] = expr

    def _add_submodel(self, name, model):
        self.submodels[name] = model

    def _add_compiled_expr(self, name, expr):

        @memoize
        def compile(model):
            logging.info('Compiling {}...'.format(name))
            obs = model.get_observed()
            params = model.get_params()
            return function(obs + params, , allow_input_downcast=True)

        setattr(self, 'compile_' + name, compile)

        @property
        def compiled(model, *args):
            compiler = getattr(self, 'compile_' + name)
            func = compiler()
            return func(*args)
        
        setattr(self, name, compiled)

