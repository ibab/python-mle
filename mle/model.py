
from scipy.optimize import minimize
from theano import function, scan, gof, shared, config
import theano.tensor as T
from numpy import inf, array, ndarray
from numpy.core.records import recarray
import numpy as np
import logging
from itertools import chain
from time import clock
from .minuit import *

from .util import hessian_, memoize

__all__ = ['Model']

class Model(object):
    def __init__(self):
        self.exprs = dict()
        self.submodels = dict()

    def fit(self, data, init, method='BFGS', verbose=False):
        
        data_args = []
        shared_params = []
        for var in self.observed:
            if type(data) is ndarray or type(data) is recarray:
                if var.name not in data.dtype.names:
                    raise ValueError('Random variable {} required by model not found in dataset'.format(var.name))
            else:
                if var.name not in data:
                    raise ValueError('Random variable {} required by model not found in dataset'.format(var.name))
            data_args.append(np.array(data[var.name]))
            shared_params.append((var, shared(data[var.name].astype(config.floatX), borrow=True)))

        const = []
        x0 = []
        for par in self.parameters:
            if par.name not in init:
                raise ValueError('No initial value specified for Parameter {}'.format(par.name))
            if par._const:
                const.append(init[par.name])
            else:
                x0.append(init[par.name])

        logp = function(self.constant + self.floating, -T.sum(self._logp), givens=shared_params, allow_input_downcast=True)
        g_logp = function(self.constant + self.floating, T.grad(-T.sum(self._logp), self.floating), givens=shared_params, allow_input_downcast=True)

        func = lambda pars: logp(*(const + list(pars)))
        g_func = lambda pars: np.array(g_logp(*(const + list(pars))))

        logging.info('Minimizing negative log-likelihood of model...')
        start = clock()

        if method.upper() == 'MINUIT':
            results = fmin_minuit(func, x0, map(str, self.floating), verbose=verbose)
        else:
            results = minimize(func, method=method, jac=g_func, x0=x0, options={'disp':True})

        fit_time = clock() - start
        results['fit_time'] = fit_time

        return results

    def _add_expr(self, name, expr):
        self.exprs[name] = expr

    def _add_submodel(self, name, model):
        self.submodels[name] = model

    def _add_compiled_expr(self, name, expr):

        @memoize
        def compile():
            logging.info('Compiling {}...'.format(name))
            return function(self.observed + self.parameters, expr, allow_input_downcast=True)

        setattr(self, 'compile_' + name, compile)

        def compiled(model, *args):
            compiler = getattr(self, 'compile_' + name)
            func = compiler()
            return func(*args)
        
        setattr(self, name, compiled)

    @property
    def observed(self):
        result = gof.graph.inputs([self._logp])
        return list(filter(lambda x: isinstance(x, T.TensorVariable) and x._observed, result))

    @property
    def parameters(self):
        result = gof.graph.inputs([self._logp])
        return list(filter(lambda x: isinstance(x, T.TensorVariable) and not x._observed, result))

    @property
    def constant(self):
        result = gof.graph.inputs([self._logp])
        return list(filter(lambda x: isinstance(x, T.TensorVariable) and not x._observed and x._const, result))

    @property
    def floating(self):
        result = gof.graph.inputs([self._logp])
        return list(filter(lambda x: isinstance(x, T.TensorVariable) and not x._const, result))

