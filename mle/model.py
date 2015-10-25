import logging
from time import clock

import numpy as np
from scipy.optimize import minimize
from theano import function, gof, shared, config
import theano.tensor as T

from mle.util import memoize

__all__ = ['Model']


class Model(object):
    def __init__(self):
        self.exprs = dict()
        self.submodels = dict()

    def fit(self, data, init, method='BFGS', verbose=False):
        data_args = []
        shared_params = []
        for var in self.observed:
            try:
                data[var.name]
            except KeyError:
                raise ValueError('Random variable {} required by model not found in dataset'
                                 .format(var.name))
            except:
                raise ValueError('The fitted dataset must support string indexing')

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

        logp = function(self.constant + self.floating, -T.sum(self._logp),
                        givens=shared_params, allow_input_downcast=True)
        g_logp = function(self.constant + self.floating, T.grad(-T.sum(self._logp), self.floating),
                          givens=shared_params, allow_input_downcast=True)

        def func(pars):
            return logp(*(const + list(pars)))

        def g_func(pars):
            return np.array(g_logp(*(const + list(pars))))

        logging.info('Minimizing negative log-likelihood of model...')

        start = clock()
        if method.upper() == 'MINUIT':
            from .minuit import fmin_minuit
            results = fmin_minuit(func, x0, map(str, self.floating), verbose=verbose)
        else:
            results = minimize(func, method=method, jac=g_func, x0=x0, options={'disp': True})
            names = [x.name for x in self.parameters]
            results.x = {n: x for n, x in zip(names, results.x)}
        fit_time = clock() - start

        # Add constant parameters to results
        for par in self.parameters:
            if par._const:
                results.x[par.name] = init[par.name]

        results['fit_time'] = fit_time

        return results

    def _add_expr(self, name, expr):
        self.exprs[name] = expr

    def _add_submodel(self, name, model):
        self.submodels[name] = model

    def _add_compiled_expr(self, name, expr):

        @memoize
        def compiler():
            logging.info('Compiling {}...'.format(name))
            return function(self.observed + self.parameters, expr, allow_input_downcast=True)

        setattr(self, 'compile_' + name, compiler)

        def compiled(model, *args):
            compiler = getattr(self, 'compile_' + name)
            func = compiler()
            return func(*args)

        setattr(self, name, compiled)

    @property
    def variables(self):
        return filter(lambda x: isinstance(x, T.TensorVariable), gof.graph.inputs([self._logp]))

    @property
    def observed(self):
        return list(filter(lambda x: x._observed, self.variables))

    @property
    def parameters(self):
        return list(filter(lambda x: not x._observed, self.variables))

    @property
    def constant(self):
        return list(filter(lambda x: not x._observed and x._const, self.variables))

    @property
    def floating(self):
        return list(filter(lambda x: not x._const, self.variables))
