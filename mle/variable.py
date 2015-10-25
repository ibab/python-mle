import theano.tensor as T
import numpy as np

__all__ = ['var']


def var(name, label=None, observed=False, const=False, vector=False, lower=None, upper=None):
    if vector and not observed:
        raise ValueError('Currently, only observed variables can be vectors')

    if observed and const:
        raise ValueError('Observed variables are automatically const')

    if vector:
        var = T.vector(name)
    else:
        var = T.scalar(name)

    var._name = name
    var._label = label
    var._observed = observed
    var._const = observed or const
    var._lower = lower or -np.inf
    var._upper = upper or np.inf

    return var
