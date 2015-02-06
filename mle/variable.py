
import theano.tensor as T
import numpy as np

__all__ = ['var', 'vec']

def var(*args, **kwargs):
    return Variable(*args, **kwargs)

def vec(*args, **kwargs):
    return Vector(*args, **kwargs)

class Variable(T.TensorVariable):
    '''
    A random variable or a parameter of the model.
    '''
    def __init__(self, name, label=None, observed=False, const=False, lower=None, upper=None):
        super(Variable, self).__init__(T.TensorType('floatX', ()), name=name)
        self.name = name
        self.label = label
        self.observed = observed
        self.const = const
        if lower is None:
            self.lower = -np.inf
        else:
            self.lower = lower
        if upper is None:
            self.upper = np.inf
        else:
            self.upper = upper

    def add_bounds(self, lower=None, upper=None):
        if not lower and not upper:
            raise ValueError('No bounds specified')
        if lower:
            if lower >= self.lower and lower < self.upper:
                self.lower = lower
            else:
                raise ValueError('Enforced value {} for lower not in [{}, {})'.format(lower, self.lower, self.upper))
        if upper:
            if upper > self.lower and upper <= self.upper:
                self.upper = upper
            else:
                raise ValueError('Enforced value {} for upper  not in ({}, {}]'.format(upper, self.lower, self.upper))

class Vector(T.TensorVariable):
    '''
    A vector of random variables.
    Currently, these can only function as observed data.
    '''
    def __init__(self, name, label=None):
        super(Vector, self).__init__(T.TensorType('floatX', (True,)))
        self.name = name
        self.label = label

