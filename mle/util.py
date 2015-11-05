import collections
import functools

from matplotlib import pylab
import numpy as np
from scipy import integrate
from theano import Variable, scan
from theano.tensor import arange, grad, stack
from theano.gradient import format_as


def hessian_(cost, wrt, consider_constant=None, disconnected_inputs='raise'):
    """

    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Vector (1-dimensional tensor) 'Variable' or list of
               vectors (1-dimensional tensors) Variables
    :param consider_constant: a list of expressions not to backpropagate through
    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.
    :return: either a instance of Variable or list/tuple of Variables
            (depending upon `wrt`) repressenting the Hessian of the `cost`
            with respect to (elements of) `wrt`. If an element of `wrt` is not
            differentiable with respect to the output, then a zero
            variable is returned. The return value is of same type
            as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    # Check inputs have the right format
    assert isinstance(cost, Variable), "tensor.hessian expects a Variable as `cost`"
    assert cost.ndim == 0, "tensor.hessian expects a 0 dimensional variable as `cost`"

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    expr = grad(cost, wrt, consider_constant=consider_constant,
                disconnected_inputs=disconnected_inputs)

    # It is possible that the inputs are disconnected from expr,
    # even if they are connected to cost.
    # This should not be an error.
    hess, updates = scan(
        lambda i, y: grad(y[i], wrt, consider_constant=consider_constant,
                          disconnected_inputs='ignore'),
        sequences=arange(len(expr)), non_sequences=[expr]
    )
    assert not updates, ("Scan has returned a list of updates. This should not happen! Report "
                         "this to theano-users (also include the script that generated the error)")

    return format_as(using_list, using_tuple, stack(hess)[0])


def memoize(obj):
    """An expensive memoizer that works with unhashables."""
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = (hashable(args), hashable(kwargs))

        if key not in cache:
            cache[key] = obj(*args, **kwargs)

        return cache[key]
    return memoizer


def hashable(a):
    """Turn some unhashable objects into hashable ones."""
    if isinstance(a, dict):
        return hashable(a.items())
    try:
        return tuple(map(hashable, a))
    except:
        return a


def make_plotter(data, results, observables):
    """Make a plot function for `results`."""
    assert(len(observables) == 1)
    observable = observables[0]

    # TODO Find a nicer solution to having this wrapper
    def _pdf(data):
        if isinstance(data, collections.Iterable):
            return np.array([_pdf(val) for val in data])
        else:
            return float(results['func'](data))

    def _plot(*args, **kwargs):
        return plot_fitted_hist(_pdf, data, observable, *args, **kwargs)

    return _plot


def plot_fitted_hist(pdf, data, observable, nbins=None, lower=None, upper=None, residuals=True):
    data = data[observable.name]
    # Configuration
    nbins = nbins or 100
    lower = lower or max(observable._lower, min(data))
    upper = upper or min(observable._upper, max(data))

    figure = pylab.figure(1)

    # Plot the data in a histogram with sqrt(N) error bars
    figure.add_axes((.1, .3, .8, .8))
    bin_vals, bin_edges = np.histogram(
        data[(lower < data) & (data < upper)],
        bins=np.linspace(lower, upper, nbins+1)
    )
    bin_centers = bin_edges[:-1] + 0.5*np.diff(bin_edges)
    pylab.errorbar(
        bin_centers, bin_vals,
        xerr=0.5*np.diff(bin_edges), yerr=np.sqrt(bin_vals), fmt='none'
    )

    # Plot fitted pdf
    x = np.linspace(lower, upper, nbins*10)
    integral = integrate.quad(pdf, lower, upper)[0]
    correction = sum(bin_vals*np.diff(bin_edges))/integral
    y = pdf(x)*correction

    pylab.plot(x, y, lw=1.5)
    pylab.xlim([lower, upper])
    pylab.tick_params(top='off', right='off', left='off')

    if not residuals:
        return

    # Make the residual plot
    figure.add_axes((.1, .04, .8, .2))
    difference = (pdf(bin_centers)*correction - bin_vals) / np.sqrt(bin_vals)
    pylab.bar(bin_edges[:-1], difference, np.diff(bin_edges))
    pylab.xlim([lower, upper])
    pylab.ylim([-4, 4])
    pylab.yticks(np.arange(-4, 5, 1))
    pylab.tick_params(top='off', right='off', left='off')
