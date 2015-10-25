import functools

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
