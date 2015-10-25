try:
    from iminuit import Minuit
except ImportError:
    raise ImportError("The iminuit package must be installed in order to use `method='MINUIT'`")

from iminuit.util import make_func_code
from scipy.optimize import OptimizeResult


class Min_Func:

    def __init__(self, f, names):
        self.f = f
        self.func_code = make_func_code(names)
        self.func_defaults = None

    def __call__(self, *args):
        return self.f(args)


def fmin_minuit(func, x0, names=None, verbose=False):
    inits = dict()

    if verbose:
        print_level = 2
    else:
        print_level = 0

    if names is None:
        names = map(lambda x: 'param' + str(x), range(len(x0)))
    else:
        assert(len(x0) == len(names))

    for n, x in zip(names, x0):
        inits[n] = x
        # TODO use a method to set this correctly
        inits['error_' + n] = 1

    m = Minuit(Min_Func(func, names), print_level=print_level, errordef=1, **inits)
    a, b = m.migrad()

    return OptimizeResult(
        x=m.values,
        fun=a['fval'],
        edm=a['edm'],
        nfev=a['nfcn'],
        is_valid=a['is_valid'],
        has_valid_parameters=a['has_valid_parameters'],
    )
