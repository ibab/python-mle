
from iminuit import Minuit
from iminuit.util import make_func_code

class Min_Func:
    def __init__(self, f, names):
        self.f = f
        self.func_code = make_func_code(names)
        self.func_defaults = None
    def __call__(self, *args):
        return self.f(args)

def fmin_minuit(func, names, x0):
    inits = dict()
    for n, x in zip(names, x0):
        inits[n] = x

    m = Minuit(Min_Func(func, names), print_level=2, **inits)
    m.migrad()
    return m

