
def test_formula_transform():
    """
    Check if variables can be added/multiplied/transformed.
    The result should be a formula that can be plugged into a model.
    """
    from mle import var, par

    x = var('x')
    a = par('a')
    b = par('b')

    formula = a * x**2 + b

def test_simple_fit():
    """
    Check if generating/fitting Gaussian data works
    """
    from mle import Normal, var, par
    import theano.tensor as T
    import numpy as np

    x = var('x')
    mu = par('mu')
    sigma = par('sigma')

    dist = Normal(x, mu, sigma)
    np.random.seed(42)
    data = dist.sample(1e6, {'mu': 0, 'sigma': 1})
    results = dist.fit({'x': data}, {'mu': 1, 'sigma': 2}, method='L-BFGS-B')

def test_linear_regression():
    """
    Check if fitting a linear model works
    """
    from mle import Normal, var, par
    import theano.tensor as T
    import numpy as np

    x = var('x')
    y = var('y')

    a = par('a')
    b = par('b')
    sigma = par('sigma')

    dist = Normal(y, a * x + b, sigma)
    np.random.seed(42)

    xs = linspace(0, 1, 20)
    ys = dist.sample(20, {'x': xs, 'a': 1, 'b': 0, 'sigma': 0.5})

    results = dist.fit({'x': xs, 'y': ys}, {'a': 2, 'b': 1, 'sigma': 1})


