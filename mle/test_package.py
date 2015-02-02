
def test_formula_transform():
    """
    Check if variables can be added/multiplied/transformed.
    The resulting expression can be plugged into a model.
    """
    from mle import vec, var

    x = vec('x')
    a = var('a')
    b = var('b')

    formula = a * x**2 + b

def test_const():
    """
    Check if parameters can be set to be constant.
    """
    from mle import var, vec, Normal
    import numpy as np
    x = vec('x', observed=True)
    mu = var('mu', const=True)
    sigma = var('sigma')

    model = Normal(x, mu, sigma)
    np.random.seed(42)
    data = model.sample(200, {'mu': 0, 'sigma': 1})

    results = model.fit(data, {'mu': 1, 'sigma'})
    assert(results.x['mu'] == 1)

@raises(ValueError)
def test_error_on_illegal_bound():
    """
    Check if exception is raised when user specifies illegal bound.
    Some distributions automatically apply certain bounds.
    Example: sigma > 0 for the Normal distribution.
    If a user-specified bound conflicts with that, an exception should be thrown.
    """
    from mle import vec, var

    x = vec('x', observed=True)
    mu = var('mu')
    sigma = var('sigma', lower=-1)

    Normal(x, mu, sigma)

def test_simple_fit():
    """
    Check if generating/fitting Gaussian data works
    """
    from mle import Normal, vec, var
    import numpy as np

    x = vec('x', observed=True)
    mu = var('mu')
    sigma = var('sigma')

    dist = Normal(x, mu, sigma)
    np.random.seed(42)
    data = dist.sample(1e6, {'mu': 0, 'sigma': 1})
    results = dist.fit(data, {'mu': 1, 'sigma': 2}, method='L-BFGS-B')

def test_linear_regression():
    """
    Check if fitting a linear model works.
    This requires several things to work properly:
        - Passing expressions to distributions
        - Generating with certain fixed observed values
        - Fitting in general
    """
    from mle import Normal, vec, var
    import numpy as np

    x = vec('x', observed=True)
    y = vec('y', observed=True)

    a = par('a')
    b = par('b')
    sigma = par('sigma')

    dist = Normal(y, a * x + b, sigma)
    np.random.seed(42)

    xs = linspace(0, 1, 20)
    data = dist.sample(20, {'x': xs, 'a': 1, 'b': 0, 'sigma': 0.5})

    data['x'] = xs

    results = dist.fit(data, {'a': 2, 'b': 1, 'sigma': 1})


