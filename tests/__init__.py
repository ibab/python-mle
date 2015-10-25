def test_formula_transform():
    """
    Check if variables can be added/multiplied/transformed.
    The resulting expression can be plugged into a model.
    """
    from mle import var

    x = var('x', vector=True, observed=True)
    a = var('a')
    b = var('b')

    a * x**2 + b


def test_const():
    """
    Check if parameters can be set to be constant.
    """
    from mle import var, Normal
    import numpy as np
    x = var('x', vector=True, observed=True)
    mu = var('mu', const=True)
    sigma = var('sigma')

    model = Normal(x, mu, sigma)
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)

    results = model.fit({'x': data}, {'mu': 1, 'sigma': 1})
    assert(results.x['mu'] == 1)


# @raises(ValueError)
# def test_error_on_illegal_bound():
#     """
#     Check if exception is raised when user specifies illegal bound.
#     Some distributions automatically apply certain bounds.
#     Example: sigma > 0 for the Normal distribution.
#     If a user-specified bound conflicts with that, an exception should be thrown.
#     """
#     from mle import var, Normal

#     x = var('x', vector=True, observed=True)
#     mu = var('mu')
#     sigma = var('sigma', lower=-1)

#     Normal(x, mu, sigma)


def test_simple_fit():
    """
    Check if fitting Gaussian data works
    """
    from mle import Normal, var
    import numpy as np

    x = var('x', vector=True, observed=True)
    mu = var('mu')
    sigma = var('sigma')

    dist = Normal(x, mu, sigma)
    np.random.seed(42)

    data = np.random.normal(0, 1, 100000)

    try:
        dist.fit({'x': data}, {'mu': 1, 'sigma': 2}, method='BFGS')
    except:
        assert False, 'Fitting generated data failed'


def test_linear_regression():
    """
    Check if fitting a linear model works.
    """
    from mle import Normal, var
    import numpy as np

    x = var('x', vector=True, observed=True)
    y = var('y', vector=True, observed=True)

    a = var('a')
    b = var('b')
    sigma = var('sigma')

    model = Normal(y, a * x + b, sigma)
    np.random.seed(42)

    xs = np.linspace(0, 1, 20)
    ys = 0.5 * xs - 0.3 + np.random.normal(0, 0.2, 20)

    results = model.fit({'x': xs, 'y': ys}, {'a': 2, 'b': 1, 'sigma': 1})
    print(results)


def test_pdf_product():
    """
    Check if PDF models can be joined
    """
    from mle import var, Normal, Join
    x = var('x', vector=True, observed=True)
    y = var('y', observed=True)
    mu = var('mu')
    sigma = var('sigma')

    model = Join(Join(Normal(x, mu, sigma)), Normal(y, mu, sigma))
    assert(model.observed == [x, y])
