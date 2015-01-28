
def test_distribution():
    from mle import Normal, var, par
    import theano.tensor as T

    x = var('x')
    mu = par('mu')
    sigma = par('sigma')

    dist = Normal(x, mu, sigma)

    assert(len(dist.get_vars()) == 1)
    assert(len(dist.get_params()) == 2)
    assert(len(dist.get_dists()) == 0)


