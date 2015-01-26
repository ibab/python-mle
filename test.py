
def test_normal():
    from theano import function
    import theano.tensor as T
    from distributions import Normal

    x = T.dvector('x')
    mu = T.dscalar('mu')
    sigma = T.dscalar('sigma')

    norm = Normal(mu, sigma)

    ret = T.sum(norm.logp(x))
    f = function([x, mu, sigma], ret)
    f([1, 2, 3], 0, 1)

def test_mix2():
    from theano import function
    import theano.tensor as T
    from distributions import Normal, Uniform, Mix2

    x = T.dvector('x')
    mu = T.dscalar('mu')
    sigma = T.dscalar('sigma')
    frac = T.dscalar('frac')
    lower = T.dscalar('lower')
    upper = T.dscalar('upper')

    unif = Uniform(lower, upper)
    norm = Normal(mu, sigma)

    mix = Mix2(frac, unif, norm)

    ret = T.sum(mix.logp(x))
    f = function([x, mu, sigma, frac, lower, upper], ret)
    f([1, 2, 3], 0, 1, 0.5, 0, 1)

