# python-mle

[![Join the chat at https://gitter.im/ibab/python-mle](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ibab/python-mle?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A Python package for performing Maximum Likelihood Estimates.

Inspired by [RooFit](https://root.cern.ch/drupal/content/roofit) and [pymc](https://github.com/pymc-devs/pymc).

mle is a Python framework for constructing probability models and estimating their parameters from data using the [Maximum Likelihood](http://en.wikipedia.org/wiki/Maximum_likelihood) approach.
While being less flexible than a full Bayesian probabilistic modeling framework, it can handle larger datasets (> 10^6 entries) and more complex statistical models.

To achieve maximum performance, this package (like pymc) uses [Theano](http://deeplearning.net/software/theano/tutorial/) to optimize and compile statistical models.
This also means that models can automatically be evaluated using multiple CPU cores or GPUs.
Derivatives used for the likelihood optimization are calculated using automatic differentiation.

Currently, the package is only a basic prototype and will change heavily in the future.

## Example

```python
# Create model
from mle import Normal, Mix2, var, par

x = var('x')
mu = par('mu')
frac = par('frac')
sigma1 = par('sigma1')
sigma2 = par('sigma2')

# Mixture model of two Gaussians centered on each other
model = Mix2(frac, Normal(x, mu, sigma1), Normal(x, mu, sigma2))

true_vals = {'mu': 0, 'sigma1': 2, 'sigma2': 4, 'frac': 0.5}
xs = model.sample(1e6, true_vals)

# Fit model
init = {'mu': 0, 'sigma1': 1, 'sigma2': 3, 'frac': 0.6}
print(model.fit({'x': xs}, init))
```

