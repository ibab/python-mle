# python-mle

[![Build Status](https://travis-ci.org/ibab/python-mle.svg?branch=master)](https://travis-ci.org/ibab/python-mle)

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
import numpy as np
from mle import *

# Define model
x = var('x', observed=True, vector=True)
y = var('y', observed=True, vector=True)

a = var('a')
b = var('b')
sigma = var('sigma')

model = Normal(y, a * x + b, sigma)

# Generate data
xs = np.linspace(0, 2, 20)
ys = 0.5 * xs + 0.3 + np.random.normal(0, 0.1, 20)

# Fit model to data
result = model.fit({'x': xs, 'y': ys}, {'a': 1, 'b': 1, 'sigma': 1})
print(result)
```

