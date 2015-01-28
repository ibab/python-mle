# python-mle

A Python package for performing Maximum Likelihood Estimates efficiently.

## Example

```python
# Create model
from mle import Normal, Mix2, var, par

x = var('x')
mu = par('mu')
frac = par('frac')
sigma1 = par('sigma1')
sigma2 = par('sigma2')

dist1 = Normal(x, mu, sigma1)
dist2 = Normal(x, mu, sigma2)
model = Mix2(frac, dist1, dist2)

# Generate data

import numpy as np
from numpy import append
from numpy.random import normal, shuffle

xs = append(normal(0, 1, 1000), normal(0, 1.5, 1000))
shuffle(xs)

# Fit model

init = {'mu': 0, 'sigma1': 1, 'sigma2': 3, 'frac': 0.6}
print(model.fit({'x': xs}, init))
```

