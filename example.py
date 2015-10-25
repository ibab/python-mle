import numpy as np
from mle import var, Normal

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
