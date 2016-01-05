from setuptools import setup

setup(name='mle',
      version='0.1',
      description='High Performance Maximum Likelihood Estimations',
      url='http://github.com/ibab/python-mle',
      author='Igor Babuschkin',
      author_email='igor@babuschk.in',
      license='MIT',
      packages=['mle', 'mle.distributions'],
      install_requires=[
          'numpy',
          'scipy',
          'theano',
      ],
      zip_safe=False)
