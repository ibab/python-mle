from scipy.stats import chisquare, kstest
from scipy.integrate import quad
import matplotlib.pyplot as plt
from numpy import inf, zeros_like

def kolsmi(dist, fit_result, data):
    """
    Kolmogorow–Smirnow-Test for goodness of fit,
    tests the H0 hypothesis, if data is a sample of dist

    Parameters:
        dist :         A mle.Distribution instance
        fit_result:    The solution dict, returned by the Distribution.fit method
        data :         The data used in Distribution.fit
        alpha :        The wanted confidence level
    Returns:
        teststat :     the test statistic, e.g. the max distance between the
                       cumulated distributions
        p-value :      the p-value, probability that dist describes the data
    """
    teststat, pvalue = kstest(data, lambda x: dist.cdf(x, **fit_result["x"]))
    return teststat, pvalue
