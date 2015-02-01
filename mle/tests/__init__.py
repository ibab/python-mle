from scipy.stats import chisquare as _chisquare, kstest as _kstest
import numpy as _np

def kolsmi(dist, fit_result, data):
    """
    Perform a Kolmogorow-Smirnow-Test for goodness of fit.

    This tests the H0 hypothesis, if data is a sample of dist

    Args:
        dist:         A mle.Distribution instance
        fit_result:   The solution dict, returned by the Distribution.fit method
        data:         The data used in Distribution.fit
    Returns:
        teststat:     the test statistic, e.g. the max distance between the
                      cumulated distributions
        p-value:      the p-value, probability that dist describes the data
    """
    teststat, pvalue = _kstest(data, lambda x: dist.cdf(x, **fit_result["x"]))
    return teststat, pvalue

def chisquare(dist, fit_result, data, bins=None, range=None):
    """
    Perform a Chi^2 test for goodness of fit.

    Tests the H0 hypothesis if the distances between fit result and
    data are compatible  with random fluctuations.

    Args:
        dist:         A mle.Distribution instance
        fit_result:   The solution dict, returned by the Distribution.fit method
        data:         The data used in Distribution.fit
        bins:         Number of bins for the histogram (default: 1+log2(N))
        range:        Range for the histogram (default: min(data), max(data))
    Returns:
        chisquare:    the test statistic, chi^2/ndf
        p-value:      the p-value, probability that differences between dist
                      and data are compatible with random fluctuation
    """

    # rule of thumb for number if bins if not provided
    if bins is None:
        bins = _np.ceil(2*len(data)**(1.0/3.0))

    entries, edges = _np.histogram(data, bins=bins, range=range)

    # get expected frequencies from the cdf
    cdf = dist.cdf(edges, **fit_result["x"])
    exp_entries = _np.round(len(data) * (cdf[1:] - cdf[:-1]))

    # use only bins where more then 4 entries are expected
    mask = exp_entries >= 5

    chisq, pvalue = _chisquare(entries[mask],
                               exp_entries[mask],
                               ddof=len(fit_result["x"])
                               )
    chisq = chisq/(_np.sum(mask) - len(fit_result["x"]) - 1)
    return chisq, pvalue
