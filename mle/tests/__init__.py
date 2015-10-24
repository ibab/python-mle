from scipy import stats
import numpy as np

__all__ = ['chisquare', 'kolsmi']


def kolsmi(dist, fit_result, data):
    """Perform a Kolmogorow-Smirnow-Test for goodness of fit.

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
    variables = dist.get_vars()
    if len(variables) > 1:
        raise ValueError("Kolmogorov-Smirnov-Test is only valid for 1d distributions")
    var = variables[0]
    teststat, pvalue = stats.kstest(data[var.name], lambda x: dist.cdf(x, **fit_result["x"]))
    return teststat, pvalue


def chisquare(dist, fit_result, data, bins=None, range=None):
    """Perform a Chi^2 test for goodness of fit.

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
    variables = dist.get_vars()
    if len(variables) > 1:
        raise ValueError("This is a 1d only chisquare test")
    var = variables[0]

    # rule of thumb for number if bins if not provided
    if bins is None:
        bins = np.ceil(2*len(data[var.name])**(1.0/3.0))

    entries, edges = np.histogram(data[var.name], bins=bins, range=range)

    # get expected frequencies from the cdf
    cdf = dist.cdf(edges, **fit_result["x"])
    exp_entries = np.round(len(data[var.name]) * (cdf[1:]-cdf[:-1]))

    # use only bins where more then 4 entries are expected
    mask = exp_entries >= 5

    chisq, pvalue = stats.chisquare(entries[mask], exp_entries[mask], ddof=len(fit_result["x"]))
    chisq = chisq/(np.sum(mask) - len(fit_result["x"]) - 1)
    return chisq, pvalue
