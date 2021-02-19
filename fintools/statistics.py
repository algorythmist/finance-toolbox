import scipy.stats
import pandas as pd


def moment(data, n):
    demeaned = data - data.mean()
    sigma = data.std(ddof=0)
    numerator = (demeaned**n).mean()
    return numerator/sigma**n


def skewness(data):
    """
    Alternative to scipy.stats.skew
    """
    return moment(data,3)


def kurtosis(data, excess=False):
    """
    Alternative to scipy.stats.kurtosis
    """
    if excess:
        return moment(data,4)-3
    return moment(data,4)


def is_normal(data: pd.Series, confidence_level=0.001):
    """
    Applies the Jarque-Bera test to determin if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if normal hypothesis is accepted, False otherwise
    """
    result = scipy.stats.jarque_bera(data)
    return result.pvalue > confidence_level