import numpy as np
import pandas as pd
from scipy.stats import norm

from fintools.statistics import skewness, kurtosis


def semi_deviation(returns: pd.DataFrame):
    """
    Standard deviation of negative returns
    :param returns: historical returns sequence
    :return: the semi-deviation metric
    """
    return returns[returns < 0].std(ddof=0)


def historic_VaR(returns, confidence_level=5):
    """
    Compute Historic Value at Risk
    :param returns: historical returns sequence
    :param confidence_level: percentile at which to calculate VaR
    :return: the Value at Risk
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(historic_VaR, confidence_level=confidence_level)
    return -np.percentile(returns, confidence_level)


def parametric_VaR(returns, confidence_level=5, modified=True):
    """
    Compute parametric Value at Risk
    If 'modified is False, then it returns Gaussian VaR, otherwise Cornish-Fisher
    :param returns: a vector of returns
    :param confidence_level: percentile at which to calculate VaR
    :param modified: return modified VaR if True, otherwise normal VaR
    :return: the Value at Risk
    """
    # Z score
    z = norm.ppf(confidence_level / 100)
    if modified:
        # Modify Z score by skewness and kurtosis
        s = skewness(returns)
        k = kurtosis(returns)
        z += (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * (k - 3) / 24 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36
    return -(returns.mean() + z * returns.std(ddof=0))


def conditional_VaR(returns, confidence_level=5):
    """
    Compute the Conditional VaR of an array, Series, or DataFrame
    :param returns: A vector of returns
    :param confidence_level: percentile at which to calculate VaR
    :return: the Value at Risk
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(conditional_VaR, confidence_level=confidence_level)
    is_beyond = returns <= - historic_VaR(returns, confidence_level)
    return -returns[is_beyond].mean()


def drawdown(returns: pd.Series, initial_wealth=1000):
    """
    Takes a time series of asset returns
    :param returns: historical returns sequence
    :param initial_wealth: Initial wealth invested
    :rtype: Drawdown
    :return: a Drawdown object
    """
    wealth_index = initial_wealth * (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return Drawdown(wealth_index, previous_peaks, drawdowns)


class Drawdown:
    """
    History of drawdowns from a return sequence
    """

    def __init__(self, wealth, peaks, drawdowns):
        self.wealth = wealth
        self.peaks = peaks
        self.drawdowns = drawdowns
        self.history = pd.DataFrame({
            "Wealth": wealth,
            "Peaks": peaks,
            "Drawdown": drawdowns
        })
        self.max_drawdown = drawdowns.min()
        self.max_drawdown_index = drawdowns.idxmin()

