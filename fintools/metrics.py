import numpy as np

import pandas as pd
from scipy.stats import norm, skew, kurtosis

from fintools import annualize_returns, annualize_volatility, annualized_sharpe_ratio, compute_compound_return


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
        s = skew(returns)
        k = kurtosis(returns, fisher=True, bias=False)
        z += (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36
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


def compute_drawdown(returns: pd.Series, initial_wealth=1000):
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
        self.__df = pd.DataFrame({
            "wealth": wealth,
            "peaks": peaks,
            "drawdown": drawdowns
        })
        self.max_drawdown = drawdowns.min()
        self.max_drawdown_index = drawdowns.idxmin()

    def as_data_frame(self):
        return self.__df


def collect_metrics(returns, risk_free_rate=0.0):
    """
    Return a DataFrame that contains aggregated summary stats for the returns
    :param: returns: A vector or Data Frame of returns
    :param: risk_free_rate: The risk free rate (constant)
    """
    compound_return = compute_compound_return(returns)
    annualized_return = returns.aggregate(annualize_returns, periods_in_year=12)
    annualized_volatility = returns.aggregate(annualize_volatility, periods_in_year=12)
    annualized_sharpe = returns.aggregate(annualized_sharpe_ratio, risk_free_rate=risk_free_rate, periods_in_year=12)
    dd = returns.agg(lambda r: compute_drawdown(r).max_drawdown)
    skewness = returns.skew()
    kurt = returns.kurt()
    cf_var5 = parametric_VaR(returns, confidence_level=5)
    hist_var5 = historic_VaR(returns, confidence_level=5)
    conditional_var5 = conditional_VaR(returns, confidence_level=5)

    result = {
        "compound_return": compound_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "skewness": skewness,
        "excess_kurtosis": kurt,
        "cornish_fisher_var": cf_var5,
        "historic_var": hist_var5,
        "conditional_var": conditional_var5,
        "sharpe_ratio": annualized_sharpe,
        "max_drawdown": dd
    }
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(result)
    elif isinstance(returns, pd.Series):
        return pd.Series(result)
    else:
        return result