import numpy as np
from scipy.stats import gmean

TRADING_DAYS_IN_YEAR = 252
MONTHS_IN_YEAR = 12
QUARTERS_IN_YEAR = 4


def compute_returns(prices):
    return prices.pct_change().dropna()


def compute_compound_return(returns):
    """
    Compound a stream of returns
    :param returns: vector of returns
    :return: the compounded return
    """
    # This implementation uses sum instead of prod and is faster than: (returns + 1).prod() - 1
    return np.expm1(np.log1p(returns).sum())


def annualized_return(r, periods_in_year):
    """
    Annualize a periodic return
    :param r: the periods return
    :param periods_in_year: periods in year when return is awarded
    :return: the annualized return
    """
    return (1 + r) ** periods_in_year - 1


def annualized_monthly_return(r):
    return annualized_return(r, MONTHS_IN_YEAR)


def annualized_quarterly_return(r):
    return annualized_return(r, QUARTERS_IN_YEAR)


def annualized_daily_return(r):
    return annualized_return(r, TRADING_DAYS_IN_YEAR)


def annualize_returns(returns, periods_in_year):
    """
    Annualizes a set of returns
    :param returns: the vector of returns
    :param periods_in_year: periods in year that returns are awarded
    """
    return (returns + 1).prod() ** (periods_in_year / len(returns)) - 1


def annualize_volatility(returns, periods_in_year):
    return returns.std()*np.sqrt(periods_in_year)


def annualized_sharpe_ratio(returns, risk_free_rate, periods_in_year):
    annualized_volatility = returns.std() * np.sqrt(periods_in_year)
    annualized_returns = annualize_returns(returns, periods_in_year)
    excess_return = annualized_returns - risk_free_rate
    return excess_return / annualized_volatility


def annual_to_daily_rate(rate, trading_days_in_year=TRADING_DAYS_IN_YEAR):
    """
    Infer daily rate from annual rate
    :param rate: the annual rate of return
    :param trading_days_in_year: optional, trading days in year (default = 252)
    :return: the daily rate
    """
    return subdivide_rate(rate, trading_days_in_year)


def subdivide_rate(rate, periods):
    """
    Subdivide a return rate into smaller periods
    :param rate:
    :param periods:
    :return:
    """
    return (1 + rate) ** (1 / periods) - 1


def geometric_return(returns):
    """
    Compute the geometric return
    :param returns: array-like object with returns
    :return: the geometric return
    """
    return gmean(np.array(returns)+1)-1


def resample_returns(returns, period_type='M'):
    return returns.resample(period_type).apply(compute_compound_return).to_period(period_type)
