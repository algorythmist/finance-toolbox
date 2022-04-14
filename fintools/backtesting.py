from fintools.portfolio import compute_portfolio_return
from fintools.allocation_scheme import *


def backtest_buy_and_hold(portfolio_weights, returns):
    """
    Buy and Hold
    For this portfolio each period the allocation change according to the wealth
    of each position:
    a_i(t) = a_i(t-1)*(1+r_i(t-1))/(1+r_p(t-1))
    where:
        a_i(t): Allocation to position i at time t
        r_i(t): Return of position i at time t
        r_p(t): Total portfolio return at time t
    :param portfolio_weights: The vector of portfolio weights
    :param returns: Series or Data Frame of returns
    :return: the sequence of portfolio returns
    """
    portfolio_returns = (returns + 1).cumprod()
    portfolio_returns = portfolio_returns.apply(
        lambda row: compute_portfolio_return(weights=portfolio_weights, returns=row),
        axis=1)
    portfolio_returns = portfolio_returns.pct_change()
    # The first return is simply the weighted average of the initial returns
    portfolio_returns.iloc[0] = compute_portfolio_return(weights=portfolio_weights, returns=returns.iloc[0])
    return portfolio_returns


def backtest_daily_rebalance(portfolio_weights, returns):
    """
    Rebalance at every time interval where prices are available
    :param portfolio_weights: The vector of portfolio weights
    :param returns: Series or Data Frame of returns
    :return: the sequence of portfolio returns
    """
    return returns.dot(portfolio_weights)


def final_wealth(portfolio_returns, initial_investment=1.0):
    return initial_investment * (1 + portfolio_returns).prod()


def backtest_allocation(returns, estimation_window=60,
                        allocation_scheme: AllocationScheme = EquallyWeightedAllocationScheme()):
    """
    Backtests a given allocation scheme, given some parameters:
    returns : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = returns.shape[0]
    # return windows
    windows = [(start, start + estimation_window) for start in range(n_periods - estimation_window)]
    weights = [allocation_scheme.get_allocation(returns.iloc[win[0]:win[1]]) for win in windows]
    # convert List of weights to DataFrame
    weights = pd.DataFrame(weights, index=returns.iloc[estimation_window:].index, columns=returns.columns)
    returns = (weights * returns).sum(axis="columns", min_count=1)
    return returns


