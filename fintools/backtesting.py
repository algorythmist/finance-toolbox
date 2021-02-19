from fintools.calculator import *
from fintools.metrics import *
from fintools.portfolio import portfolio_return


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
    portfolio_returns = portfolio_returns.apply(lambda row: portfolio_return(weights=portfolio_weights, returns=row),
                                                axis=1)
    portfolio_returns = portfolio_returns.pct_change()
    # The first return is simply the weighted average of the initial returns
    portfolio_returns.iloc[0] = portfolio_return(weights=portfolio_weights, returns=returns.iloc[0])
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


def collect_metrics(returns, risk_free_rate=0.0):
    """
    Return a DataFrame that contains aggregated summary stats for the returns
    :param: returns: A vector or Data Frame of returns
    :param: risk_free_rate: The risk free rate (constant)
    """
    annualized_return = returns.aggregate(annualize_returns, periods_in_year=12)
    annualized_volatility = returns.aggregate(annualize_volatility, periods_in_year=12)
    annualized_sharpe = returns.aggregate(annualized_sharpe_ratio, risk_free_rate=risk_free_rate, periods_in_year=12)
    dd = returns.agg(lambda r: drawdown(r).max_drawdown)
    skewness = returns.skew()
    kurt = returns.kurt()
    cf_var5 = parametric_VaR(returns, confidence_level=5)
    hist_var5 = historic_VaR(returns, confidence_level=5)
    conditional_var5 = conditional_VaR(returns, confidence_level=5)

    result = {
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
