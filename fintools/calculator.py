TRADING_DAYS_IN_YEAR = 252
MONTHS_IN_YEAR = 12
QUARTERS_IN_YEAR = 4


def compute_returns(prices):
    return prices.pct_change().dropna()


def compute_compound_return(returns):
    return (returns + 1).prod() - 1


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
    TODO: infer the periods per year
    """
    return (returns + 1).prod() ** (periods_in_year / len(returns)) - 1