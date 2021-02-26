import numpy as np
import pandas as pd

from asset_model import geometric_brownian_motion


def backtest_cppi(risky_returns,
                  safe_returns=None,
                  risk_free_rate=0.03,
                  multiplier=3,
                  cushion_ratio=0.8,
                  drawdown=None,
                  start_value=1000):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    :param risky_returns: history of risky returns
    :param safe_returns: history of safe returns. If None, default to risk free rate
    :param risk_free_rate: Rate of return of the risk free asset
    :param multiplier: multiplier to allocate to risky asset = multiplier*(1-cushion)*wealth
    :param cushion_ratio: ratio of the wealth to protect
    :param drawdown: max drawdown allowed (as ratio)
    :param start_value: Initial monetary value of the account
    """

    # Ensure returns are a data frame
    if isinstance(risky_returns, pd.Series):
        risky_returns = pd.DataFrame(risky_returns, columns=["R"])

    # If no safe asset is specified, default to the risk free rate
    if safe_returns is None:
        safe_returns = pd.DataFrame().reindex_like(risky_returns)
        safe_returns.values[:] = risk_free_rate / 12  # fast way to set all values to a number

    # set up the CPPI parameters
    dates = risky_returns.index
    steps = len(dates)
    account_value = start_value
    floor_value = start_value * cushion_ratio
    peak = account_value

    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_returns)
    risky_w_history = pd.DataFrame().reindex_like(risky_returns)
    cushion_history = pd.DataFrame().reindex_like(risky_returns)
    floorval_history = pd.DataFrame().reindex_like(risky_returns)
    peak_history = pd.DataFrame().reindex_like(risky_returns)

    for step in range(steps):
        # If a drawdown is specified, re-calibrate the floor value
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_weight = multiplier * cushion
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)
        safe_weight = 1 - risky_weight
        risky_alloc = account_value * risky_weight
        safe_alloc = account_value * safe_weight
        # recompute the new account value at the end of this step
        account_value = risky_alloc * (1 + risky_returns.iloc[step]) + safe_alloc * (1 + safe_returns.iloc[step])

        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_weight
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start_value * (1 + risky_returns).cumprod()
    return {
        "wealth": account_history,
        "risky_wealth": risky_wealth,
        "risk_budget": cushion_history,
        "risky_allocation": risky_w_history,
        "peak": peak_history,
        "floor": floorval_history
    }


def cppi_monte_carlo(scenarios=50, mu=0.07, sigma=0.15,
                     multiplier=3, cushion_ratio=0.0, risk_free_rate=0.03, start_value=1000):
    risky_returns = geometric_brownian_motion(scenarios=scenarios, mu=mu, sigma=sigma, prices=False)
    return backtest_cppi(risky_returns=risky_returns, risk_free_rate=risk_free_rate,
                         multiplier=multiplier, start_value=start_value, cushion_ratio=cushion_ratio)
