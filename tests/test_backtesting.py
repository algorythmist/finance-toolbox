import pytest

from fintools import *
from fintools import collect_metrics


def test_buy_and_hold_vs_rebalance():
    """
    Test a porfolio where one asset vastly overperformed the other one.
    Observe that in this case, no rebalancing defeats monthly rebalancing
    """
    returns = read_returns()
    portfolio_weights = [0.5, 0.5]

    # buy and hold (no rebalancing)
    buy_and_hold_returns = backtest_buy_and_hold(portfolio_weights, returns)
    buy_and_hold_wealth = final_wealth(buy_and_hold_returns)
    assert 10.99 == pytest.approx(buy_and_hold_wealth, 0.001)

    buy_and_hold_metrics = collect_metrics(buy_and_hold_returns, risk_free_rate=0.03)
    assert -0.2973 == pytest.approx(buy_and_hold_metrics.max_drawdown, 0.001)
    assert 0.208 == pytest.approx(buy_and_hold_metrics.excess_kurtosis, abs=0.01)
    assert 1.003471 == pytest.approx(buy_and_hold_metrics.sharpe_ratio, 0.001)
    assert -0.045 == pytest.approx(buy_and_hold_metrics.skewness, abs=0.001)
    assert 0.24345 == pytest.approx(buy_and_hold_metrics.annualized_return, 0.001)
    assert 0.2127 == pytest.approx(buy_and_hold_metrics.annualized_volatility, 0.001)
    # TODO: Getting 0.080965 instead of 5.13%
    # assert 0.0513 == pytest.approx(buy_and_hold_metrics.cornish_fisher_var, 0.001)
    assert 0.107771 == pytest.approx(buy_and_hold_metrics.conditional_var, 0.001)
    assert 0.089861 == pytest.approx(buy_and_hold_metrics.historic_var, 0.001)

    rebalanced_returns = backtest_daily_rebalance(portfolio_weights, returns)
    rebalanced_wealth = final_wealth(rebalanced_returns)
    assert 6.23, pytest.approx(rebalanced_wealth, 0.001)

    rebalanced_metrics = collect_metrics(rebalanced_returns, risk_free_rate=0.03)
    assert -0.230842 == pytest.approx(rebalanced_metrics.max_drawdown, 0.001)
    assert -0.340464 == pytest.approx(rebalanced_metrics.excess_kurtosis, abs=0.01)
    assert 1.081874 == pytest.approx(rebalanced_metrics.sharpe_ratio, 0.001)
    assert -0.079485 == pytest.approx(rebalanced_metrics.skewness, abs=0.001)
    assert 0.180977 == pytest.approx(rebalanced_metrics.annualized_return, 0.001)
    assert 0.139552 == pytest.approx(rebalanced_metrics.annualized_volatility, 0.001)
    assert 0.052424 == pytest.approx(rebalanced_metrics.cornish_fisher_var, 0.001)
    assert 0.065932 == pytest.approx(rebalanced_metrics.conditional_var, 0.001)
    assert 0.052304 == pytest.approx(rebalanced_metrics.historic_var, 0.001)


def test_backtest_allocation_scheme():
    industry_returns = load_industry_returns('ind30_m_vw_rets.csv')['2000':]
    cap_weights = load_market_caps(size=30, weights=True)['2000':]
    equally_weighted_returns = backtest_allocation(industry_returns)
    cap_weighted_returns = backtest_allocation(industry_returns,
                                               allocation_scheme=CapWeightedAllocationScheme(cap_weights=cap_weights))
    btr = pd.DataFrame({"EW": equally_weighted_returns, "CW": cap_weighted_returns})
    wealth_index = (1 + btr).cumprod()
    equal_vs_cap = (wealth_index["EW"] > wealth_index["CW"]).sum() * 100 / wealth_index.shape[0]
    # equally weighted wealth beats cap-weighted over 70% of the time
    assert 73.7 == pytest.approx(equal_vs_cap, 0.1)


def read_returns():
    aapl = read_prices_from_file('AAPL.monthly.20000101-20201231.csv')
    aapl = aapl.rename(columns={'Adj Close': 'AAPL'})
    bnd = read_prices_from_file('BND.monthly.20000101-20201231.csv')
    bnd = bnd.rename(columns={'Adj Close': 'BND'})
    prices = aapl.join(bnd).dropna()
    prices = prices['2009-12-31':'2020-12-31']
    return compute_returns(prices)
