from fintools import *
import pytest


def test_buy_and_hold_vs_rebalance():
    """
    Test a porfolio where one asset vastly overperformed the other one.
    Observe that no rebalancing defeats monthly rebalancing
    """
    aapl = read_prices('AAPL.monthly.20000101-20201231.csv')
    aapl = aapl.rename(columns={'Adj Close': 'AAPL'})
    bnd = read_prices('BND.monthly.20000101-20201231.csv')
    bnd = bnd.rename(columns={'Adj Close': 'BND'})
    prices = aapl.join(bnd).dropna()
    prices = prices['2009-12-31':'2020-12-31']
    returns = compute_returns(prices)

    portfolio_weights = [0.5, 0.5]

    # buy and hold (no rebalancing)
    buy_and_hold_returns = backtest_buy_and_hold(portfolio_weights, returns)
    buy_and_hold_wealth = final_wealth(buy_and_hold_returns)
    assert 10.99 == pytest.approx(buy_and_hold_wealth, 0.001)
    buy_and_hold_metrics = collect_metrics(buy_and_hold_returns)
    #print(buy_and_hold_metrics.max_drawdown)  #TODO: -0.2973 vs -.2967
    print(buy_and_hold_metrics.excess_kurtosis)
    #TODO: assert 0.208 == pytest.approx(buy_and_hold_metrics.excess_kurtosis,abs=0.01)
    # self.assertAlmostEqual(1.003471, stats['sharpe_ratio'], 4)
    # self.assertAlmostEqual(-0.04, stats['skewness'], 2)
    # self.assertAlmostEqual(0.24345, stats['annualized_return'], 4)
    # self.assertAlmostEqual(0.2127, stats['annualized_volatility'], 4)