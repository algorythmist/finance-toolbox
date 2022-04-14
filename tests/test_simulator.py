from fintools import *
import pytest


def test_monthly_rebalance():
    returns = read_returns()
    simulator = StrategySimulator()
    stats = simulator.simulate(returns=returns, initial_portfolio_weights=[0.5, 0.5], initial_balance=1)
    end_value = final_wealth(stats.return_history)
    assert 6.23 == pytest.approx(end_value, 0.001)
    metrics = collect_metrics(stats.return_history, risk_free_rate=0.03)
    assert -0.230842 == pytest.approx(metrics.max_drawdown, 0.001)
    assert -0.340464 == pytest.approx(metrics.excess_kurtosis, abs=0.01)
    assert 1.081874 == pytest.approx(metrics.sharpe_ratio, 0.001)
    assert -0.079485 == pytest.approx(metrics.skewness, abs=0.001)
    assert 0.180977 == pytest.approx(metrics.annualized_return, 0.001)
    assert 0.139552 == pytest.approx(metrics.annualized_volatility, 0.001)
    assert 0.052424 == pytest.approx(metrics.cornish_fisher_var, 0.001)
    assert 0.065932 == pytest.approx(metrics.conditional_var, 0.001)
    assert 0.052304 == pytest.approx(metrics.historic_var, 0.001)


def test_buy_and_hold():
    returns = read_returns()
    initial_weights = [0.5, 0.5]
    simulator = StrategySimulator(investment_strategy=NoRebalanceInvestmentStrategy())
    stats = simulator.simulate(returns=returns, initial_portfolio_weights=initial_weights, initial_balance=1)
    end_value = final_wealth(stats.return_history)
    assert 10.99 == pytest.approx(end_value, 0.001)
    buy_and_hold_metrics = collect_metrics(stats.return_history, risk_free_rate=0.03)
    assert -0.2973 == pytest.approx(buy_and_hold_metrics.max_drawdown, 0.001)
    assert 0.208 == pytest.approx(buy_and_hold_metrics.excess_kurtosis, abs=0.01)
    assert 1.003471 == pytest.approx(buy_and_hold_metrics.sharpe_ratio, 0.001)
    assert -0.045 == pytest.approx(buy_and_hold_metrics.skewness, abs=0.001)
    assert 0.24345 == pytest.approx(buy_and_hold_metrics.annualized_return, 0.001)
    assert 0.2127 == pytest.approx(buy_and_hold_metrics.annualized_volatility, 0.001)
    assert 0.107771 == pytest.approx(buy_and_hold_metrics.conditional_var, 0.001)
    assert 0.089861 == pytest.approx(buy_and_hold_metrics.historic_var, 0.001)


def read_returns():
    aapl = load_prices('AAPL.monthly.20000101-20201231.csv')
    aapl = aapl.rename(columns={'Adj Close': 'AAPL'})
    bnd = load_prices('BND.monthly.20000101-20201231.csv')
    bnd = bnd.rename(columns={'Adj Close': 'BND'})
    prices = aapl.join(bnd).dropna()
    prices = prices['2009-12-31':'2020-12-31']
    return compute_returns(prices)
