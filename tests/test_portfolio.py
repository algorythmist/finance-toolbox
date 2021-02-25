import pytest

from fintools import *

industry_returns = load_industry_data('ind30_m_vw_rets.csv') / 100


def test_portfolio_return():
    r = [1, 2, 3]
    w = [3, 2, 1]
    total_return = compute_portfolio_return(w, r)
    assert np.equal(10, total_return)
    assert np.equal(10, np.array(total_return))


def test_portfolio_variance():
    w = [0.5, 0.5]
    cov = [[2.5, 3.0],
           [3.0, 10.0]]
    volatility = compute_portfolio_variance(w, cov)
    assert 2.150 == pytest.approx(volatility, 0.001)


def test_minimize_volatility():
    assets = ['Games', 'Fin']
    returns = industry_returns["1996":"2000"][assets]
    expected_returns = annualize_returns(returns, 12)
    covariance = returns.cov()
    target_return = 0.15
    w = minimize_volatility(target_return, expected_returns, covariance)
    assert 0.47287631 == pytest.approx(w[0], 0.0001)
    assert 1 == w.sum()
    vol = compute_portfolio_variance(w, covariance)
    assert 0.056163 == pytest.approx(vol, 0.0001)


def test_maximize_sharpe_ratio():
    assets = ['Games', 'Smoke', 'Beer', 'Food']
    returns = industry_returns["1996":"2000"][assets]
    expected_returns = annualize_returns(returns, 12)
    covariance = returns.cov()
    risk_free_rate = 0.01
    w = maximize_sharpe_ratio(expected_returns, covariance,
                              risk_free_rate=risk_free_rate)
    assert 1.0 == pytest.approx(w.sum(), 0.000001)
    expected = [0.11430192, 0.06604457, 0.22653882, 0.59311468]
    assert np.all([expected[i] == pytest.approx(w[i], 0.00001) for i in range(0, len(expected))])


def test_global_minimum_variance_portfolio():
    assets = ['Games', 'Smoke', 'Beer', 'Food']
    returns = industry_returns["1996":"2000"][assets]
    # expected_returns = annualize_returns(returns, 12)
    covariance = returns.cov()
    # risk_free_rate = 0.01
    w = global_minimum_variance_portfolio(covariance)
    assert 1 == w.sum()
    expected = [0.37456, 0.07577, 0.0, 0.54966]
    assert np.all([expected[i] == pytest.approx(w[i], 0.0001) for i in range(0, len(expected))])


def test_portfolio_class():
    symbols = ['US', 'IN', 'BD']
    weights = [.5, .3, .2]
    p = Portfolio(weights, symbols)
    assert 3 == len(p)
    assert .3 == p['IN']
    assert np.array_equal(['US', 'IN', 'BD'], p.symbols)

    returns = [0.1, -0.1, 0.2]
    pr = p.portfolio_return(returns)
    assert 0.06 == pytest.approx(pr, 5)


def test_portfolio_enc():
    symbols = ['A', 'B']
    weights = [.5, .5]
    p = Portfolio(weights, symbols)
    assert 2 == p.enc()

    weights = [1, 0]
    p = Portfolio(weights, symbols)
    assert 1 == p.enc()

    weights = [0.8, 0.2]
    p = Portfolio(weights, symbols)
    assert 1.471 == pytest.approx(p.enc(), 0.001)

    symbols = ['A', 'B', 'C', 'D']
    weights = [.25, .25, .25, .25]
    p = Portfolio(weights, symbols)
    assert 4 == p.enc()

    weights = [.5, .3, .1, .1]
    p = Portfolio(weights, symbols)
    assert 2.777777, pytest.approx(p.enc(), 0.0001)
