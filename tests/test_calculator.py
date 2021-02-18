import pytest

from fintools import *


def test_compute_returns():
    prices = pd.DataFrame({"BLUE": [8.70, 8.91, 8.71, 8.43, 8.73],
                           "ORANGE": [10.66, 11.08, 10.71, 11.50, 12.11]})
    returns = compute_returns(prices)
    assert 4 == len(returns)

    compound = compute_compound_return(returns)
    assert 0.003448 == pytest.approx(compound['BLUE'], 0.0001)
    assert 0.136023 == pytest.approx(compound['ORANGE'], 0.0001)


def test_annualized_return():
    r_monthly = 0.01
    assert 0.1268 == pytest.approx(annualized_monthly_return(r_monthly), 0.001)
    r_daily = 0.0001
    assert 0.02552 == pytest.approx(annualized_daily_return(r_daily), 0.0001)


def test_annualized():
    returns = load_small_large_cap_returns()

    annualized_vol = annualize_volatility(returns, 12)
    assert 0.368193, pytest.approx(annualized_vol['SmallCap'], 0.0001)
    assert 0.186716, pytest.approx(annualized_vol['LargeCap'], 0.0001)

    annualized_ret = annualize_returns(returns, 12)
    assert 0.454825 == pytest.approx((annualized_ret / annualized_vol)['SmallCap'], 0.0001)
    assert 0.497063 == pytest.approx((annualized_ret / annualized_vol)['LargeCap'], 0.0001)

    sharpe_ratio = annualized_sharpe_ratio(returns, 0.03, 12)
    assert 0.373346 == pytest.approx(sharpe_ratio['SmallCap'], 0.0001)
    assert 0.336392 == pytest.approx(sharpe_ratio['LargeCap'], 0.0001)


def test_annual_to_daily_rate():
    daily_rate = annual_to_daily_rate(0.20)
    assert 0.0007237 == pytest.approx(daily_rate, 0.0001)
    monthly_rate = subdivide_rate(0.20, 12)
    assert 0.01531 == pytest.approx(monthly_rate, 0.0001)


def test_geometric_return():
    returns = [0.02, 0.08, -0.04]
    assert 0.01882 == pytest.approx(geometric_return(returns), 0.001)
