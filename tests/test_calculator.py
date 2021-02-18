import pytest

from fintools import *


def test_compute_returns():
    prices = pd.DataFrame({"BLUE": [8.70, 8.91, 8.71, 8.43, 8.73],
                           "ORANGE": [10.66, 11.08, 10.71, 11.50, 12.11]})
    returns = compute_returns(prices)
    assert 4 == len(returns)

    compound = compute_compound_return(returns)
    assert 0.003448 ==  pytest.approx(compound['BLUE'], 0.0001)
    assert 0.136023 == pytest.approx(compound['ORANGE'], 0.0001)


def test_annualized_return():
    r_monthly = 0.01
    assert 0.1268 == pytest.approx(annualized_monthly_return(r_monthly), 0.001)
    r_daily = 0.0001
    assert 0.02552 == pytest.approx(annualized_daily_return(r_daily), 0.0001)
