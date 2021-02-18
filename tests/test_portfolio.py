import pytest

from fintools import *


def test_portfolio_return():
    r = [1, 2, 3]
    w = [3, 2, 1]
    total_return = portfolio_return(w, r)
    assert np.equal(10, total_return)
    assert np.equal(10, np.array(total_return))


def test_portfolio_volatility():
    w = [0.5, 0.5]
    cov = [[2.5, 3.0],
           [3.0, 10.0]]
    volatility = portfolio_volatility(w, cov)
    assert 2.150 == pytest.approx(volatility, 0.001)
