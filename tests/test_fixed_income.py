import pytest

from fintools.fixed_income import *


def test_calculate_future_value():
    value = calculate_future_value(1000, 0.03, 5, 1)
    assert 1159.27407 == pytest.approx(value, 0.001)
    value = calculate_future_value(1000, 0.03, 5, 4)
    assert 1161.184142 == pytest.approx(value, 0.001)
    value = calculate_future_value(1000, 0.03, 5, continuous_compounding=True)
    assert 1161.83424 == pytest.approx(value, 0.001)


def test_discount():
    assert .7441 == pytest.approx(discount(0.03, 10), 0.0001)


def test_present_value():
    liabilities = pd.Series(data=[1, 1.5, 2, 2.5], index=[3, 3.5, 4, 4.5])
    assert 6.2333 == pytest.approx(present_value(liabilities, 0.03), 0.0001)


def test_funding_ratio():
    liabilities = pd.Series(data=[1, 1.5, 2, 2.5], index=[3, 3.5, 4, 4.5])
    assets = 5
    assert 0.80214 == pytest.approx(funding_ratio(assets, liabilities, 0.03), 0.0001)
    assert 0.7720 == pytest.approx(funding_ratio(assets, liabilities, 0.02), 0.0001)
    assert 0.8649 == pytest.approx(funding_ratio(assets, liabilities, 0.05), 0.0001)


def test_inst_to_annual():
    assert inst_to_annual(0.5) == pytest.approx(0.6487, 0.001)


def test_annual_to_inst():
    assert annual_to_inst(0.5) == pytest.approx(0.4054, 0.001)
