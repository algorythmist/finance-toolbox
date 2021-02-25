from fintools import *
import pytest

hfi = load_hfi_returns()


def test_semi_deviation():
    sd = semi_deviation(hfi)
    assert 0.012443 == pytest.approx(sd['CTA Global'], 0.001)


def test_historic_VaR():
    var = historic_VaR(hfi)
    assert 0.03169 == pytest.approx(var['CTA Global'], 0.001)


def test_parametric_VaR():
    var = parametric_VaR(hfi)
    assert 0.033094 == pytest.approx(var['CTA Global'], 0.001)


def test_conditional_VaR():
    var = conditional_VaR(hfi)
    assert 0.041264 == pytest.approx(var['CTA Global'], 0.001)


def test_compute_drawdown():
    returns = load_small_large_cap_returns()
    small_cap_drawdown = compute_drawdown(returns['SmallCap'])
    large_cap_drawdown = compute_drawdown(returns['LargeCap'])
    assert (1110, 3) == small_cap_drawdown.as_data_frame().shape
    assert small_cap_drawdown.drawdowns is not None
    assert small_cap_drawdown.wealth is not None
    assert small_cap_drawdown.peaks is not None
    assert -0.8400 == pytest.approx(large_cap_drawdown.max_drawdown, 0.001)
    assert -0.8330 == pytest.approx(small_cap_drawdown.max_drawdown, 0.001)
    assert pd.Period('1932-05', 'M') == large_cap_drawdown.max_drawdown_index
    assert pd.Period('1932-05', 'M') == small_cap_drawdown.max_drawdown_index
