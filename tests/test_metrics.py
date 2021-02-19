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

def test_conditional_value_at_risk():
    var = conditional_VaR(hfi)
    assert 0.041264 == pytest.approx(var['CTA Global'], 0.001)