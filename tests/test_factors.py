import pytest

from fintools.calculator import resample_returns
from fintools.factors import french_fama_regression, RegressionType
from fintools.industry_data import load_prices


def test_factors():
    brka_d = load_prices('brka_d_ret.csv')
    brka_m = resample_returns(brka_d, 'M')
    capm_lm = french_fama_regression(brka_m, '1990-01', '2012-05',
                                     regression_type=RegressionType.CAPM)
    params = capm_lm.params

    assert 2 == len(params)
    assert 0.006134 == pytest.approx(params['Alpha'], 0.0001)
    assert 0.540066 == pytest.approx(params['Mkt-RF'], 0.0001)

    ff_lm = french_fama_regression(brka_m, '1990-01', '2012-05',
                                   regression_type=RegressionType.THREE_FACTOR)
    params = ff_lm.params
    assert 4 == len(params)
    assert 0.005226 == pytest.approx(params['Alpha'], 0.0001)
    assert 0.678344 == pytest.approx(params['Mkt-RF'], 0.0001)
    assert 0.504416 == pytest.approx(params['HML'], 0.0001)
    assert -0.469505 == pytest.approx(params['SMB'], 0.0001)

