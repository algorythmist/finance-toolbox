import numpy as np
import pytest

from fintools.calculator import resample_returns
from fintools.style import *
from fintools.industry_data import *
from fintools.factor_data import *


def test_style():
    np.random.seed(999)
    ind = load_industry_data('ind30_m_vw_rets.csv')['2000':]
    mgr_r = 0.3 * ind["Beer"] + .5 * ind["Smoke"] + 0.2 * np.random.normal(scale=0.15 / (12 ** .5),
                                                                           size=ind.shape[0])
    weights = style_analysis(mgr_r, ind)*100
    assert weights['Beer'] == pytest.approx(32.14, 4)
    assert weights['Smoke'] == pytest.approx(48.59, 4)


def test_style_ff():
    start_index = '1990-01'
    end_index = '2012-05'

    # dependent variable: Berkshire Hathaway returns
    brka_d = load_prices('brka_d_ret.csv')
    brka_m = resample_returns(brka_d, 'M')[start_index:end_index]

    # explanatory variable: Some FF factors
    fff_return = load_fff_returns_monthly()[start_index:end_index]
    fff = fff_return[['Mkt-RF', 'HML', 'SMB', 'RMW', 'CMA']]

    weights = style_analysis(brka_m, fff)*100
    print(weights)
    assert weights['SMB'] == pytest.approx(0, 4)
    assert weights['CMA'] == pytest.approx(0, 4)
    assert weights['HML'] == pytest.approx(27.75, 4)
    assert weights['RMW'] == pytest.approx(18.09, 4)
    assert weights['Mkt-RF'] == pytest.approx(54.15, 4)

