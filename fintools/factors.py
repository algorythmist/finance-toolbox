from enum import Enum

import statsmodels.api as sm

from fintools.factor_data import *


class RegressionType(Enum):
    CAPM = 1,
    THREE_FACTOR = 3,
    FIVE_FACTOR = 5


def french_fama_regression(portfolio_returns, start_index, end_index,
                           regression_type: RegressionType):
    """
    Regress a series of returns on the FFF parameters
    RegressionType.CAPM: Just use excess market return and alpha
    RegressionType.THREE_FACTOR: Use the CAPM factors plus High Minus Low (HML) and Small minus Big (SMB)
    RegressionType.FIVE_FACTOR: Use the three factors plus RMW and CMA
    """
    fff = load_fff_returns_monthly()[start_index:end_index]
    excess_returns = portfolio_returns[start_index:end_index] - fff[['RF']].values
    excess_market = fff[['Mkt-RF']].copy()
    exp_var = excess_market
    exp_var['Alpha'] = 1
    if regression_type is RegressionType.THREE_FACTOR or regression_type is RegressionType.FIVE_FACTOR:
        exp_var['HML'] = fff[['HML']]
        exp_var['SMB'] = fff[['SMB']]
    if regression_type is RegressionType.FIVE_FACTOR:
        exp_var['RMW'] = fff[['RMW']]
        exp_var['CMA'] = fff[['CMA']]
    return sm.OLS(excess_returns, exp_var).fit()
