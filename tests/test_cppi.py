import pytest
from fintools import *


def test_cppi_no_drawdown():
    sp500_prices = read_prices_from_file(filename='SP500_monthly.csv')
    sp500_returns = compute_returns(sp500_prices[['Close']])
    result = backtest_cppi(sp500_returns)
    print(result.keys())
    floor = result['floor']
    
