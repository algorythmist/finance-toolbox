from fintools.rate_data import TBILL_5_YEAR, TBILL_FRED_1_YEAR
from datetime import datetime
from fintools.rate_data import load_yahoo_tbill, load_fred_tbill
import pytest


def test():
    import yfinance as yf
    df = yf.download("^FVX", '2020-01-01', '2021-01-01')

def test_yahoo():
    """
    Test Yahoo Finance T-bill data
    """
    from_date = '2020-01-01'
    to_date = '2021-01-01'
    tbill = load_yahoo_tbill(TBILL_5_YEAR, from_date, to_date)
    assert len(tbill) == 253
    v = tbill['Adj Close']['2020-02-03']
    assert v == pytest.approx(0.01343)

def test_fred():
    """
    Test FRED T-bill data
    """
    from_date = datetime(2020, 1, 1)
    to_date = datetime(2021, 1, 1)
    tbill = load_fred_tbill(TBILL_FRED_1_YEAR, from_date, to_date)
    assert len(tbill) == 263
    v = tbill['2020-02-03']
    assert v == pytest.approx(0.0142)