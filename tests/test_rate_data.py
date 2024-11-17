from fintools.rate_data import TBILL_5_YEAR, TBILL_FRED_1_YEAR
from datetime import datetime
from fintools.rate_data import load_yahoo_tbill, load_fred_tbill


def test_yahoo():
    """
    Test Yahoo Finance T-bill data
    """
    from_date = datetime(2020, 1, 1)
    to_date = datetime(2021, 1, 1)
    tbill = load_yahoo_tbill(TBILL_5_YEAR, from_date, to_date)
    assert len(tbill) == 253

def test_fred():
    """
    Test FRED T-bill data
    """
    from_date = datetime(2020, 1, 1)
    to_date = datetime(2021, 1, 1)
    tbill = load_fred_tbill(TBILL_FRED_1_YEAR, from_date, to_date)
    assert len(tbill) == 263