import pytest
import os
import pandas as pd
from datetime import datetime
from fintools.price_data import download_daily_prices


def test_download_daily_prices():
    download_daily_prices('SPY', datetime(2010, 1, 1), datetime(2018, 12, 31))
    quotes = pd.read_csv('SPY.2010-01-01_2018-12-31.csv')
    assert 2264 == len(quotes)
    os.remove('SPY.2010-01-01_2018-12-31.csv')


def test_download_prices_with_str_dates():
    download_daily_prices('SPY', '2010-01-01', '2018-12-31')
    quotes = pd.read_csv('SPY.2010-01-01_2018-12-31.csv')
    assert 2264 == len(quotes)
    os.remove('SPY.2010-01-01_2018-12-31.csv')


def test_download_prices_with_filename():
    download_daily_prices('SPY', '2010-1-1', '2018-12-31', filename='SPY.csv')
    quotes = pd.read_csv('SPY.csv')
    assert 2264 == len(quotes)
    os.remove('SPY.csv')
