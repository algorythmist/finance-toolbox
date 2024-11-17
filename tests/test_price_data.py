import os
import pandas as pd
from datetime import datetime
from fintools.price_data import download_daily_prices, load_daily_prices


def test_load_daily_prices():
    sp500_prices = load_daily_prices('^GSPC', '1970-01-01', '2022-01-01')
    assert len(sp500_prices) == 13118
    #TODO: Fix this test
    # monthly_prices = sp500_prices.resample('ME').last()
    # assert len(monthly_prices) == 52*12


def test_download_daily_prices():
    download_daily_prices('SPY', datetime(2010, 1, 1), datetime(2018, 12, 31))
    quotes = pd.read_csv('SPY.2010-01-01_2018-12-31.csv')
    assert len(quotes) == 2264
    os.remove('SPY.2010-01-01_2018-12-31.csv')


def test_download_prices_with_str_dates():
    download_daily_prices('SPY', '2010-01-01', '2018-12-31')
    quotes = pd.read_csv('SPY.2010-01-01_2018-12-31.csv')
    assert len(quotes) == 2264
    os.remove('SPY.2010-01-01_2018-12-31.csv')


def test_download_prices_with_filename():
    download_daily_prices('SPY', '2010-1-1', '2018-12-31', filename='SPY.csv')
    quotes = pd.read_csv('SPY.csv')
    assert len(quotes) == 2264
    os.remove('SPY.csv')
