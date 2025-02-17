import os
import pandas as pd
import yfinance as yf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DATA_DIR = os.path.join(ROOT_DIR, '../data/prices')

def load_daily_prices(symbols, from_date, to_date):
    """
    Load daily prices from Yahoo Finance
    :param symbols: the tickers
    :param from_date: price history start date
    :param to_date: price history end date
    :return: a data frame with all the price columns
    """
    data = yf.download(symbols, start=from_date, end=to_date)
    return data


def download_daily_prices(symbol, from_date, to_date, filename=None):
    """
    Load prices from yahoo and store them in CSV
    :param symbol:
    :param from_date:
    :param to_date:
    :param filename:
    :return:
    """
    prices = load_daily_prices(symbol, from_date, to_date)
    if filename is None:
        filename = "{}.{}_{}.csv".format(symbol, _to_str(from_date), _to_str(to_date))
    prices.to_csv(filename)


def read_prices_from_file(filename):
    return pd.read_csv(os.path.join(PRICE_DATA_DIR, filename),
                       index_col="Date",
                       parse_dates=True,
                       na_values=['nan'])


def read_prices(symbol, relative_path=PRICE_DATA_DIR, filename=None):
    """
    Read prices from a CSV file names {symbol}.csv
    :param symbol: symbol to load
    :param relative_path: directory where price file is relative to project root
    :param filename specific filename overrides default
    :return: a DataFrame of prices
    """
    if filename is None:
        filename = '{}.csv'.format(symbol)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, relative_path + filename)
    return pd.read_csv(path,
                       index_col="Date",
                       parse_dates=True,
                       infer_datetime_format=True,
                       na_values=['nan'])


def read_all_prices(symbols, relative_path='data/prices/'):
    return {symbol: read_prices(symbol, relative_path) for symbol in symbols}


def _to_str(dt):
    """ Convert date to string """
    return dt if type(dt) is str else dt.strftime("%Y-%m-%d")
