"""
Sources for interest rates etc
"""

import pandas_datareader.data as web
import yfinance as yf

TBILL_13_WEEK = "^IRX"
TBILL_5_YEAR = "^FVX"
TBILL_10_YEAR = "^TNX"

TBILL_FRED_4_WEEK = "TB4WK"
TBILL_FRED_3_MONTH = "IR3TIB01USM156N"
TBILL_FRED_6_MONTH = "DTB6MO"
TBILL_FRED_1_YEAR = "DTB1YR"

def load_yahoo_tbill(symbol, from_date, to_date):
    """
    Load T-bill data from Yahoo Finance
    """
    df = yf.download(symbol, from_date, to_date)/100.0
    return df.xs(key=symbol, axis=1, level=1)  # Use .xs to select by level

def load_fred_tbill(symbol, from_date, to_date):
    """
    Load T-bill data from FRED
    """
    df = web.DataReader(symbol, 'fred', from_date, to_date)/100
    return df[symbol]