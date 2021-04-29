import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTOR_DATA_DIR = os.path.join(ROOT_DIR, '../data/factors')


def load_fff_returns_monthly(filename='F-F_Research_Data_5_Factors_2x3_monthly.csv'):
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    returns = pd.read_csv(os.path.join(FACTOR_DATA_DIR, filename),
                          header=0, index_col=0) / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m").to_period('M')
    return returns


def load_fff_returns_daily(filename='F-F_Research_Data_5_Factors_2x3_daily.csv'):
    """
    Load the Fama-French Research Factor Daily Dataset
    """
    returns = pd.read_csv(os.path.join(FACTOR_DATA_DIR, filename),
                          header=0, index_col=0) / 100
    returns.index = pd.to_datetime(returns.index, format="%Y%m%d")
    return returns
