import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INDUSTRY_DATA_DIR = os.path.join(ROOT_DIR, '../data/industry')
PRICE_DATA_DIR = os.path.join(ROOT_DIR, '../data/prices')


def load_industry_data(filename):
    data = pd.read_csv(os.path.join(INDUSTRY_DATA_DIR, filename),
                       header=0, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index, format='%Y%m').to_period('M')
    data.columns = data.columns.str.strip()
    return data


def load_industry_returns(filename='ind30_m_vw_rets.csv'):
    return load_industry_data(filename) / 100


def load_firms(filename='ind30_m_nfirms.csv'):
    return load_industry_data(filename)


def load_sizes(filename='ind30_m_size.csv'):
    return load_industry_data(filename)


def load_market_caps(size=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    firms_filename = f'ind{size}_m_nfirms.csv'
    ind_nfirms = load_firms(firms_filename)
    size_filename = f'ind{size}_m_size.csv'
    ind_size = load_sizes(size_filename)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap


def load_small_large_cap_returns():
    returns = pd.read_csv(
        os.path.join(INDUSTRY_DATA_DIR, 'Portfolios_Formed_on_ME_monthly_EW.csv'),
        header=0,
        index_col=0,
        parse_dates=True,
        na_values=-99.99)
    # choose subset of columns
    columns = ['Lo 10', 'Hi 10']
    # divide percentages by 100 to get actual returns
    returns = returns[columns] / 100
    # give more meaningful column names
    returns.columns = ['SmallCap', 'LargeCap']
    # convert index to datetime and set the period to month
    returns.index = pd.to_datetime(returns.index, format='%Y%m').to_period('M')
    return returns


def load_hfi_returns():
    hfi = pd.read_csv(
        os.path.join(INDUSTRY_DATA_DIR, 'edhec-hedgefundindices.csv'),
        header=0, index_col=0, parse_dates=True)
    hfi /= 100
    hfi.index = hfi.index.to_period('M')
    return hfi


def read_prices(filename):
    return pd.read_csv(os.path.join(PRICE_DATA_DIR, filename),
                       index_col="Date",
                       parse_dates=True,
                       na_values=['nan'])
