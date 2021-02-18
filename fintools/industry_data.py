import os

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INDUSTRY_DATA_DIR = os.path.join(ROOT_DIR, '..\data\industry')


def load_industry_data(filename):
    data = pd.read_csv(os.path.join(INDUSTRY_DATA_DIR, filename),
                       header=0, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index, format='%Y%m').to_period('M')
    data.columns = data.columns.str.strip()
    return data
