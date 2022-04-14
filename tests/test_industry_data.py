from fintools import *


def test_load_industry_data():
    df = load_industry_data('ind30_m_vw_rets.csv')
    assert (1110, 30) == df.shape
    df.head()