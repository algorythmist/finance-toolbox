from fintools import *


def test_load_fff_returns():
    df = load_fff_returns_daily()
    assert (14514, 6) == df.shape
    df = load_fff_returns_monthly()
    assert (692, 6) == df.shape