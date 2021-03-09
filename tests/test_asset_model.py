from fintools import *

# TODO: come up with better tests


def test_asset_model_prices():
    prices = geometric_brownian_motion(mu=0.2, sigma=0.1, years=10, scenarios=1000)
    assert (120, 1000) == prices.shape


def test_asset_model_returns():
    returns = geometric_brownian_motion(mu=0.2, sigma=0.1, years=11, scenarios=2000, prices=False)
    assert (132, 2000) == returns.shape
    assert (132, 1) == (returns[[1000]].shape)
