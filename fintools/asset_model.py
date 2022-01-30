import numpy as np
import pandas as pd


# TODO: use random generator with seed
def geometric_brownian_motion(mu, sigma,
                              years, scenarios,
                              initial_price=1.0,
                              steps_per_year=12,
                              prices=True):
    """
     Evolution of stock price using a Geometric Brownian Motion model
     Generates an ensemble of time series of prices according to GBM
    :param mu: the mean drift
    :param sigma: the price volatility
    :param years: number of years to simulate
    :param steps_per_year: Number of periods per year
    :param scenarios: number of sample paths to simulate
    :param initial_price: initial price
    :param prices: return prices if True, returns if False
    :return: A data frame with all the price (or return) sample paths
    """
    dt = 1 / steps_per_year
    n_steps = int(years * steps_per_year)
    rets_plus_1 = np.random.normal(size=(n_steps, scenarios), loc=1 + mu * dt, scale=sigma * np.sqrt(dt))
    # fix the first row
    rets_plus_1[0] = 1
    return initial_price * pd.DataFrame(rets_plus_1).cumprod() if prices else pd.DataFrame(rets_plus_1 - 1)
