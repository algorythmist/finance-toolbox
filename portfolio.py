import numpy as np


def portfolio_return(weights, returns):
    """
    Compute the portfolio return
    :param weights: the weights of the portfolio
    :param returns: the array of returns corresponding to each asset
    :return: the return
    """
    return np.array(weights).T @ np.array(returns)
