import numpy as np


def portfolio_return(weights, returns):
    """
    Compute the portfolio return given an array of returns
    :param weights: the weights of the portfolio
    :param returns: the array of returns corresponding to each asset
    :return: the portfolio return
    """
    return np.array(weights).T @ np.array(returns)


def portfolio_volatility(weights, covariance):
    """
    Compute the portfolio volatility, given a covariance matrix
    :param weights: the weights of the portfolio
    :param covariance: the covariance matrix
    :return: the portfolio volatility (variance)
    """
    np_weights = np.array(weights)
    return (np_weights.T @ covariance @ np_weights) ** 0.5