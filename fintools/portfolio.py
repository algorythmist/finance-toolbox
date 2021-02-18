import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from functools import partial

def portfolio_return(weights, returns):
    """
    Compute the portfolio return given an array of returns
    :param weights: the weights of the portfolio
    :param returns: the array of returns corresponding to each asset
    :return: the portfolio return
    """
    return np.array(weights).T @ np.array(returns)


def portfolio_variance(weights, covariance):
    """
    Compute the portfolio variance, given a covariance matrix
    :param weights: the weights of the portfolio
    :param covariance: the covariance matrix
    :return: the portfolio variance
    """
    np_weights = np.array(weights)
    return (np_weights.T @ covariance @ np_weights) ** 0.5





def minimize_volatility(target_return, expected_returns, covariance,
                        debug=False):
    """
    Find the weights that minimize variance at a specific return
    :param target_return: the target return
    :param expected_returns: the vector of expected returns for each asset
    :param covariance: the covariance matrix of the assets
    :param debug: Display optimization details if True
    :return: the optimal weights
    """
    n = len(expected_returns)
    initial_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # make n copies of the (0,1) tuple

    # constraints
    return_equals_to_target = {
        'type': 'eq',
        'fun': lambda w: target_return - portfolio_return(w, expected_returns)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    }
    solution = minimize(fun=portfolio_variance,
                        method='SLSQP',
                        x0=initial_guess,
                        bounds=bounds,
                        constraints=(return_equals_to_target, weights_sum_to_1),
                        args=(covariance,),
                        options={'disp': debug})
    return solution.x


def sharpe_ratio(weights, expected_returns, covariance, risk_free_rate = 0):
    excess_return = portfolio_return(weights, expected_returns) - risk_free_rate
    volatility = portfolio_variance(weights, covariance)
    return -excess_return / volatility


def maximize_sharpe_ratio(expected_returns, covariance,
                          risk_free_rate=0,
                          targeted_annual_return=None,
                          debug=False):
    """
    Find the portfolio weights that maximize the sharpe ration of excess returns
    :param expected_returns: The vector of expected returns
    :param covariance: The covariance matrix of assets
    :param risk_free_rate: the risk free rate (default 0)
    :param targeted_annual_return: Optional target rate of return
    :param debug: Display optimization details if True
    :return: the optimal weights
    """

    sharpe = partial(sharpe_ratio, expected_returns = expected_returns, covariance = covariance, risk_free_rate = risk_free_rate)
    n = len(expected_returns)
    guess = np.repeat(1 / n, n)
    constraints = []
    fully_invested_constraint = LinearConstraint(np.ones(n), [1], [1])
    constraints.append(fully_invested_constraint)
    if targeted_annual_return:
        constraints.append(LinearConstraint(expected_returns, [targeted_annual_return], [targeted_annual_return]))
    bounds = Bounds(np.zeros(n), np.ones(n))
    solution = minimize(sharpe,
                        method='SLSQP',
                        x0=guess,
                        constraints=constraints,
                        bounds=bounds,
                        options={'disp': debug})
    return solution.x


class Portfolio:

    def __init__(self, weights, symbols):
        self.symbols = symbols
        self.weights = weights
        self.__lookup = {symbols[i]: weights[i] for i in range(len(weights))}

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, symbol):
        return self.__lookup[symbol]

    def __iter__(self):
        for i in range(len(self.weights)):
            yield self.symbols[i], self.weights[i]

    def portfolio_return(self, returns):
        return portfolio_return(self.weights, returns)

    def portfolio_variance(self, covariance):
        return portfolio_variance(self.weights, covariance)

    def enc(self):
        """
        Effective Number of Constituents is defined as 1/sum(weights^2)
        :return: the ENC
        """
        return 1 / np.sum(np.square(self.weights))
