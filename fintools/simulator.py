from abc import ABC, abstractmethod

import pandas as pd

from fintools.calculator import *
from fintools.portfolio import compute_portfolio_return


class InvestmentStrategy(ABC):

    @abstractmethod
    def update_portfolio_weighs(self, current_weights, account_value, returns):
        """
        Update portfolio weights based on an investment strategy
        :param current_weights: current portfolio weights
        :param account_value:current account balance
        :param returns: last period returns
        :return: the adjusted portfolio weights
        """
        pass


class NoRebalanceInvestmentStrategy(InvestmentStrategy):

    def update_portfolio_weighs(self, current_weights, account_value, returns):
        denominator = current_weights @ (1 + returns)
        return np.multiply(current_weights, (1 + returns)) / denominator


class Stats:

    def __init__(self, returns):
        self.account_history = pd.DataFrame().reindex_like(returns)
        self.return_history = pd.DataFrame().reindex_like(returns)

    def update(self, step, preturn, account_value):
        self.account_history.iloc[step] = account_value
        self.return_history.iloc[step] = preturn


class StrategySimulator:

    def __init__(self, investment_strategy: InvestmentStrategy = None):
        self.__investment_strategy = investment_strategy

    def simulate(self,
                 returns,
                 initial_portfolio_weights,
                 initial_balance):
        dates = returns.index
        steps = len(dates)
        portfolio_weights = initial_portfolio_weights
        account_value = initial_balance
        stats = Stats(returns)
        for step in range(steps):
            asset_returns = returns.iloc[step]
            portfolio_return = compute_portfolio_return(returns=asset_returns, weights=portfolio_weights)
            account_value = (1 + portfolio_return) * account_value
            # update stats
            stats.update(step, portfolio_return, account_value)
            # update weights
            if self.__investment_strategy:
                portfolio_weights = self.__investment_strategy \
                    .update_portfolio_weighs(portfolio_weights, account_value, asset_returns)

        return stats
