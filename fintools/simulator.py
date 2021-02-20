from abc import ABC, abstractmethod

from fintools import *


class InvestmentStrategy(ABC):

    @abstractmethod
    def update_portfolio_weighs(self, account_value, returns):
        pass


class NoRebalanceInvestmentStrategy(InvestmentStrategy):

    def __init__(self, initial_weights):
        self.__previous_weights = initial_weights

    def update_portfolio_weighs(self, account_value, returns):
        denominator = self.__previous_weights @ (1+returns)
        new_weights =  np.multiply(self.__previous_weights, (1+returns))/denominator
        self.__previous_weights = new_weights
        return new_weights


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
                 start_value):
        dates = returns.index
        steps = len(dates)
        portfolio_weights = initial_portfolio_weights
        account_value = start_value
        stats = Stats(returns)
        for step in range(steps):
            asset_returns = returns.iloc[step]
            portfolio_return = compute_portfolio_return(returns=asset_returns, weights=portfolio_weights)
            account_value = (1 + portfolio_return) * account_value
            # update stats
            stats.update(step, portfolio_return, account_value)
            # update weights
            if self.__investment_strategy:
                portfolio_weights = self.__investment_strategy.update_portfolio_weighs(account_value, asset_returns)

        return stats
