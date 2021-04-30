import numpy as np
import pandas as pd


class AllocationScheme:

    def get_allocation(self, returns):
        """
        Provide an allocation scheme for a collection of return vectors
        """
        pass


class EquallyWeightedAllocationScheme(AllocationScheme):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted
    """

    def __init__(self, cap_weights=None, microcap_threshold=None, max_cap_weight_multiplier=None):
        self.cap_weights = cap_weights
        self.microcap_threshold = microcap_threshold
        self.max_cap_weight_multiplier = max_cap_weight_multiplier

    def get_allocation(self, returns):
        n = len(returns.columns)
        allocation = pd.Series(1 / n, index=returns.columns)
        if self.cap_weights is not None:
            cw = self.cap_weights.loc[returns.index[0]]  # starting cap weight
            # exclude microcaps
            if self.microcap_threshold is not None and self.microcap_threshold > 0:
                microcap = cw < self.microcap_threshold
                allocation[microcap] = 0
                allocation = allocation / allocation.sum()  # reweight
            # limit weight to a multiple of capweight
            if self.max_cap_weight_multiplier is not None and self.max_cap_weight_multiplier > 0:
                allocation = np.minimum(allocation, cw * self.max_cap_weight_multiplier)
                allocation = allocation / allocation.sum()  # reweight
        return allocation


class CapWeightedAllocationScheme(AllocationScheme):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """

    def __init__(self, cap_weights):
        self.cap_weights = cap_weights

    def get_allocation(self, returns):
        w = self.cap_weights.loc[returns.index[1]]
        return w / w.sum()
