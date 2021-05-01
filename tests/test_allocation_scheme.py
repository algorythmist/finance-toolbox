from fintools import *
import pytest


def test_equally_weighted():
    returns = load_industry_returns('ind30_m_ew_rets.csv').loc['2000']
    scheme = EquallyWeightedAllocationScheme()
    allocation = scheme.get_allocation(returns)
    assert 30 == len(allocation)
    for a in allocation:
        assert 0.033333 == pytest.approx(a, 0.0001)

    #Specify a cap threshold
    cap_weights = load_market_caps('30', weights=True).loc['2000']
    scheme = EquallyWeightedAllocationScheme(cap_weights=cap_weights, microcap_threshold=0.01)
    allocation = scheme.get_allocation(returns)
    assert 30 == len(allocation)
    # collect non-zero allocations
    survivors = allocation[allocation > 0.0]
    assert 18 == len(survivors)
    for a in survivors:
        assert 0.055555 == pytest.approx(a, 0.0001)


def test_cap_weighted():
    returns = load_industry_returns('ind30_m_ew_rets.csv').loc['2000']
    cap_weights = load_market_caps('30', weights=True).loc['2000']
    scheme = CapWeightedAllocationScheme(cap_weights=cap_weights)
    allocation = scheme.get_allocation(returns)
    assert 30 == len(allocation)
    allocation = allocation.sort_values(ascending=False)
    assert allocation.index[0] == 'BusEq'
    assert allocation.index[1] == 'Fin'
    assert allocation.index[2] == 'Servs'


