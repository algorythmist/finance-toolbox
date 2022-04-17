import numpy as np
import timeit
from fintools.industry_data import load_industry_returns


def compound1(r):
    return (1+r).prod()-1


def compound2(r):
    return np.expm1(np.log1p(r).sum())


def test_efficiency():
    """
    The claim that fancy compounding is more efficient is not verified by this test
    :return:
    """
    ind_returns = load_industry_returns()

    def compound_prod():
        return compound1(ind_returns)

    def compound_sum():
        return compound2(ind_returns)

    time1 = timeit.timeit(stmt=compound_prod, number=1000)
    print(time1)

    time2 = timeit.timeit(stmt=compound_sum, number=1000)
    print(time2)


