from fintools import *

def test_portfolio_return():
    r = [1, 2, 3]
    w = [3, 2, 1]
    total_return = portfolio_return(w, r)
    assert np.equal(10, total_return)
    assert np.equal(10, portfolio_return([1, 2, 3], [3, 2, 1]))
