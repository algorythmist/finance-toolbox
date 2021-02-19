from fintools import *


def test_is_normal():
    hfi = load_hfi_returns()
    results = hfi.aggregate(is_normal)
    # show all False but one
    for i in range(len(results)):
        if i == 1:
            assert results.iloc[i]
        else:
            assert not results.iloc[i]