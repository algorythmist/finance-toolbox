import numpy as np
import pytest

from fintools import *


def test_skewness_normal():
    samples = np.random.normal(0, 0.15, size=(100000, 1))
    assert 0 == pytest.approx(skewness(samples), abs=0.1)


def test_kurtosis_normal():
    samples = np.random.normal(0, 0.15, size=(100000, 1))
    assert 3 == pytest.approx(kurtosis(samples), abs=0.1)


def test_is_normal():
    hfi = load_hfi_returns()
    results = hfi.aggregate(is_normal)
    # show all False but one
    for i in range(len(results)):
        if i == 1:
            assert results.iloc[i]
        else:
            assert not results.iloc[i]