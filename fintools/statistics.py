import pandas as pd
import scipy.stats


def is_normal(data: pd.Series, confidence_level=0.001):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if normal hypothesis is accepted, False otherwise
    """
    result = scipy.stats.jarque_bera(data)
    return result.pvalue > confidence_level
