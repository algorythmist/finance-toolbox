import numpy as np
import pandas as pd


def funding_ratio(assets, liabilities, risk_free_rate):
    """ assets over liabilities """
    return assets/present_value(liabilities, risk_free_rate)


def present_value(liabilities: pd.Series, risk_free_rate):
    """
    Computes the present value of a sequence of liabilities
    :param liabilities a Series sequence indexed by date
    :param risk_free_rate risk free rate per period
    :return: the present value of the sequence
    """
    dates = liabilities.index
    discounts = discount(risk_free_rate, dates)
    return (discounts * liabilities).sum()


def discount(risk_free_rate_per_period, periods):
    """
    Compute the value of a pure discount bond that pays a dollar in the future
    :param risk_free_rate_per_period: the risk free rate
    :param periods: the number of periods in the future
    :return: The discount on a nominal value of $1
    """
    return (1 + risk_free_rate_per_period) ** (-periods)


def calculate_future_value(present_value, annual_rate, years,
                           compounding_periods_per_year=1,
                           continuous_compounding=False):
    """
    Calculate the future value V_n of an fixed income investment worth V_0:
    V_n = V_0 * (1+r)^n
    :param present_value: the current value V_0
    :param annual_rate: the annual rate of return
    :param years: years ahead
    :param compounding_periods_per_year: periods per year when interest is paid
    :param continuous_compounding: set to True if compounding continuously
    :return:
    """
    if continuous_compounding:
        return present_value * np.exp(annual_rate * years)
    periods = years * compounding_periods_per_year
    period_rate = annual_rate / compounding_periods_per_year
    return present_value * ((1 + period_rate) ** periods)
