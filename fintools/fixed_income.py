import numpy as np
import pandas as pd


def funding_ratio(assets: float, liabilities: pd.Series, risk_free_rate: float):
    """
    assets over liabilities
    :param assets current value of assets
    :param liabilities a Series of liabilities
    :param risk_free_rate the risk-free rate
    """
    return assets / present_value(liabilities, risk_free_rate)


def present_value(values: pd.Series, risk_free_rate: float):
    """
    Computes the present value of a sequence of liabilities
    :param values a Series sequence indexed by date
    :param risk_free_rate risk free rate per period
    :return: the present value of the sequence
    """
    dates = values.index
    discounts = discount(risk_free_rate, dates)
    return (discounts * values).sum()


def discount(risk_free_rate_per_period: float, periods: float):
    """
    Compute the value of a pure discount bond that pays a dollar in the future
    :param risk_free_rate_per_period: the risk free rate
    :param periods: the number of periods in the future
    :return: The discount on a nominal value of $1
    """
    return (1 + risk_free_rate_per_period) ** (-periods)


def calculate_future_value(present_value,
                           annual_rate,
                           years,
                           compounding_periods_per_year=1,
                           continuous_compounding=False):
    """
    Calculate the future value V_n of a fixed income investment worth V_0:
    V_n = V_0 * (1+r/p)^{np}
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


def calculate_present_value(future_value,
                            annual_rate,
                            years,
                            compounding_periods_per_year=1,
                            continuous_compounding=False):
    """
        Calculate the presnt value V_0 of a fixed income investment worth V_n after n years
        V_0 = V_n / (1+r/p)^{np}
        :param future_value: the future value V_n
        :param annual_rate: the annual rate of return
        :param years: years ahead
        :param compounding_periods_per_year: periods per year when interest is paid
        :param continuous_compounding: set to True if compounding continuously
        :return:
        """
    if continuous_compounding:
        return future_value / np.exp(annual_rate * years)
    periods = years * compounding_periods_per_year
    period_rate = annual_rate / compounding_periods_per_year
    return future_value / ((1 + period_rate) ** periods)


def accrued_interest(rate, number_of_periods):
    return (1 + rate / number_of_periods) ** number_of_periods


def inst_to_annual(rate):
    """
    Instantaneous to annual rate of return with continuous compounding
    :param rate: the instantaneous rate of return
    :return: the annual rate of return
    """
    return np.expm1(rate)


def annual_to_inst(rate):
    """
    Annual to Instantaneous rate of return with continuous compounding
    :param rate: the annual rate of return
    :return: the instantaneous rate of return
    """
    return np.log1p(rate)


def simulate_cir(a, b, sigma,
                 n_years: int = 10,
                 n_scenarios: int = 1,
                 steps_per_year: int = 12,
                 initial_rate: float = None):
    """
    CIR model for interest rates
    :param a: mean reverting rate
    :param b: long term mean of rate
    :param sigma: volatility
    :param n_years: number of years to simulate
    :param n_scenarios: number of scenarios
    :param steps_per_year: time steps per year
    :param initial_rate: initial rate of return
    :return:
    """
    if initial_rate is None:
        initial_rate = b
    # convert to instantaneous rate
    initial_rate = annual_to_inst(initial_rate)
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(n_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = initial_rate

    for step in range(1, n_steps):
        r_t = rates[step - 1]
        # TODO: optionally remove sqrt to simulate Vasicek model
        dr_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        rates[step] = abs(r_t + dr_t)
    return pd.DataFrame(data=inst_to_annual(rates), index=range(n_steps))
