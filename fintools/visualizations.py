import numpy as np
import pandas as pd
from fintools import *


def plot_efficiency_frontier(expected_return, covariance,
                             risk_free_rate=0,
                             n_points =20,
                             style='.-',
                             legend=False,
                             show_cml=False,
                             show_ew=False,
                             show_gmv=False):
    """
    Plots the multi-asset efficient frontier

    :param expected_return:
    :param covariance:
    :param risk_free_rate: the risk free rare
    :param n_points: points to evaluate
    :param style:
    :param show_cml: Show the capital market line
    :param show_ew : Show equally weighted portfolio
    :param show_gmv: Show the global minimum volatility portfolio
    """
    target_returns = np.linspace(expected_return.min(), expected_return.max(), n_points)
    weights = [minimize_volatility(tr, expected_return, covariance) for tr in target_returns]
    rets = [compute_portfolio_return(w, expected_return) for w in weights]
    vols = [compute_portfolio_variance(w, covariance) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left=0)
        # get MSR
        w_msr = maximize_sharpe_ratio(expected_return, covariance, risk_free_rate=risk_free_rate)
        r_msr = compute_portfolio_return(w_msr, expected_return)
        vol_msr = compute_portfolio_variance(w_msr, covariance)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [risk_free_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = expected_return.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = compute_portfolio_return(w_ew, expected_return)
        vol_ew = compute_portfolio_variance(w_ew, covariance)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = global_minimum_variance_portfolio(covariance)
        r_gmv = compute_portfolio_return(w_gmv, expected_return)
        vol_gmv = compute_portfolio_variance(w_gmv, covariance)
        # add GMV
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
    return ax
