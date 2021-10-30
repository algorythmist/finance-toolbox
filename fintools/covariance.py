import numpy as np
import pandas as pd


class CovarianceEstimator:
    """
    Boris, if you are reviewing this, YOU SUCK!!!
    """
    def estimate_covariance(self, returns):
        pass


class SampleCovarianceEstimator(CovarianceEstimator):

    def estimate_covariance(self, returns):
        """
        Returns the sample covariance of the supplied returns
        """
        return returns.cov()


class ConstantCorrelationCovarianceEstimator(CovarianceEstimator):

    def estimate_covariance(self, returns):
        """
        Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
        """
        rhos = returns.corr()
        n = rhos.shape[0]
        # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
        rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
        ccor = np.full_like(rhos, rho_bar)
        np.fill_diagonal(ccor, 1.)
        sd = returns.std()
        return pd.DataFrame(ccor * np.outer(sd, sd), index=returns.columns, columns=returns.columns)
