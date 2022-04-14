import pytest
from fintools.industry_data import *
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS
from sklearn.model_selection import train_test_split

def test_covariance():
    inds = ['Food', 'Beer', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths', 'Hlth',
            'Chems', 'Txtls', 'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos', 'Carry',
            'Mines', 'Coal', 'Oil', 'Util', 'Telcm', 'Servs', 'BusEq', 'Paper',
            'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin', 'Other']
    ind_rets = load_industry_data('ind49_m_ew_rets.csv')
    #print(ind_rets.head())
    #ind_caps = load_market_caps(size=49, weights=True)["1974":]
    ind_rets_train = ind_rets['1990':'2000']
    ind_rets_test = ind_rets['2000':'2018']
    print()
    emp_model = EmpiricalCovariance().fit(ind_rets_train)
    print(emp_model)
    loglik_emp = emp_model.score(ind_rets_test)
    print(loglik_emp)
    loglik_lw = LedoitWolf().fit(ind_rets_train).score(ind_rets_test)
    print(loglik_lw)
    loglik_oas = OAS().fit(ind_rets_train).score(ind_rets_test)
    print(loglik_oas)

    #######
    mixed_ret_train, mixed_ret_test = train_test_split(ind_rets['1990':'2018'], test_size=0.4)

    emp_model = EmpiricalCovariance().fit(mixed_ret_train)
    loglik_emp = emp_model.score(mixed_ret_test)
    print(loglik_emp)
    loglik_lw = LedoitWolf().fit(mixed_ret_train).score(mixed_ret_test)
    print(loglik_lw)

