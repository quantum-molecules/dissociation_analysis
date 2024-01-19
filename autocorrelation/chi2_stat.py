import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import sympy as sym
from sympy import lambdify
import math
import matplotlib.pyplot as plt

# other at least once used modules:
# import numpy.ma as ma
# import pandas as pd
# import matplotlib.font_manager as font_manager
# import scipy.constants as spC
# import os
# import random
# from uncertainties import ufloat



def chi2_nu(x, y, y_err=None, fit_model=None, N_param=None, analysis_details=0):
    alpha = 0
    if np.sum(y_err) == None:
        y_err = np.ones(len(y))
        alpha = 1
    if fit_model == None:
        return None
    elif N_param == None:
        return None
    nu = len(y) - N_param  # degrees of freedom
    res = y - fit_model(x)
    res_norm = res / y_err
    chi2 = np.sum((res_norm) ** 2)
    chi2nu = chi2 / nu

    if analysis_details == 1:
        print("-------------CHI2--------------")
        print("chi2 =", chi2)
        print("nu =", nu)
        print("chi2_nu =", chi2nu)
        print("residuals sigma distance:\n", res_norm)
        print("chi2_nu summation components:\n", res_norm ** 2 / nu)
        if alpha == 1:
            alpha = np.sqrt(chi2 / nu)
            print("ideal error for all values:", alpha)
        print("-------------------------------")
    return nu, chi2nu


def chi2_stat(nu, chi_nu, plot_min, plot_max):
    print('Create Matlab CHI2 statistics')
