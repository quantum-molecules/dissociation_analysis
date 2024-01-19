import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sympy import lambdify


# other at least once used modules:
# import numpy.ma as ma
# import pandas as pd
# import matplotlib.font_manager as font_manager
# import scipy.constants as spC
# import os
# import random
# from uncertainties import ufloat

import chi2_stat as SCRc2
import other_scripts as SCRoth


def loss_function(losstype, c_huber=3, epsilon=1e-6):
    # one of the references (squared loss = quadratic loss. but here 0.5 is missing) https://www.statlect.com/glossary/loss-function
    if losstype == "squared_loss":

        def fun(t):
            return 0.5 * np.sum(t**2)

    if losstype == "huber_loss":

        def fun(t):
            return np.sum(
                (abs(t) < c_huber) * 0.5 * t**2 + (abs(t) >= c_huber) * -c_huber * (0.5 * c_huber - abs(t))
            )

    # the following are not tested
    if losstype == "absolute_loss":

        def fun(t):
            return abs(t)

    if losstype == "log-cosh_loss":

        def fun(t):
            return np.sum(np.log(np.cosh(t)))

    return fun

    if losstype == "pseudo-huber_loss":

        def fun(t):
            return c_huber**2 * np.sum(np.sqrt(1 + (t / c_huber) ** 2) - 1)

    if losstype == "epsilon-insensitive_loss":

        def fun(t):
            return np.sum(np.max([np.zeros(t.shape), abs(t) - epsilon * np.ones(t.shape)], axis=0))

    else:
        print("ERROR! Spelled losstype does not exist.")

    return fun


def loss_to_minimize_general(fit_fun, theta, x, y, y_err, losstype="squared_loss", c_huber=3):
    loss_fun = loss_function(losstype, c_huber)

    n = len(theta)
    if n == 1:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0])
            return dy / y_err

    elif n == 2:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1])
            return dy / y_err

    elif n == 3:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1], theta[2])
            return dy / y_err

    elif n == 4:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1], theta[2], theta[3])
            return dy / y_err

    elif n == 5:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1], theta[2], theta[3], theta[4])
            return dy / y_err

    elif n == 6:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])
            return dy / y_err

    elif n == 7:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])
            return dy / y_err

    elif n == 8:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7])
            return dy / y_err

    elif n == 9:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(
                x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8]
            )
            return dy / y_err

    elif n == 10:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(
                x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9]
            )
            return dy / y_err

    elif n == 11:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(
                x,
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                theta[5],
                theta[6],
                theta[7],
                theta[8],
                theta[9],
                theta[10],
            )
            return dy / y_err

    elif n == 12:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(
                x,
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                theta[5],
                theta[6],
                theta[7],
                theta[8],
                theta[9],
                theta[10],
                theta[11],
            )
            return dy / y_err

    elif n == 15:

        def residuals_norm(theta, x=x, y=y, y_err=y_err):
            dy = y - fit_fun(
                x,
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                theta[5],
                theta[6],
                theta[7],
                theta[8],
                theta[9],
                theta[10],
                theta[11],
                theta[12],
                theta[13],
                theta[14],
            )
            return dy / y_err

    else:
        print("ERROR! Too many free parameters.")

    return loss_fun(residuals_norm(theta))


def model_functions(fittype):
    if fittype == "const":

        def fun(x, c):
            return c

        def fun_sym(x, c):
            return c

        fun_str = "c"
        var_names = "c"

    elif fittype == "lin1":

        def fun(x, a, b):
            return a * x + b

        def fun_sym(x, a, b):
            return a * x + b

        fun_str = "a*x + b"
        var_names = ("a", "b")

    elif fittype == "lin2":

        def fun(x, a, x0):
            return a * (x - x0)

        def fun_sym(x, a, x0):
            return a * (x - x0)

        fun_str = "a*(x-x0)"
        var_names = ("a", "x0")

    elif fittype == "lin_noOff":

        def fun(x, a):
            return a * x

        def fun_sym(x, a):
            return a * x

        fun_str = "a*x"
        var_names = "a"

    elif fittype == "quad1":

        def fun(x, a, b, c):
            return a * x**2 + b * x + c

        def fun_sym(x, a, b, c):
            return a * x**2 + b * x + c

        fun_str = "a*x^2 + b*x + c"
        var_names = ("a", "b", "c")

    elif fittype == "quad2":

        def fun(x, a, x0, c):
            return a * (x - x0) ** 2 + c

        def fun_sym(x, a, x0, c):
            return a * (x - x0) ** 2 + c

        fun_str = "a*(x - x0)^2 + c"
        var_names = ("a", "x0", "c")

    elif fittype == "quad2_noOff":

        def fun(x, a, x0):
            return a * (x - x0) ** 2

        def fun_sym(x, a, x0):
            return a * (x - x0) ** 2

        fun_str = "a*(x - x0)^2"
        var_names = ("a", "x0")

    elif fittype == "quad_noOff_noShi":

        def fun(x, a):
            return a * x**2

        def fun_sym(x, a):
            return a * x**2

        fun_str = "a*x^2"
        var_names = "a"

    elif fittype == "quad_noShi":

        def fun(x, a, c):
            return a * x**2 + c

        def fun_sym(x, a, c):
            return a * x**2 + c

        fun_str = "a*x^2 + c"
        var_names = ("a", "c")

    elif fittype == "Gauss":
        # FWHM = 2*sigma*np.sqrt(2*np.log(2))
        def fun(x, A, x0, sigma, c):
            return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c

        def fun_sym(x, A, x0, sigma, c):
            return A * sym.exp(-0.5 * ((x - x0) / sigma) ** 2) + c

        fun_str = "A*exp( -0.5*((x - x0)/sigma)^2) + c"
        var_names = ("a", "x0", "sigma", "c")

    elif fittype == "Lorentz":
        # FWHM = 2*gamma
        def fun(x, A, x0, gamma, c):
            return A / ((1 + ((x - x0) / gamma) ** 2)) + c

        def fun_sym(x, A, x0, gamma, c):
            return A / ((1 + ((x - x0) / gamma) ** 2)) + c

        fun_str = "A/((1 + ((x - x0)/gamma)**2)) + c"
        var_names = ("a", "x0", "gamma", "c")

    elif fittype == "Lorentz_noOff":
        # FWHM = 2*gamma
        def fun(x, A, x0, gamma):
            return A / ((1 + ((x - x0) / gamma) ** 2))

        def fun_sym(x, A, x0, gamma):
            return A / ((1 + ((x - x0) / gamma) ** 2))

        fun_str = "A/((1 + ((x - x0)/gamma)**2))"
        var_names = ("a", "x0", "gamma")

    elif fittype == "logistic":

        def fun(x, L, x0, k, c):
            return L / (1 + np.exp(-k * (x - x0))) + c

        def fun_sym(x, L, x0, k, c):
            return L / (1 + sym.exp(-k * (x - x0))) + c

        fun_str = "L/(1 + exp(-k*(x - x0))) + c"
        var_names = ("L", "x0", "k", "c")

    elif fittype == "logistic_noOff":

        def fun(x, L, x0, k):
            return L / (1 + np.exp(-k * (x - x0)))

        def fun_sym(x, L, x0, k):
            return L / (1 + sym.exp(-k * (x - x0)))

        fun_str = "L/(1 + exp(-k*(x - x0)))"
        var_names = ("L", "x0", "k")

    elif fittype == "sin":

        def fun(t, A, omega, t0, c):
            return A * np.sin(omega * (t - t0)) + c

        def fun_sym(t, A, omega, t0, c):
            return A * sym.sin(omega * (t - t0)) + c

        fun_str = "A*sin(omega*(t-t0)) + c"
        var_names = ("A", "omega", "t0", "c")

    elif fittype == "sin2":

        def fun(t, A, omega, t0, c):
            return A * (np.sin(omega * (t - t0) / 2)) ** 2 + c

        def fun_sym(t, A, omega, t0, c):
            return A * (sym.sin(omega * (t - t0) / 2)) ** 2 + c

        fun_str = "A*(sin(omega*(t - t0)/2))^2 + c"
        var_names = ("A", "omega", "t0", "c")

    else:
        print("ERROR! Spelled fittype does not exist.")

    return fun, fun_sym, fun_str, var_names


def fit_procedure(
    x_fit,
    y_fit,
    y_fit_err,
    fittype="lin1",
    losstype="squared_loss",
    c_huber=3,
    p0=None,
    bounds=None,
    print_fitparams=1,
    cusom_model_function_list=None,
    absolute_sigma=True,
):
    """
    :param x_fit: x data
    :param y_fit: y data
    :param y_fit_err: y error
    :param fittype: Specify model function as string
    :param losstype: Specify loss function to minimize for fitting as string
    :param c_huber: huber_loss parameter
    :param p0: initial guess of fitparameters
    :param bounds: bound in between which the fit parameters should stay
    :param print_fitparams: result of fitting procedure
    :param cusom_model_function_list: 4-tupel containing fun, fun_sym, fun_str and var_names
    :return: theta, theta_err, cov, fun2, fun2_err
    """

    if fittype == "custom":
        fun = cusom_model_function_list[0]
        fun_sym = cusom_model_function_list[1]
        fun_str = cusom_model_function_list[2]
        var_names = cusom_model_function_list[3]
    else:
        fun, fun_sym, fun_str, var_names = model_functions(fittype)

    N_param = len(var_names)
    if p0 == None:
        p0 = np.ones(N_param)

    def loss_to_minimize(theta, x, y, y_err):
        return loss_to_minimize_general(
            fun, theta, x, y, y_err, losstype, c_huber
        )  # returns loss_fun(residuals_norm(theta))

    fitObj = minimize(loss_to_minimize, x0=p0, args=(x_fit, y_fit, y_fit_err), bounds=bounds)
    cov = fitObj["hess_inv"]
    theta = fitObj["x"]
    theta_err = np.sqrt(np.diagonal(cov))

    if N_param == 1:

        def fun2(x):
            return fun(x, theta[0])

    elif N_param == 2:

        def fun2(x):
            return fun(x, theta[0], theta[1])

    elif N_param == 3:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2])

    elif N_param == 4:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2], theta[3])

    elif N_param == 5:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2], theta[3], theta[4])

    elif N_param == 6:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])

    elif N_param == 7:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])

    elif N_param == 8:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7])

    elif N_param == 9:

        def fun2(x):
            return fun(x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8])

    elif N_param == 10:

        def fun2(x):
            return fun(
                x, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9]
            )

    elif N_param == 11:

        def fun2(x):
            return fun(
                x,
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                theta[5],
                theta[6],
                theta[7],
                theta[8],
                theta[9],
                theta[10],
            )

    elif N_param == 12:

        def fun2(x):
            return fun(
                x,
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                theta[5],
                theta[6],
                theta[7],
                theta[8],
                theta[9],
                theta[10],
                theta[11],
            )

    elif N_param == 15:

        def fun2(x):
            return fun(
                x,
                theta[0],
                theta[1],
                theta[2],
                theta[3],
                theta[4],
                theta[5],
                theta[6],
                theta[7],
                theta[8],
                theta[9],
                theta[10],
                theta[11],
                theta[12],
                theta[13],
                theta[14],
            )

    else:
        print("ERROR! Too many free parameters.")

    nu, chi2nu = SCRc2.chi2_nu(x_fit, y_fit, y_fit_err, fit_model=fun2, N_param=N_param, analysis_details=0)

    if absolute_sigma == False:
        cov = cov * chi2nu
        theta_err = np.sqrt(np.diagonal(cov))

    if N_param <= 5:
        fun2_err, fun2_err_sym = SCRoth.Konfidenzband(fun_sym, theta, theta_err, cov)
    else:
        fun2_err = None
        fun2_err_sym = None
        print("CAREFUL! No confidence band created, fun2 = None")

    if print_fitparams == 1:
        print("Model: f(x) = " + fun_str)
        for i, t_str in enumerate(var_names):
            _theta = np.format_float_scientific(SCRoth.signif(theta[i], 7))
            _theta_err = np.format_float_scientific(SCRoth.signif(theta_err[i], 2))

            if math.isnan(theta_err[i]):
                strr = "nan"
            elif theta_err[i] > abs(theta[i]):
                strr = "error too big"
            else:
                strr = SCRoth.propper_error_display(theta[i], theta_err[i])
            print(
                "   " + t_str + " (theta[" + str(i) + "]) =", _theta, "(" + str(_theta_err) + ")           -> " + strr
            )

        print("   chi^2_nu = " + str(SCRoth.signif(chi2nu, 3)) + ", nu =", nu)

    # ax.plot(x_fit, y_fit)
    # t = np.linspace(min(x_fit), max(x_fit), 20)
    # ax.plot(t, fun(t, theta[0], theta[1]), '-.', label=label, c=color)
    # plt.show()

    return theta, theta_err, cov, fun2, fun2_err
