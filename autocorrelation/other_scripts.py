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


def student_t():
    print("WIP")



def Konfidenzband(fit_fun, param, param_err, cov_var):  # could work for up to five parameters. testing required
    x = sym.Symbol('x')

    r_ij = np.zeros(cov_var.shape)
    for i in range(len(cov_var[:, 0])):
        for j in range(len(cov_var[0, :])):
            r_ij[i, j] = cov_var[i, j] / np.sqrt(cov_var[i, i] * cov_var[j, j])

    def summing(f_vec, err_vec):
        alpha_y2 = 0
        for i in range(len(cov_var[:, 0])):
            alpha_y2 += (f_vec[i] * err_vec[i]) ** 2
            for j in range(i):
                alpha_y2 += 2 * r_ij[i, j] * (f_vec[i] * err_vec[i]) * (f_vec[j] * err_vec[j])
        return alpha_y2

    n = len(param)

    if n == 1:
        a1 = sym.symbols('a1')
        a1_err = sym.symbols('a1_err')

        f1 = sym.diff(fit_fun(x, a1), a1)

        f_vec = np.array([f1])
        err_vec = np.array([a1_err])

        alpha_y2 = summing(f_vec, err_vec)
        alpha_y_sym = (alpha_y2.subs({a1: param[0], a1_err: param_err[0]})) ** (1 / 2)

    if n == 2:
        a1, a2 = sym.symbols('a1, a2')
        a1_err, a2_err = sym.symbols('a1_err, a2_err')

        f1 = sym.diff(fit_fun(x, a1, a2), a1)
        f2 = sym.diff(fit_fun(x, a1, a2), a2)

        f_vec = np.array([f1, f2])
        err_vec = np.array([a1_err, a2_err])

        alpha_y2 = summing(f_vec, err_vec)
        alpha_y_sym = (alpha_y2.subs({a1: param[0], a1_err: param_err[0],
                                      a2: param[1], a2_err: param_err[1]})) ** (1 / 2)

    if n == 3:
        a1, a2, a3 = sym.symbols('a1, a2, a3')
        a1_err, a2_err, a3_err = sym.symbols('a1_err, a2_err, a3_err')

        f1 = sym.diff(fit_fun(x, a1, a2, a3), a1)
        f2 = sym.diff(fit_fun(x, a1, a2, a3), a2)
        f3 = sym.diff(fit_fun(x, a1, a2, a3), a3)

        f_vec = np.array([f1, f2, f3])
        err_vec = np.array([a1_err, a2_err, a3_err])

        alpha_y2 = summing(f_vec, err_vec)
        alpha_y_sym = (alpha_y2.subs({a1: param[0], a1_err: param_err[0],
                                      a2: param[1], a2_err: param_err[1],
                                      a3: param[2], a3_err: param_err[2]})) ** (1 / 2)

    if n == 4:
        a1, a2, a3, a4 = sym.symbols('a1, a2, a3, a4')
        a1_err, a2_err, a3_err, a4_err = sym.symbols('a1_err, a2_err, a3_err, a4_err')

        f1 = sym.diff(fit_fun(x, a1, a2, a3, a4), a1)
        f2 = sym.diff(fit_fun(x, a1, a2, a3, a4), a2)
        f3 = sym.diff(fit_fun(x, a1, a2, a3, a4), a3)
        f4 = sym.diff(fit_fun(x, a1, a2, a3, a4), a4)

        f_vec = np.array([f1, f2, f3, f4])
        err_vec = np.array([a1_err, a2_err, a3_err, a4_err])

        alpha_y2 = summing(f_vec, err_vec)
        alpha_y_sym = (alpha_y2.subs({a1: param[0], a1_err: param_err[0],
                                      a2: param[1], a2_err: param_err[1],
                                      a3: param[2], a3_err: param_err[2],
                                      a4: param[3], a4_err: param_err[3]})) ** (1 / 2)

    if n == 5:
        a1, a2, a3, a4, a5 = sym.symbols('a1, a2, a3, a4, a5')
        a1_err, a2_err, a3_err, a4_err, a5_err = sym.symbols('a1_err, a2_err, a3_err, a4_err, a5_err')

        f1 = sym.diff(fit_fun(x, a1, a2, a3, a4, a5), a1)
        f2 = sym.diff(fit_fun(x, a1, a2, a3, a4, a5), a2)
        f3 = sym.diff(fit_fun(x, a1, a2, a3, a4, a5), a3)
        f4 = sym.diff(fit_fun(x, a1, a2, a3, a4, a5), a4)
        f5 = sym.diff(fit_fun(x, a1, a2, a3, a4, a5), a5)

        f_vec = np.array([f1, f2, f3, f4, f5])
        err_vec = np.array([a1_err, a2_err, a3_err, a4_err, a5_err])

        alpha_y2 = summing(f_vec, err_vec)
        alpha_y_sym = (alpha_y2.subs({a1: param[0], a1_err: param_err[0],
                                      a2: param[1], a2_err: param_err[1],
                                      a3: param[2], a3_err: param_err[2],
                                      a4: param[3], a4_err: param_err[3],
                                      a5: param[4], a5_err: param_err[4]})) ** (1 / 2)

    alpha_y = lambdify(x, alpha_y_sym, 'numpy')
    # print(alpha_y_sym)
    # print(alpha_y(0))
    return alpha_y, alpha_y_sym




def ChauvenetsCriterion_crit_val(N, N_outside):
    def crit_k(k):
        return math.erf(k / np.sqrt(2)) - (1 - N_outside / N)

    sol = sp.optimize.root_scalar(crit_k, x0=1, x1=2)
    k_cr = sol.root

    return k_cr

def ChauvenetsCriterion_plot(x, y, y_err, fit_model, x_min, x_max, range_k_cr,
                             indexing=True):  # Verwerfen von messdaten
    # k is the deviation of the suspect data points from the model value, normalized using its error
    # k_cr criterion:
    N_outside = 1 / 2  # motivated critical value
    N = len(y)
    k_cr = ChauvenetsCriterion_crit_val(N, N_outside)

    res = (y - fit_model(x)) / y_err

    ##### the following is not necessary
    # res_m = np.mean(res)
    # res_s = np.std(res, ddof=1)   #empirische standard abweichung.
    ##### ddf = Delta Degrees of Freedom. The divisor used in calculations is N - ddof. By default ddof is zero.
    ##### = np.sqrt(np.sum((res-np.mean(res))**2)/(len(res)-1))
    # k_res = (res - 0)/res_s

    # print('hi', k_cr, res, res_s, k_res)

    ##### plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2), dpi=100)
    plt.subplots_adjust(left=.25, bottom=0.3, right=.99, top=.99)

    label_size = 16
    if indexing == True:
        _x = np.arange(0, N)
        _x_min = -0.5
        _x_max = N - 0.5
        plt.xlabel(r"Index $i$", math_fontfamily='cm', fontsize=label_size, family="STIXGeneral")
        plt.ylabel(r"Residuals $r_i$", math_fontfamily='cm', fontsize=label_size, family="STIXGeneral")
        plt.title("", math_fontfamily='cm', fontsize=label_size, family="STIXGeneral")
    elif indexing == False:
        _x = x
        _x_min = x_min
        _x_max = x_max
        plt.xlabel(r"$x_i$-values", math_fontfamily='cm', fontsize=label_size, family="STIXGeneral")
        plt.ylabel(r"Residuals $r(x_i)$", math_fontfamily='cm', fontsize=label_size, family="STIXGeneral")
        plt.title("", math_fontfamily='cm', fontsize=label_size, family="STIXGeneral")
    else:
        print("Error")
        return None

    plt.scatter(_x, res,
                # marker
                marker='o',
                color=(0, 0.4470, 0.7410),
                edgecolor=(0, 0.4470, 0.7410),
                linewidths=1,
                # markerfacecolor='none',
                s=2,
                label="Data")

    t = np.linspace(_x_min, _x_max, 2)
    k_cr_notation_offset = 0
    for i, k_cr_text, ver_align in ((-1, "$-k_\mathrm{cr}$", "top"), (1, "$k_\mathrm{cr}$", "bottom")):
        ax.text(_x_min + (_x_max - _x_min) * .95, i * (k_cr + k_cr_notation_offset),
                k_cr_text,
                ha="center",
                va=ver_align,
                size=14,
                family="STIXGeneral",
                math_fontfamily='cm')
        ax.plot(t, i * k_cr * np.ones(2),
                color='black',
                linewidth=.5,
                linestyle='-')

    ax.grid(visible=True,
            which='major',  # 'major', 'minor'
            axis='both',  # 'x', 'y'
            color='.8',
            alpha=.5,
            linestyle='-',
            linewidth=.5)

    plt.xlim([_x_min, _x_max])
    plt.ylim([-range_k_cr * k_cr, range_k_cr * k_cr])

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(label_size)

    for tick in ax.get_xticklabels():
        tick.set_fontname("STIXGeneral")
    for tick in ax.get_yticklabels():
        tick.set_fontname("STIXGeneral")

    plt.show()




def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def propper_error_display(a, a_err):
    expo = int(np.floor(np.log10(a_err)))
    err_value = a_err*10**-expo
    if err_value < 2 and err_value >= 1:
        signif_v = -expo + 1
        err_scale = 10
    else:
        signif_v = -expo
        err_scale = 1
    if signif_v < 0:
        str_num = "negative float format error...:"+'.'+str(signif_v)+'f'
    else:
        str_num = f"{format(a, '.'+str(signif_v)+'f')}({round(err_value*err_scale)})"
    return str_num


def comb_val(x, x_err):
    w = 1 / x_err ** 2
    x_comb = 0
    for i in range(len(x)):
        x_comb += x[i] * w[i]
    return x_comb / sum(w), 1 / np.sqrt(sum(w))


def rule_of_succession(p, n=100):    #for Rabi flops if certain pulse length only showed 0 or only showed 1
    return (p*n+1)/(n+2)

def quantumProjectionNoise(p, n=100):   #error for binomial distribution
    return np.sqrt(p*(1-p)/n)





