import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fitting as SCRfit
import scipy.constants as spC

plot_params = {
    "font.size": 8,
    "legend.fontsize": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.titlepad": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": "#C0C0C0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "figure.figsize": (3.3, 2.2), # in inches
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Arial",
    "mathtext.default": "regular",
    "figure.subplot.right": 0.92,
    "figure.subplot.left": 0.18,
    "figure.subplot.bottom": 0.2,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.1,
    "figure.subplot.hspace": 0.1,
}

################# AUTOCORRELATION MEASUREMENT AND PLOT
path = r"Autocorrelatoin measurements.ods"
df = pd.read_excel(path, engine="odf", skiprows=[0, 39, 40, 41, 38, 37, 36], usecols=[4, 5])


t_delay = df["time delay (fs).1"].to_numpy()
V_osci = df["intensity (V).1"].to_numpy()


theta, theta_err, cov, fun2, fun2_err = SCRfit.fit_procedure(
    t_delay, V_osci, np.ones(len(V_osci)) * 1, fittype="Gauss", p0=[8, 2240, 10, -0.5], absolute_sigma=False
)

FWHM_t = 2 * theta[2] * np.sqrt(2 * np.log(2))
FWHM_t_err = 2 * theta_err[2] * np.sqrt(2 * np.log(2))

print("FWHM = (", FWHM_t, "±", FWHM_t_err, ") fs")
print("FWHM/sqrt(2) = (", FWHM_t / np.sqrt(2), "±", FWHM_t_err / np.sqrt(2), ") fs")


######################## SPECTRUM MEASUREMENT AND PLOT

path_spec = "HR2001601__0__16-47-52-080.txt"
df_spec = pd.read_csv(path_spec, skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], sep="\t", header=None)

# print(df_spec)


f_spec = spC.c / (df_spec[0] * 1e-9) / 1e12
V_spec = df_spec[1] / 10000

# f_spec = f_spec[1170:1190]
# V_spec = V_spec[1170:1190]

theta_spec, theta_spec_err, cov_spec, fun2_spec, fun2_spec_err = SCRfit.fit_procedure(
    f_spec, V_spec, np.ones(len(V_spec)) * 1, fittype="Gauss", p0=[3.061, 507.9, 1.3, -0.0895], absolute_sigma=False
)  # , bounds=np.array([[2.8,500,1,-1],[3.1,510,1.5,1]]).T)

FWHM_f = 2 * theta_spec[2] * np.sqrt(2 * np.log(2))
FWHM_f_err = 2 * theta_spec_err[2] * np.sqrt(2 * np.log(2))
print("FWHM_f = (", FWHM_f, "±", FWHM_f_err, ") THz")


print(
    "TBP = (",
    FWHM_t * 1e-15 / np.sqrt(2) * FWHM_f * 1e12,
    "±",
    np.sqrt(
        (FWHM_t_err * 1e-15 / np.sqrt(2) * FWHM_f * 1e12) ** 2 + (FWHM_t * 1e-15 / np.sqrt(2) * FWHM_f_err * 1e12) ** 2
    ),
    ")",
)


################################ PLOTTING

plot_params["figure.dpi"] = 300


############### plot autocorrelation
plt.rcParams.update(plot_params)
fig, ax0 = plt.subplots(ncols=1, nrows=1)
# plt.subplots_adjust(bottom=0.13, left=0.1, right=0.98, top=.98)#, wspace=0.25, hspace = 0.45)


tt = np.linspace(min(t_delay) - 50 - theta[1], max(t_delay) + 50 - theta[1], 1000)
norm_t = np.max(fun2(tt + theta[1]))
ax0.plot(tt, fun2(tt + theta[1]) / norm_t, c='r', linewidth=2.0)
ax0.scatter(t_delay - theta[1], V_osci / norm_t, marker="o", c="b", s=12, zorder=10)

ax0.set_xlim([-320, 320])
ax0.set_xlabel("Time delay (fs)")
ax0.set_ylabel("SHG intensity (a.u.)")
ax0.grid(which="both")


# fig.text(.02, .94, r"(a)", ha='center', fontsize=12)
ax0.text(-0.22, 1.0, "(a)", transform=plt.gca().transAxes)
# plt.tight_layout()
# plt.subplots_adjust(right=0.82,\
#                     left=0.18,\
#                     bottom=0.2,\
#                     top=0.95,\
#                     wspace=0.1,\
#                     hspace=0.1)
plt.savefig("OPA_autocorrelation_590.eps", format="eps")


############### plot spectrum
plt.rcParams.update(plot_params)
fig, ax1 = plt.subplots(ncols=1, nrows=1)
# plt.subplots_adjust(bottom=0.13, left=0.1, right=0.98, top=.98)#, wspace=0.25, hspace = 0.45)

ff = np.linspace(min(f_spec), max(f_spec), 10000)
norm_f = np.max(fun2_spec(ff))
ax1.plot(ff, fun2_spec(ff) / norm_f, c='r', linewidth=2.0)
ax1.scatter(f_spec, V_spec / norm_f, marker="o", c="b", s=12, zorder=10)

ax1.set_xlim(theta_spec[1] - 6.5, theta_spec[1] + 6.5)
ax1.set_xlabel("Frequency (THz)")
ax1.set_ylabel("Intensity (a.u.)")
ax1.grid(which="both")


# fig.text(.02, .94, r"(b)", ha='center', fontsize=12)
ax1.text(-0.22, 1.0, "(b)", transform=plt.gca().transAxes)
# plt.tight_layout()
# plt.subplots_adjust(right=0.82,\
#                     left=0.18,\
#                     bottom=0.2,\
#                     top=0.95,\
#                     wspace=0.1,\
#                     hspace=0.1)
plt.savefig("OPA_spectrum_590.eps", format="eps")


plt.show()
