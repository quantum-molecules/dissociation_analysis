from pathlib import Path
from enum import Enum
from typing import TypedDict, List

# Directories
cwd = Path().cwd()

experiment = "dissociation measurement of CaOH+"

logfile = "dissociation_analysis.log"


def datainfofile(task):
    """
    Information file for loading and analzying all data [json file]
    """
    return Path(f"data_info_json/data_information_{task}.json")


def indatadir(task):
    """
    Local input data file directory (on git repo)
    """
    return cwd.joinpath("input").joinpath(task)


def outdatadir(task):
    """
    Local output data file directory (on git repo)
    """
    return cwd.joinpath("output").joinpath("data").joinpath(task)


def outplotdir(task):
    """
    Local output plots directory (on git repo)
    """
    return cwd.joinpath("output").joinpath("plots").joinpath(task)


ylinesumfile = "y_line_sum.txt"
"""
File for storing y_line_sum during an event
"""
imgprocfile = "image_analysis_data.txt"
"""
File for storing ion finder data from images during an event
"""
diss_access = "access_log_dissociation_main_py.txt"
"""
Access log filename to verify connection to QOS server
"""

# Format specifiers and delimiters
datadelim = "    "
datafmt = "%s"

# Approaches & measurement types specs
approach_f = "frequentist"  # frequentist statistical approach to Process class
approach_b = "bayesian"  # bayesian statistical approach to Process class
type_int = "intervals"  # Measurement type argument (string to pass to Spectra class)
type_liv = "live"  # Measurement type argument (string to pass to Spectra class)


class ApproachEnum(Enum):
    frequentist = approach_f  # frequentist statistical approach to Process class
    bayesian = approach_b  # bayesian statistical approach to Process class


class MeasureEnum(Enum):
    intervals = type_int  # Measurement type argument (string to pass to Spectra class)
    live = type_liv  # Measurement type argument (string to pass to Spectra class)


def headers(approach: ApproachEnum) -> str:
    x = datadelim
    if approach.value == approach_b:
        _header_list = [
            "wl(nm)",
            "mean_lt(s)",
            "mle_lt(s)",
            "CI_low[lt]",
            "CI_high[lt]",
            "int(W/m2)",
            "int_er",
            "label",
            "file",
        ]
        header = x.join(_header_list)
        return header
    if approach.value == approach_f:
        return f"wavelength(nm){x}lifetime(s){x}lifetime_lobound(s){x}lifetime_upbound(s){x}power(uW){x}ppdivider{x}intensity{x}file{x}label"
    return ""


def lifetime_outfile(task, measure: MeasureEnum, approach: ApproachEnum):
    return outdatadir(task).joinpath(f"dissociation_lifetimes_{measure.value}_{approach.value}.dat")

def header_success_rate(do_2photon) -> str:
    x = datadelim
    if do_2photon:
        return x.join(["wl(nm)","1","5","10","30","60","file"])
    else:
        return x.join(["wl(nm)","1","5","10","30","60","120","file"])

def succ_rate_outfile(task):
    return outdatadir(task).joinpath(f"dissociation_successful_rate.dat")

# Plot parameters
cmap_post_evo = "inferno"
"""
Color map for Bayesian inference method density plot showing evolution of posterior for lifetime
(options: viridis, plasma, inferno, magma, cividis)
"""
cmap_y_line_sum = "gray"
"""
Color map for images of y_line_sum time series for data inspection (black and white)
"""

# Numerical parameters
f_rep_pump = 100e03
"""
OPA pump repetition rate [Hz] (100kHz) -> this number is fixed
"""
N_p = 10000
"""
N_prior for Bayesian approach
"""
max_t = 3000
"""
Max time for perform Bayesian inference
"""
min_t = max_t / N_p
"""
Min time for perform Bayesian inference
"""
er = 0.01
"""
Error rate for Bayesian inference
"""
confidence = 0.68
"""
Confidence for setting confidence interval in Bayesian inference plotting
"""
jwidth = 0.5
"""
Jitter width for multiple points at a single wavelength in Spectra class
"""
t_resolution = 0.1
"""
Resolution of grid points for y_line_sum images in units of seconds in ImageSeries class
"""


def power_er(do_2photon):
    """
    power measurement uncertainty in uW (1 sigma for MC)
    """
    if do_2photon:
        return 100.0  # for 2-photon
    else:
        return 0.1  # for single photon


waist_er = 10
"""
FWHM measurement uncertainty in um (1 sigma for MC)
"""
position_er = 10
"""
alignment to ion position uncertainty in um (1 sigma for MC)
"""
N_sample = 100000
"""
Number of samples in Monte Carlo calculation of the uncertainties
"""

# Boolean parameters
save_y_line_sum = True
"""
Save y_line_sum plots or not
"""

# For generating plots
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
