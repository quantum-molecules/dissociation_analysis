# Module containing classes and associated methods for analyzing dissociation data for dissociation_main.py.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from logger import Logger
from readin import InputData
from constants import (
    datadelim,
    datafmt,
    N_p,
    max_t,
    min_t,
    er,
    confidence,
    power_er,
    waist_er,
    position_er,
    N_sample,
    cmap_post_evo,
    logfile,
    ApproachEnum,
    MeasureEnum,
    lifetime_outfile,
    headers,
    outdatadir,
    outplotdir,
    header_success_rate,
    succ_rate_outfile,
)

matplotlib.use("pgf")
matplotlib.rc("text", usetex=True)
# Use 'Agg' backend, since the default 'TKAgg' will cause an error
# related to memory when processing ImageSeries.image() when there are many data sets


class ProcessData:
    """
    Class which contains methods to processes dissociation probability data.
    This class processes intervals and live data simultaneously but separately.
    The frequentist approach fits the data to a cumlative probability distribution using lmfit.
    The Bayesian approach utilizes Bayesian inference to estimate the lifetime and confidence intervals.
    The 'plot' method takes an argument the output of either of the estimation methods.
    """

    def __init__(
        self,
        task,
        copy_files=False,
        do_2photon=False,
        approach: ApproachEnum = ApproachEnum.bayesian,
        process_y_line_sum=False,
    ):
        """Initialization of the class

        Args:
            copy_files (bool, optional): whether or not to copy files from remote qos folder. Defaults to False.
            approach (ApproachEnum, optional): bayesian or frequentist (not supported). Defaults to approach_b (bayesian).
            cmap2 (str, optional): color map for plotting. Defaults to constants.cmap2.
        """
        self.logger = Logger("data_processing", logfile).get_logger()
        self.task = task
        self.approach = approach
        if self.approach == ApproachEnum.frequentist:
            self.logger.error("Frequentist method is not support for now")
            return
        self.do_2photon = do_2photon
        self.data = InputData(self.task)
        if copy_files:
            self.data.copy_files_from_qos()
        self.data.read(process_y_line_sum=process_y_line_sum)
        if not self.data.isRead:
            self.logger.error("Reading input data failed, exiting...")
            return
        self.datadict = {
            MeasureEnum.intervals: self.data.result_int,
            MeasureEnum.live: self.data.result_liv,
        }
        """Data dictionary as ouput from readin method"""

        self.t_values = np.linspace(min_t, max_t, N_p)
        """
        t_values for prior distribution (these correspond to different lifetime values
        to which self.prior is assigned as y_values to build a probability distribution)
        """
        self.bayesian_result = {MeasureEnum.intervals: {}, MeasureEnum.live: {}}

    def run(self, do_int: bool = True, do_liv: bool = True, plot: bool = True, save_plot: bool = True):
        """Run analysis of the data

        Args:
            do_int (bool, optional): analyse the intervals data. Defaults to True.
            do_liv (bool, optional): analyse the live data. Defaults to True.
            plot (bool, optional): plot the results or not. Defaults to True.
            save_plot (bool, optional): save the plot to files or not. Defaults to True.
        """
        outdatadir(self.task).mkdir(parents=True, exist_ok=True)
        outplotdir(self.task).mkdir(parents=True, exist_ok=True)
        self.write_success_rate()
        if do_int:
            self.write_bayesian(MeasureEnum.intervals, plot=plot, save_plot=save_plot)
        if do_liv:
            self.write_bayesian(MeasureEnum.live, plot=plot, save_plot=save_plot)
        self.logger.info("Data processing finished.")

    def write_success_rate(self):
        outf = succ_rate_outfile(self.task)
        with open(outf, "w") as f:
            np.savetxt(f, [], header=header_success_rate(self.do_2photon), delimiter=datadelim, comments="")
            for measurement in self.data.datainfo_json.keys():
                self.logger.info(f"Writing dissociation successful rate: {measurement}")
                succ_rate = self.get_success_rate(measurement)
                res = [[self.data.datainfo_json[measurement]["wavelength"]] + list(succ_rate.values()) + [measurement]]
                np.savetxt(
                    f,
                    res,
                    delimiter=datadelim,
                    fmt=datafmt,
                ) 

    def write_bayesian(self, meas_type: MeasureEnum, plot: bool = True, save_plot: bool = True):
        """Perform bayesian analysis and write the results to files

        Args:
            meas_type (MeasureEnum): intervals or live data
            plot (bool, optional): plot the results or not. Defaults to True.
            save_plot (bool, optional): save the plot to files or not. Defaults to True.
        """
        self.logger.info(f"Bayesian inference method on {meas_type.value} data")

        outf = lifetime_outfile(self.task, meas_type, ApproachEnum.bayesian)  # filename of result output
        plotevofile1 = outplotdir(self.task).joinpath(
            f"{meas_type.value}_bayesian_likelihood_and_mean_CI_evolution.pdf"
        )
        plotevofile2 = outplotdir(self.task).joinpath(f"{meas_type.value}_bayesian_posterior_evolution.pdf")

        # add header
        with open(outf, "w") as f:
            np.savetxt(
                f,
                [],
                header=headers(ApproachEnum.bayesian),
                delimiter=datadelim,
                comments="",
            )

        # Estimate dissociation lifetime for each measurement, save output to file, and plot

        with open(outf, "a") as f, PdfPages(str(plotevofile1)) as pdf1, PdfPages(str(plotevofile2)) as pdf2:
            for measurement in self.data.datainfo_json.keys():  # For each run (measurement), do the following:
                self.logger.info(f"Executing Bayesian inference method on {meas_type.value} data: {measurement}")

                # Analyze data with bayesian inference approach
                posterior, mean_array, max_array, conf_array, posterior_array = self.bayesian_inference(
                    measurement, meas_type
                )
                # Save in result dictionary the final value for the mean lifetime and confidence interval
                self.bayesian_result[meas_type][measurement] = {
                    "mean_lifetime": mean_array[-1],
                    "mle_lifetime": max_array[-1],
                    "confidence_interval": conf_array[-1],
                    "probability_distribution": posterior,
                }
                self.calc_uncertainty(measurement, meas_type, conf=confidence)

                save_list = []  # line to be saved in file
                save_list.append(
                    [  # measurement info from the json file shall prevail (there are some errors in meas_info.txt)
                        self.data.datainfo_json[measurement]["wavelength"],
                        self.bayesian_result[meas_type][measurement]["mean_lifetime"],
                        self.bayesian_result[meas_type][measurement]["mle_lifetime"],
                        self.bayesian_result[meas_type][measurement]["confidence_interval"][0],
                        self.bayesian_result[meas_type][measurement]["confidence_interval"][1],
                        self.bayesian_result[meas_type][measurement]["intensity"],
                        self.bayesian_result[meas_type][measurement]["intensity_er"],
                        self.data.datainfo_json[measurement]["label"],
                        measurement,
                    ]
                )
                np.savetxt(f, save_list, delimiter=datadelim, fmt=datafmt)  # Format for parameter output file

                # plot this measurement if plot=True
                if plot:
                    """
                    Plot and save intervals Bayesian inference method analysis:
                    Figure 1:
                        Plot 0 is resulting likelihood function
                        Plot 1 is the evolution of the mean and confidence intervals of the lifetime with trial number.
                    Figure 2:
                        Plot evolution of posterior (or prior) density vs. trial number.
                    """
                    label = self.data.datainfo_json[measurement]["label"]
                    date_time = measurement[0:22]
                    t_mean = mean_array[-1]
                    t_max = max_array[-1]
                    title = f"{label}@{date_time}; $T_{{mean}}$={t_mean:.2f}s, $T_{{mle}}$={t_max:.2f}s"
                    # Confidence interval evolution & peak value
                    fig1, axs = plt.subplots(2)
                    fig1.suptitle(title)
                    axs[0].plot(self.t_values, posterior)
                    axs[0].set(xlabel="Lifetime (s)", ylabel="Likelihood")
                    axs[1].plot(mean_array, "b")
                    axs[1].plot(max_array, "k")
                    axs[1].plot(conf_array[:, 0], "skyblue")
                    axs[1].plot(conf_array[:, 1], "skyblue")
                    axs[1].set(xlabel="Trial", ylabel="Lifetime (s)")
                    plt.tight_layout()
                    if save_plot:
                        pdf1.savefig()  # Save figure
                    else:
                        plt.show()
                    plt.close()

                    # Density plot of evolution of posteriors
                    fig2 = plt.figure()
                    fig2.suptitle(title)
                    plt.imshow(
                        np.transpose(posterior_array),
                        aspect="auto",
                        extent=self.extents(np.arange(len(mean_array))) + self.extents(self.t_values),
                        interpolation="none",
                        origin="lower",
                        cmap=cmap_post_evo,
                    )
                    # Note that extents is of the form [left,right,bottom,top]
                    # and that + operation with lists combines lists like [a,b] + [c,d] = [a,b,c,d]
                    plt.xlabel("Trial")
                    plt.ylabel("Lifetime (s)")
                    plt.tight_layout()
                    if save_plot:
                        pdf2.savefig()
                    else:
                        plt.show()
                    plt.close()

        self.logger.info(f"Bayesian inference method on {meas_type.value} data completed.")
        return

    def bayesian_inference(self, measurement, meas_type: MeasureEnum):
        """Bayesian inference processing of the data in given measurement

        Args:
            measurement (_type_): name of measurement being analyzed
            meas_type (MeasureEnum): intervals or live

        Returns:
            posterior: final posterior
            mean_array: array of mean lifetime
            max_array: array of most likely lifetime
            conf_array: array pf confidence interval
            posterior_array: array of posterior evolution
        """
        trial_list = self.datadict[meas_type][measurement][0]  # event_list in old version
        # trial list has 2 columns:
        # * illumination time (called dissociation time),
        # * successful (1) or unsuccessful (0)
        posterior_list = []  # Recording posterior evolution
        mean_list = []  # Initialize mean lifetime list
        max_list = []  # Initiallzw most likely lifetime list
        conf_list = []  # Initialize confidence interval list
        posterior = np.ones(N_p) / N_p  # Initialize posterior distribution to homogeneous prior
        for t_light, success in trial_list:
            # t_light: illumination time t; success: dissociation is successful (or not)
            # TODO: modify the likelihood to consider the distribution of t_light due to error
            likelihood = self.likelihood(t_light, meas_type, success)
            # likelihood = likelihood * (1 - er) + er  # Update likelihood function with small uniform error rate
            posterior = posterior * likelihood  # Update posterior = prior (old posterior) * likelihood
            posterior = posterior / np.sum(posterior)  # normalization of posterior

            posterior_list.append(posterior)
            mean_list.append(self.mean_t(posterior))
            arg_idx = np.argmax(posterior)  # Index of maximum as the most likely value of lifetime
            max_list.append(self.t_values[arg_idx])
            conf_list.append(self.get_confidence(posterior, conf=confidence))
        mean_array = np.array(mean_list)
        max_array = np.array(max_list)
        conf_array = np.array(conf_list)
        posterior_array = np.array(posterior_list)
        return posterior, mean_array, max_array, conf_array, posterior_array

    def likelihood(self, t_light, meas_type: MeasureEnum, success: bool):
        if meas_type == MeasureEnum.intervals:
            if success:
                return 1 - np.exp(-t_light / self.t_values)
            elif not success:
                return np.exp(-t_light / self.t_values)
        elif meas_type == MeasureEnum.live:
            if success:
                # self.logger.info("likelihood func: success in real-time")
                # Probability density function of successful event is used instead of probability
                # Probability = Probability density * delta_t
                # delta_t is gone after normalization so is replaced by 1000 to reduce error
                return 1000 * np.exp(-t_light / self.t_values) / self.t_values
            elif not success:
                return np.exp(-t_light / self.t_values)
        return 0

    def extents(self, f):
        """Generate extents for matrix/density plot (imshow).

        Args:
            f: list of elements with interval = delta

        Returns:
            extent=[min, max]: [first value in list - 1/2 step size, end value in list + 1/2 step size]
        """
        delta = f[1] - f[0]  # Calculates step size of input list (assuming equally spaced)
        extent = [f[0] - delta / 2, f[-1] + delta / 2]
        return extent

    def mean_t(self, pr: np.ndarray):
        if len(pr) != N_p:
            self.logger.error("Length of the probability function incorrect, failed to calculate mean_t")
            return 0

        return sum([self.t_values[i] * pr[i] for i in range(N_p)])

    def get_confidence(self, posterior, conf=0.68):
        """Calculate confidence intervals in the lifetime and
        track their evolution with the Bayesian inference method.

        Args:
            posterior (list): Probability distribution of posterior
            conf (float, optional): Confidence level. Defaults to 0.68.

        Returns:
            bound: [lower_bound, upper_bound]
        """
        sum_pr = np.cumsum(posterior)
        low_pr = (1 - conf) / 2
        high_pr = conf + low_pr
        idx_low = np.max(np.where(sum_pr < low_pr)[0])
        idx_high = np.min(np.where(sum_pr > high_pr)[0])
        bound = [self.t_values[idx_low], self.t_values[idx_high]]
        return bound  # change tuple to list

    def calc_uncertainty(self, measurement, meas_type: MeasureEnum, conf=0.68):
        """Calculating uncertainty (confidence interval) of intensity and lifetime

        Args:
            measurement (_type_): _description_
            meas_type (MeasureEnum): _description_
            conf (float, optional): _description_. Defaults to 0.68.
        """
        power = self.data.datainfo_json[measurement]["power"]
        waist_x = self.data.datainfo_json[measurement]["FWHM(x)"] / np.sqrt(2 * np.log(2))
        waist_y = self.data.datainfo_json[measurement]["FWHM(y)"] / np.sqrt(2 * np.log(2))
        self.waist_er = waist_er / np.sqrt(2 * np.log(2))
        position_x = 0
        position_y = 0
        post = self.bayesian_result[meas_type][measurement]["probability_distribution"]

        samples_power = np.random.normal(power, power_er(self.do_2photon), N_sample)
        samples_lifetime = np.random.choice(self.t_values, N_sample, p=post)
        with open(f"samples/samples_lifetime/{measurement}_lifetime_{meas_type.value}.npy", "wb") as f:
            np.save(f, samples_lifetime)
        samples_waist_x = np.random.normal(waist_x, self.waist_er, N_sample)
        samples_waist_y = np.random.normal(waist_y, self.waist_er, N_sample)
        samples_position_x = np.random.normal(position_x, position_er, N_sample)
        samples_position_y = np.random.normal(position_y, position_er, N_sample)

        samples_intensity = (
            2
            * samples_power
            * 10**6  # convert from uW/um^2 to W/m^2
            / (np.pi * samples_waist_x * samples_waist_y)
            * np.exp(
                -2 * samples_position_x**2 / samples_waist_x**2
                - 2 * samples_position_y**2 / samples_waist_y**2
            )
        )
        self.bayesian_result[meas_type][measurement]["intensity"] = np.mean(samples_intensity)
        self.bayesian_result[meas_type][measurement]["intensity_er"] = np.std(samples_intensity)
        with open(f"samples/samples_intensity/{measurement}_intensity.npy", "wb") as f:
            np.save(f, samples_intensity)

        return

    def get_success_rate(self, measurement):
        trial_list = self.datadict[MeasureEnum.intervals][measurement][0]
        succ_list = {}
        for t_light, success in trial_list:
            if t_light not in succ_list:
                succ_list[t_light] = {True: 0, False: 0}
            if success:
                succ_list[t_light][True] += 1
            else:
                succ_list[t_light][False] += 1
        succ_rate = {}
        for t_light in succ_list:
            succ_rate[t_light] = succ_list[t_light][True] / (succ_list[t_light][True] + succ_list[t_light][False])
        return dict(sorted(succ_rate.items()))
