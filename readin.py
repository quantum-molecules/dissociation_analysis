import json
import shutil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import getlogin
from getpass import getuser
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path
from logger import Logger
from matplotlib.backends.backend_pdf import PdfPages
from constants import experiment, diss_access, outplotdir, datainfofile, logfile, indatadir, ylinesumfile, imgprocfile, cmap_y_line_sum

matplotlib.use("pgf")
matplotlib.rc("text", usetex=True)


class ReadError(Exception):
    def __init__(self, msg: str) -> None:
        self.msg = msg
        super().__init__(self.msg)


class InputData:
    def __init__(self, task) -> None:
        """Generate input data from files
        Data information obtained from json files in ./data_info_json
        Additional info/real time data/interval data read from local files in ./input
        Results stored in:
            self.datainfo_json
            self.final_result_liv
            self.final_result_int

        * Use local data in ./input, otherwise:
            1. copy data from QOS manually or run copy_from_qos()
            2. change the data_info_json file accordingly
        * Run read() to start reading data
        * Run parse_y_line_sum() to process y_line_sum data into pdfs for post-processing
        """
        self.logger = Logger("data_input", logfile).get_logger()

        self.task = task

        self.isRead = False  # mark whether or not self.read() is done

        self.remoteOK = False  # mark whether connection to QOS is checked to be OK

        self.result_int = {}
        """result dictionary for intervals data

        self.result_int[measurement] = [trial_list, diss_dict, no_diss_dict, ratio_return, int_data_array]
        """
        self.result_liv = {}
        """result dictionary for live data

        self.result_liv[measurement] = [trial_list, diss_list, no_diss_list, liv_data_array]
        """

        with open(datainfofile(self.task)) as fp:
            self.datainfo_json = json.load(fp)  # exclude_dict in old version
            """information dictionary for each measurement"""

    def read(self, process_y_line_sum=False):
        """
        Method for read in data for intervals and live dissociation measurement
        Excluding events based on datainfo_json.
        """
        self.isRead = False
        self.logger.info("Reading in data files...")

        # make local input file directory
        indatadir(self.task).mkdir(parents=True, exist_ok=True)

        # if going to process y_line_sum data, check first if connection to qos works
        if process_y_line_sum:
            if not self._connect_qos():
                self.logger.error("Cannot process y_line_sum data!")
                return

        for measurement in self.datainfo_json.keys():
            try:
                self.logger.info(f"Start parsing data @{measurement[0:22]}:")
                self.read_info_dict(
                    measurement
                )  # additional information in measurement_info.txt, which is entered when starting the measurement
                self.parse_int_data(measurement)  # read into self.result_int
                self.parse_liv_data(measurement)  # read into self.result_liv
                if process_y_line_sum and self.remoteOK:
                    self.parse_y_line_sum(measurement)
            except ReadError as e:
                self.logger.error(e)
                return

        self.isRead = True

    def _connect_qos(self):
        try:
            user = getlogin()
        except Exception:
            user = getuser()
        try:
            config = ConfigParser()
            config.read("cfg.ini")
            qos_paths = dict(config.items("QOS paths"))
            qos_path = Path(qos_paths[user])
            self.indatadir_qos = qos_path.joinpath("molecules").joinpath("data")
            access_file = self.indatadir_qos.joinpath(diss_access)
            with open(access_file, "a+") as f:
                f.write(f"{datetime.now()}: access on from {user} for {experiment}:{self.task}\n")
            self.logger.info(f"Successfully accessed the QOS server with username: {user}.")
            self.remoteOK = True
        except OSError:
            self.logger.error("Tried connecting to qos but failed to get qos path...")
            self.remoteOK = False
            return False
        return True

    def copy_files_from_qos(self):
        """copy txt files from each measurement folder to local directory (replacing the old files!)"""
        if not self._connect_qos():
            self.logger.error("Cannot copied files from qos! Local data will be used.")
            return

        self.logger.info("Copying data files from qos based on datainfo_json...")
        for measurement in self.datainfo_json.keys():
            year = measurement[0:4]
            year_month = measurement[0:7]
            middir = f"{year}/{year_month}"
            folderpath = self.indatadir_qos.joinpath(middir).joinpath(measurement)  # remote directory
            # copy all txt files, replace old files
            for file in folderpath.glob("*.txt"):
                shutil.copy2(file, indatadir(self.task))

            # add number of events to info file
            # with open(indatadir.joinpath(f"{measurement[0:22]}_measurement_info.txt"), "a+") as f:
            #     f.write(f"number of events    {len(list(folderpath.glob('event_*')))}\n")

    def read_info_dict(self, measurement):
        """Read info from info files into datainfo_json
        Note: only for reference, info in datainfo_json is prioritized
        Args:
            measurement: name of the measurement being processed

        Raises:
            ReadError: cannot access file
        """
        # read info array from given measurement folder
        # add additional info from datainfo_json to info_dict
        try:
            filename = f"{measurement[0:22]}_measurement_info.txt"  # "DATETIME_measurement_info.txt"
            filepath = indatadir(self.task).joinpath(filename)

            self.logger.info("reading measurement info...")
            # read info dict from info file, update datainfo_json
            # NOTE: wavelength/power/pp_divider values from the json file would be used in the data processing
            data = pd.read_csv(filepath, sep="    ", index_col=0, header=None, engine="python").T
            self.datainfo_json[measurement].update(
                data.to_dict("index")[1]  # change dataFrame (only one row) to dictionary
            )
            # waist_x = self.datainfo_json[measurement]["FWHM(x)"] / np.sqrt(2 * np.log(2))
            # waist_y = self.datainfo_json[measurement]["FWHM(y)"] / np.sqrt(2 * np.log(2))
            # self.datainfo_json[measurement]["intensity"] = (
            #     2 * self.datainfo_json[measurement]["power"] * 10**6 / (np.pi * waist_x * waist_y)
            # )

        except OSError:
            raise ReadError(f"Failed to read from {filepath}!")

    def parse_int_data(self, measurement):
        self.logger.info("loading intervals data...")

        int_data_array = self.read_data_array(
            indatadir(self.task).joinpath(f"{measurement}.txt"), sep=",", header=0
        )  # event_array in old version

        exclude_events = self.datainfo_json[measurement]["exclude_events"]
        # Events to exclude in the data file that are manually entered in the json file
        diss_times = np.unique(int_data_array[:, 2])
        # Array which stores the different illumination times for which dissociation was attempted
        # (called diss_exposure_time_[s] in file)
        no_diss_dict = {diss_time: 0 for diss_time in diss_times}
        # Initialize no dissociation dictionary.
        # This dictionary stores the number of times dissociation was not successful for a given illumination time.
        diss_dict = {diss_time: 0 for diss_time in diss_times}
        # Initialize dissociation dictionary.
        # This dictionary stores the number of times dissociation was successful for a given illumination time.
        trial_list = []  # event_list in old version
        # Initialize event list. This list stores [dissociation time (really the illumination time), success (0 or 1)]
        # for each trial.
        printskip = 0
        # make sure we only print "skipping event" once per event rather than once per line

        # Populate diss, no_diss, and trial_list and skip events for which data should be excluded
        for num, line in enumerate(int_data_array):
            if int(line[0]) in exclude_events:
                # Print in terminal that an event is being excluded if it is in the exclude_events list
                # only print once for each event
                if printskip < int(line[0]):
                    self.logger.info(f"Skipping event {int(line[0])} in the excluded event list")
                    printskip = int(line[0])
            elif line[8] == 0:  # use_data = 0, not using the data in this line
                self.logger.info(f"Skipping line {num} where use_data = 0")
            elif line[8] == 1:  # use_data = 1, use data in this line
                trial_list.append([line[2], line[3]])
                if line[3] == 1.0:  # diss_succes = 1
                    diss_dict[line[2]] += 1
                if line[3] == 0.0:  # diss_succes = 0
                    no_diss_dict[line[2]] += 1
            else:
                raise ReadError(f"Wrong use_data value while loading intervals data: {measurement}")

        # calculate success ratio and uncertainty
        ratio = {diss_time: 0 for diss_time in diss_times}
        uncertainty = {diss_time: 0 for diss_time in diss_times}
        for diss_time in diss_times:
            num_trials = diss_dict[diss_time] + no_diss_dict[diss_time]
            if num_trials != 0:
                ratio[diss_time] = diss_dict[diss_time] / num_trials
                # assign uncertainty
                if diss_dict[diss_time] == 0 or no_diss_dict[diss_time] == 0:
                    prob_hedged = 1 / (num_trials + 1)  # Replace  zero uncertainty with hedging!
                    uncertainty[diss_time] = np.sqrt(prob_hedged * (1 - prob_hedged) / num_trials)
                else:
                    uncertainty[diss_time] = np.sqrt(ratio[diss_time] * (1 - ratio[diss_time]) / num_trials)

            else:
                # Check for dissociation times with no trials and remove keys from ratio and uncertainty dicts
                # (this is necessary to prevent divide by zero error in calculating ratio)
                ratio.pop(diss_time, None)
                uncertainty.pop(diss_time, None)
        ratio_return = np.array([[diss_time, ratio[diss_time], uncertainty[diss_time]] for diss_time in ratio.keys()])
        # Compute ratio return:
        # column 0 is illumination time (called dissociation time),
        # column 1 is probability of success,
        # column 2 is uncertainty

        self.result_int[measurement] = [
            trial_list,
            diss_dict,
            no_diss_dict,
            ratio_return,
            int_data_array,
        ]  # intervals results to return, indexed by file

    def parse_liv_data(self, measurement):
        self.logger.info("loading live data...")

        liv_data_array = self.read_data_array(
            indatadir(self.task).joinpath(f"{measurement[0:22]}_real_time_dissociation.txt"), sep="    ", header=0
        )  # event_array in old version

        exclude_events = self.datainfo_json[measurement]["exclude_events"]
        # Events to exclude in the data file that are manually entered in the json file
        no_diss_list = []  # nodiss_list in old version
        # Initialize illumination time list which resulted in no dissociation.
        # This list stores the cumulative time that a dark ion was illuminated by the OPA
        # before the event was stopped for some reason.
        diss_list = []
        # Initialize illumination time list which resulted in dissociation.
        # This list stores the cumulative time that a dark ion was illuminated by the OPA
        # before it changed to a bright ion in a given event.
        trial_list = []  # diss_array in old version
        # Initialize trial list where column 0 is the exposure time and column 1 is the success value (0 or 1)
        printskip = 0
        # make sure we only print "skipping event" once per event rather than once per line

        # Populate diss, no_diss, and trial_list and skip events for which data should be excluded
        for num, line in enumerate(liv_data_array):
            if int(line[0]) in exclude_events:
                # Print in terminal that an event is being excluded if it is in the exclude_events list
                # only print once for each event
                if printskip < int(line[0]):
                    self.logger.info(f"Skipping event {int(line[0])} in the excluded event list")
                    printskip = int(line[0])
            elif line[5] == 0:  # use_data = 0, not using the data in this line
                self.logger.info(f"Skipping event {num} where use_data = 0")
            elif line[5] == 1:  # use_data = 1, use data in this line
                # TODO: rules out weird events here
                if line[1] >= 0 and line[2] >= 0:  # some data still keep default negative diss_time/accum_diss_time
                    if line[4] == 1.0:  # diss_succes = 1
                        diss_list.append(line[2])
                        trial_list.append([line[2], line[4]])  # use accumulative_diss_time
                    if line[4] == 0.0:  # diss_succes = 0
                        no_diss_list.append(line[3])
                        trial_list.append([line[3], line[4]])  # use accumulative_exposure_time
            else:
                raise ReadError(f"Wrong use_data value while loading live data: {measurement}")

        # TODO: sorted diss_list/no_diss_list or not?
        # diss_list_sorted = copy.deepcopy(diss_list)  # Copy list
        # nodiss_list_sorted = copy.deepcopy(nodiss_list)  # Copy list
        # diss_list_sorted.sort()  # Sort diss_list (in ascending order)
        # if len(nodiss_list) != 0:
        #     nodiss_list_sorted.sort()  # Sort nodiss_list (in ascending order)

        self.result_liv[measurement] = [
            trial_list,
            diss_list,
            no_diss_list,
            liv_data_array,
        ]  # intervals results to return, indexed by file

    def read_data_array(self, file, sep, header=None):
        """
        Args:
            file: path-like, data file

        Raises:
            ReadError: cannot access file

        Returns:
            data_array
        """
        try:
            with open(file, mode="r", buffering=-1, encoding="utf-8") as file:
                data_array = pd.read_csv(file, sep=sep, header=header, engine="python").values
            return data_array
        except OSError:
            raise ReadError(f"Failed to read from {file}!")

    def parse_y_line_sum(self, measurement):
        """
        Method for inspecting y_line_sum data for post-processing.
        """
        year = measurement[0:4]
        year_month = measurement[0:7]
        date_time = measurement[0:22]
        middir = f"{year}/{year_month}"
        folderpath = self.indatadir_qos.joinpath(middir).joinpath(measurement)
        outputpdf = outplotdir(self.task).joinpath(f"{date_time}_y_line_sum.pdf")
        self.logger.info("parsing y_line_sum to pdf...")
        n_events = self.datainfo_json[measurement]["last_event"]
        with PdfPages(outputpdf) as pdf:
            for i in range(n_events):
                event_path = folderpath.joinpath(f"event_{i+1}")
                ylinesum_data = self.read_data_array(event_path.joinpath(ylinesumfile), sep="    ", header=None)
                imgproc_data = self.read_data_array(event_path.joinpath(imgprocfile), sep="    ", header=0)
                event_start_time = imgproc_data[0][2]
                event_time = imgproc_data[:, 2] - event_start_time

                fig, axs = plt.subplots(4, sharex=True)
                fig.suptitle(f"event: {i+1}/{n_events}")

                zv = ylinesum_data.transpose()
                pixel_y = np.arange(len(zv))
                tv, yv = np.meshgrid(event_time, pixel_y)
                axs[0].pcolormesh(tv, yv, zv, vmin=zv.min(), vmax=zv.max(), rasterized=True, cmap=cmap_y_line_sum)
                axs[0].set_yticks([])
                axs[0].set_ylabel(r"$\sum$y image")

                axs[1].plot(event_time, np.sum(zv, axis=0), "k-", label="Integrated image", rasterized=True)
                axs[1].set_yticks([])
                axs[1].set_ylabel(r"$\sum$xy image")

                axs[2].plot(event_time, imgproc_data[:, 0], c="C1", label="Bright ions", rasterized=True)
                axs[2].plot(event_time, imgproc_data[:, 1], c="C0", label="Dark ions", rasterized=True)
                axs[2].set_yticks([])
                axs[2].set_ylabel("N ions")

                axs[3].plot(event_time, imgproc_data[:, 3], c="C3", label="Empty", rasterized=True)
                axs[3].plot(event_time, imgproc_data[:, 4], c="C4", label="Cloud", rasterized=True)
                axs[3].plot(event_time, imgproc_data[:, 5], c="C2", label="OPA", rasterized=True)
                axs[3].set_yticks([0, 1])
                axs[3].set_ylabel("Status")
                axs[3].set_xlabel("Time (s)")

                pdf.savefig()
                plt.close()
        return
