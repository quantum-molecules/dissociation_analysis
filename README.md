# Photodissociaition of CaOH+

Data analysis of the data collected in the photodissociation measurement on CaOH+.

## Dependence

- Required python: >=3.8

- Required packages: numpy, scipy, pandas, matplotlib

## Use

1. Create a new json file for specifying which files to be included in this task in `./data_info_json`
2. Copy the following files from remote folder to a new folder in `./input` manually or by setting `ProcessData(copy_files=True)`:
   - `date_time_dissociation_measurement.txt`
   - `date_time_dissociation_measurement_data.txt`
   - `date_time_measurement_info.txt`
   - `date_time_real_time_dissociation.txt`
  
   **This will overwrite the post-processed data files!**

3. Set `ProcessData(process_y_line_sum=False)` when initializing the data processing class to generate plots for y_line_sum data in `./output/plots/task_name`.
4. Post-process the data by going through the y_line_sum plots to filter out mark problematic trials (setting corresponding `use_data=0` in data files) when or after:
   - **any dark ion is generated when applying OPA light, including e.g. dark ion reappearing after disappearing in the trap.**

- *for analysing single photon dissociation data:*
  
  `data_analysis.ipynb`

- *for analysing two photon dissociation data:*
  
  `data_analysis_2photon.ipynb`

- *for checking power dependence of single photon dissociation rate:*
  
  `data_analysis_power_dependence.ipynb`

- *for checking power dependence of 2-photon dissociation rate:*
  
  `data_analysis_power_dependence_2photon.ipynb`

- *for generating plots of the mass spectrometry data:*
  
  `mass_spec.ipynb`

- *for calculating the transform-limited pulse duration from spectra:*
  
  `pulse_duration_from_spectrum.ipynb`

## Directory Hierarchy

- `./data_info_json` : json files for specifying which measurement data are included for a task
- `./input` : data files copied from QOS drive(`.txt`), not including y_line_sum data to save space
- `./output` : processed data output, including Bayesian processing plots and y_line_sum plots
- `./autocorrelation`: analysis of the autocorrelation data
- `./opa_spectrum`: raw data of the measured OPA spectrum, autocorrelation measurements
- `./samples` : Monte Carlo sampling data files

## For Developement

- `processing.py` : for processing real-time/interval data from the measurement
- `readin.py` : for reading real-time/interval data from local files, or reading y_line_sum data into pdfs from remote server, or copying remote real-time/interval data to local `./output` folder
- `logger.py` : for handling logging into `dissociation_analysis.py`
- `constants.py` : all constants related to data reading and processing
