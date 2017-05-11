import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.colors import LogNorm
import ipywidgets as ipy
from IPython.display import display
import flatfile_3 as ff

# Information about the "stm_analysis.py" module
__version__ = "2.00"
__date__ = "10th May 2017"
__status__ = "Pending"

__authors__ = "Procopi Constantinou & Tobias Gill"
__email__ = "procopios.constantinou.16@ucl.ac.uk"

# 0 - Defining some physical constants that may be used throughout the analysis
PC = {"c": 2.99792458e8, "e": 1.6021773e-19, "me": 9.109389e-31, "kB": 1.380658e-23,
      "h": 6.6260755e-34, "hbar": 1.05457e-34, "eps0": 8.85419e-12,
      "pico": 1e-12, "nano": 1e-9, "micro": 1e-6}


# 1 - Defining the class object to select the parent directory to browse through all the stm data
class DataSelection(object):
    def __init__(self, dir_path):
        """
        Defines the initialisation of the class object.
        dir_path:   String of the full path to the ..../stm_project/0_stm_data/' directory.
        """
        # 1.1 - Extract the path to each one of the directories that holds data
        self.dir_path = dir_path                            # A string of the file path to the data directory
        self.full_dir_list = glob.glob(dir_path + '*')      # List of all the folders with their full directory paths
        self.num_of_dir = len(self.full_dir_list)           # Total number of folders in the '.../0_stm_data/' folder
        # 1.2 - Extract the titles of each folder that holds data
        self.folder_list = None                             # A list of the last 6 characters of each folder loaded
        self.get_titles()                                   # Function to get all titles from '.../0_stm_data/' folder
        # 1.3 - Provide a continuous, interactive update to the data folder chosen by the user
        self.selected_folder = None                         # String of the last 6 characters of the folder chosen
        self.selected_path = None                           # Full path to the data folder chosen
        self.widgets = None                                 # Widget object that holds all the pre-defined widgets
        self.get_widgets()                                  # Function to get all of the pre-defined widgets
        self.output = None                                  # Output to the user interaction with the widgets
        self.user_interaction()                             # Function to allow the continuous user interaction

    def get_titles(self):
        """
        Extract all of the folders from the full path to the '.../0_stm_data/' directory.
        """
        all_dates = list()
        for i in range(self.num_of_dir):
            # Finding the length of the directory path given by the user
            num = len(self.dir_path)
            # Only extract the last 6 characters from the folder name and omit the directory path
            date = self.full_dir_list[i][num:]
            all_dates.append(date)
        self.folder_list = all_dates

    def get_widgets(self):
        """
        Creates a variety of widgets to be interacted with.
        """
        # Select Multiple widget to select the the flat-files to be analysed
        directory_select = ipy.ToggleButtons(options=self.folder_list, description="$Choose$ $data$:",
                                             value=self.folder_list[0],
                                             layout=ipy.Layout(display='flex', flex_flow='row',
                                                               align_items='stretch', align_content='stretch',
                                                               width='100%', height='', justify_content='center'))
        # Defining a global widget box to hold all of the widgets
        self.widgets = directory_select

    def update_function(self, option):
        """
        Updates the printed text to read the user-selected folder and all it's content.
        """
        # Define an attribute that has the date of the user selected folder
        self.selected_folder = option
        # Define an attribute that has the full file path to the user selected folder
        self.selected_path = self.full_dir_list[self.folder_list.index(self.selected_folder)] + "/"
        # Count the total number of different files within the directories
        total_topo_files = len(glob.glob(self.selected_path + '*.Z_flat'))
        total_iv_files = len(glob.glob(self.selected_path + '*.I(V)_flat'))
        total_iz_files = len(glob.glob(self.selected_path + '*.I(Z)_flat'))
        # Print out all the necessary information
        print(self.dir_path)
        print(" " + str(self.selected_folder) + " directory")
        print("\t" + str(total_topo_files) + "\t topography files.")
        print("\t" + str(total_iv_files) + "\t I(V) spectroscopy files.")
        print("\t" + str(total_iz_files) + "\t I(z) spectroscopy files.")

    def user_interaction(self):
        """
        Function that displays the widgets, whilst allowing their continuous interaction with the update function and 
        finally displaying the outcome of the interaction.
        """
        # Display the widgets
        display(self.widgets)
        # Interact with the 'update_function' using the widgets
        self.output = ipy.interactive(self.update_function, option=self.widgets)
        # Display the final output of the widget interaction
        display(self.output.children[-1])


# 2.0 - Defining the class object that will import the I(V) flat-files and perform all the necessary analysis
class STS(object):
    def __init__(self, DS):
        """
        Defines the initialisation of the class object.
        DS:     The 'DataSelection' class object.
        """
        # 2.1 -  Extract all the flat-files from the data directory selected
        self.flat_files = glob.glob(DS.selected_path + '*.I(V)_flat')   # List of all the I(V) flat file paths
        self.num_of_files = len(self.flat_files)                        # Total number of I(V) flat files loaded
        self.file_alias = None                                          # List of unique identifiers to I(V) flat files
        self.all_flatfile_extract()

        # 2.2 - Defining all the attributes associated with the I(V) file selection
        self.selected_files = None                          # List of the selected I(V) aliases
        self.num_of_selected_files = None                   # Total number of I(V) flat files selected
        self.selected_pos = None                            # List of the array positions of the I(V) files
        self.selected_data = None                           # List of the selected I(V) flat file classes
        self.selected_v_dat = None                          # List of the selected I(V) voltage data domains
        self.selected_i_dat = None                          # List of the selected I(V) current data ranges

        # 2.3 - Passing the selected files through the sts analysis functions
        # 2.3.1 Cross-correlation analysis attributes
        self.xcorr_info = None                              # Dictionary with all the cross-correlation info
        self.xcorr_v_dat = None                             # List of cross-correlated I(V) voltages
        self.xcorr_i_dat = None                             # List of cross-correlated I(V) currents
        self.v_outliers = None                              # 1D array of the outlying voltage points
        self.i_outliers = None                              # 1D array of the outlying current points
        # 2.3.2 Cropped voltage domain attributes
        self.xcrop_v_dat = None                             # Cross-correlated, cropped I(V) voltage list
        self.xcrop_i_dat = None                             # Cross-correlated, cropped  I(V) current list
        # 2.3.3 STS analysis attributes
        self.avg_i_data = None                              # Average I(V) curve over all selected files
        self.avgsq_i_data = None                            # Average of the squared I(V) curves over all selected files
        self.smooth_i_data = None                           # Smoothed I(V) curves of all selected files
        self.smooth_avg_i_data = None                       # Smoothed version of the average I(V) curve
        self.smooth_avgsq_i_data = None                     # Smoothed version of the average of the squares I(V) curve
        self.didv_data = None                               # Derivative of all smoothed I(V) curves in a 2D array form
        self.didv_avg_data = None                           # Derivative of the average I(V) curve
        self.didv_avgsq_data = None                         # Derivative of the average of the squares I(V) curve
        self.i_var = None                                   # Variance/uncertainty in the best estimation of dI/dV
        # 2.3.4 User interaction
        self.widgets = None                                 # Widget object to hold all pre-defined widgets
        self.get_widgets()                                  # Function to get all of the pre-defined widgets
        self.output = None                                  # Output to the user interaction with widgets
        self.user_interaction()                             # Function to allow continuous user interaction

    def all_flatfile_extract(self):
        """
        Function to extract the file names and total number of I(V) flat-files within the given directory.
        """
        # Initialising the variables to be used
        file_alias = list()
        file_num, scan_num, cond = 0, 0, True
        # Run a while loop until all the data is loaded
        while cond:
            scan_num += 1
            # Run a for-loop over a total of 20 repeats (if more repeats than this are taken, it will need changing)
            for repeat in range(20):
                # Define the file name to be searched through
                fname = "Spectroscopy--" + str(scan_num) + "_" + str(repeat) + ".I(V)_flat"
                # If the file name is found, save it and add one unit to the break point
                if len([x for x in self.flat_files if fname in x]) == 1:
                    # Making the file name consistent
                    if scan_num < 10:
                        file_alias.append("sts 00" + str(scan_num) + "_" + str(repeat))
                    elif scan_num < 100:
                        file_alias.append("sts 0" + str(scan_num) + "_" + str(repeat))
                    else:
                        file_alias.append("sts " + str(scan_num) + "_" + str(repeat))
                    # Add one to the file number
                    file_num += 1
                if file_num == self.num_of_files:
                    cond = False
        # Return the unique identifiers to each I(V) flat file
        self.file_alias = file_alias

    def selected_data_extract(self):
        """
        Function to extract the I(V) data from the user selected flat-files.
        """
        # Finding the total number of selected files
        self.num_of_selected_files = len(self.selected_files)

        # Extract all of the array positions of the I(V) files selected
        if len(self.selected_files) < 1:
            print("No data has been selected.")
        else:
            arg_list = list()
            for i in range(len(self.selected_files)):
                arg_list.append(self.file_alias.index(self.selected_files[i]))
            self.selected_pos = arg_list

        # Extract all of the I(V) raw data from the selected flat-files by using the flat-file load function
        # - Run a for-loop over the all the I(V) flat-files that are parsed
        sts_data = list()
        for pos in self.selected_pos:
            sts_data.append(ff.load(self.flat_files[pos]))
        # Return the necessary attribute
        self.selected_data = sts_data

        # Extract the voltage and current data for all the I(V) flat files that are parsed in a numpy.array form
        v_data, i_data = list(), list()
        min_v, max_v = list(), list()
        # - Run a for-loop over the all the I(V) flat-files that are parsed
        for i in range(self.num_of_selected_files):
            all_i_data = list()
            # Extracting the trace I(V) data as the first argument whilst omitting first and last 5 points
            all_i_data.append(self.selected_data[i][0].data[5:-5])
            # Extracting the retrace I(V) data as the second argument whilst omitting first and last 5 points
            all_i_data.append(self.selected_data[i][1].data[5:-5])
            i_data.append(all_i_data)
            # Extracting the voltage domain of the I(V) data
            v_start = self.selected_data[i][0].info['vstart']
            v_res = self.selected_data[i][0].info['vres']
            v_inc = self.selected_data[i][0].info['vinc']
            v_end = v_start + v_res * v_inc
            v_data.append(np.arange(v_start, v_end, v_inc)[5:-5])
            max_v.append(np.max(np.arange(v_start, v_end, v_inc)[5:-5]))
            min_v.append(np.min(np.arange(v_start, v_end, v_inc)[5:-5]))
        # - Assigning the voltage and current data to attributes of the class
        self.selected_v_dat = v_data
        self.selected_i_dat = i_data
        # Collating all of the cross-correlation information into a dictionary
        self.xcorr_info = {}
        self.xcorr_info['Vmax'] = np.min(max_v)
        self.xcorr_info['Vmax arg'] = np.argmin(max_v)
        self.xcorr_info['Vmin'] = np.max(min_v)
        self.xcorr_info['Vmin arg'] = np.argmax(min_v)

    def selected_data_cross_correlation(self):
        """
        Function to cross-correlate all of the I(V) curves so that they are all defined over a consistent voltage 
        domain to ensure they are ready for analysis.
        """
        # Defining the arrays to hold the cross-correlated and outlier data
        v_xcorr, i_xcorr = list(), list()
        v_outliers = np.array([])
        i_outliers = np.array([])

        # Run a for-loop over the all the I(V) selected
        for i in range(self.num_of_selected_files):
            # - If the maximum and minimum positions are identical, then the voltage domain is over one I(V) curve
            if self.xcorr_info['Vmax arg'] == self.xcorr_info['Vmin arg']:
                # Finding the cross-correlated I(V) curves
                # - Defining the cross-correlated voltage domain
                v_temp = self.selected_v_dat[self.xcorr_info['Vmax arg']]
                # - Use linear interpolation to determine the current over the cross-correlated voltage domain
                i_temp = list()
                i_temp.append(np.interp(v_temp, self.selected_v_dat[i], self.selected_i_dat[i][0]))
                i_temp.append(np.interp(v_temp, self.selected_v_dat[i], self.selected_i_dat[i][1]))
                # - Append the cross-correlated voltage and current data
                v_xcorr.append(v_temp)
                i_xcorr.append(i_temp)

                # Finding the outliers
                # - Temporarily define the voltage and current data for cross-correlation
                v_temp = self.selected_v_dat[i]
                i_temp_trace = self.selected_i_dat[i][0]
                i_temp_retrace = self.selected_i_dat[i][1]
                # - Finding the upper and lower limit outliers
                v_upper = v_temp[v_temp > self.xcorr_info['Vmax']]
                v_lower = v_temp[v_temp < self.xcorr_info['Vmin']]
                i_upper_trace = i_temp_trace[v_temp > self.xcorr_info['Vmax']]
                i_lower_trace = i_temp_trace[v_temp < self.xcorr_info['Vmin']]
                i_upper_retrace = i_temp_retrace[v_temp > self.xcorr_info['Vmax']]
                i_lower_retrace = i_temp_retrace[v_temp < self.xcorr_info['Vmin']]
                # - Appending the outliers to their relevant numpy arrays
                v_outliers = np.append(v_outliers, v_lower)
                v_outliers = np.append(v_outliers, v_upper)
                v_outliers = np.append(v_outliers, v_lower)
                v_outliers = np.append(v_outliers, v_upper)
                i_outliers = np.append(i_outliers, i_lower_trace)
                i_outliers = np.append(i_outliers, i_upper_trace)
                i_outliers = np.append(i_outliers, i_lower_retrace)
                i_outliers = np.append(i_outliers, i_upper_retrace)

            # - If the maximum and minimum positions are different, then the voltage domain is over a two I(V) curves
            else:
                # Finding the cross-correlated I(V) curves
                # - Defining the cross-correlated voltage domain
                v_lower = self.selected_v_dat[self.xcorr_info['Vmin arg']]
                v_upper = self.selected_v_dat[self.xcorr_info['Vmax arg']]
                npts = len(v_upper + v_lower) / 2.
                v_temp = np.linspace(self.xcorr_info['Vmin'], self.xcorr_info['Vmax'], npts)
                # - Use linear interpolation to determine the current over the cross-correlated voltage domain
                i_temp = list()
                i_temp.append(np.interp(v_temp, self.selected_v_dat[i], self.selected_i_dat[i][0]))
                i_temp.append(np.interp(v_temp, self.selected_v_dat[i], self.selected_i_dat[i][1]))
                # - Append the cross-correlated voltage and current data
                v_xcorr.append(v_temp)
                i_xcorr.append(i_temp)

                # Finding the outliers
                # - Temporarily define the voltage and current data for cross-correlation
                v_temp = self.selected_v_dat[i]
                i_temp_trace = self.selected_i_dat[i][0]
                i_temp_retrace = self.selected_i_dat[i][1]
                # - Finding the upper and lower limit outliers
                v_upper = v_temp[v_temp > self.xcorr_info['Vmax']]
                v_lower = v_temp[v_temp < self.xcorr_info['Vmin']]
                i_upper_trace = i_temp_trace[v_temp > self.xcorr_info['Vmax']]
                i_lower_trace = i_temp_trace[v_temp < self.xcorr_info['Vmin']]
                i_upper_retrace = i_temp_retrace[v_temp > self.xcorr_info['Vmax']]
                i_lower_retrace = i_temp_retrace[v_temp < self.xcorr_info['Vmin']]
                # - Appending the outliers to their relevant numpy arrays
                v_outliers = np.append(v_outliers, v_lower)
                v_outliers = np.append(v_outliers, v_lower)
                v_outliers = np.append(v_outliers, v_upper)
                v_outliers = np.append(v_outliers, v_upper)
                i_outliers = np.append(i_outliers, i_lower_trace)
                i_outliers = np.append(i_outliers, i_upper_trace)
                i_outliers = np.append(i_outliers, i_lower_retrace)
                i_outliers = np.append(i_outliers, i_upper_retrace)
        # Assign the values to the attributes
        self.xcorr_v_dat = v_xcorr
        self.xcorr_i_dat = i_xcorr
        self.v_outliers = v_outliers
        self.i_outliers = i_outliers

    def selected_data_crop(self, vbias_limits):
        """
        Crop the raw data of the I(V) spectroscopy curves over the given voltage bias limits.
            vbias_limits:   An np.array([X, Y]) where X and Y are the lower and upper voltage bias limits respectively.
        """
        # Defining the arrays to hold the cross-correlated and outlier data
        v_crop, i_crop = list(), list()
        v_outliers = np.array([])
        i_outliers = np.array([])
        # Run a for-loop over the all the selected I(V) curves
        for i in range(self.num_of_selected_files):
            # Finding the cropped I(V) curves
            # - Cropping the data over the lower voltage bias limit
            v_temp = self.xcorr_v_dat[i][self.xcorr_v_dat[i] > vbias_limits[0]]
            i_temp_trace = self.xcorr_i_dat[i][0][self.xcorr_v_dat[i] > vbias_limits[0]]
            i_temp_retrace = self.xcorr_i_dat[i][1][self.xcorr_v_dat[i] > vbias_limits[0]]
            # - Cropping the data over the upper voltage bias limit
            i_temp_trace = i_temp_trace[v_temp < vbias_limits[1]]
            i_temp_retrace = i_temp_retrace[v_temp < vbias_limits[1]]
            v_temp = v_temp[v_temp < vbias_limits[1]]
            # - Append the cropped voltage and current data
            i_temp = list()
            i_temp.append(i_temp_trace)
            i_temp.append(i_temp_retrace)
            v_crop.append(v_temp)
            i_crop.append(i_temp)

            # Finding the outliers
            # - Re-defining the temporary data
            v_temp = self.xcorr_v_dat[i]
            i_temp_trace = self.xcorr_i_dat[i][0]
            i_temp_retrace = self.xcorr_i_dat[i][1]
            # - Finding the upper and lower limit outliers
            i_upper_trace = i_temp_trace[v_temp > vbias_limits[1]]
            i_upper_retrace = i_temp_retrace[v_temp > vbias_limits[1]]
            i_lower_trace = i_temp_trace[v_temp < vbias_limits[0]]
            i_lower_retrace = i_temp_retrace[v_temp < vbias_limits[0]]
            v_upper = v_temp[v_temp > vbias_limits[1]]
            v_lower = v_temp[v_temp < vbias_limits[0]]
            # - Appending the outliers to their relevant numpy arrays
            v_outliers = np.append(v_outliers, v_lower)
            v_outliers = np.append(v_outliers, v_lower)
            v_outliers = np.append(v_outliers, v_upper)
            v_outliers = np.append(v_outliers, v_upper)
            i_outliers = np.append(i_outliers, i_lower_trace)
            i_outliers = np.append(i_outliers, i_lower_retrace)
            i_outliers = np.append(i_outliers, i_upper_trace)
            i_outliers = np.append(i_outliers, i_upper_retrace)
        # Assign the values to the attributes
        self.xcrop_v_dat = v_crop
        self.xcrop_i_dat = i_crop
        self.v_outliers = np.append(self.v_outliers, v_outliers)
        self.i_outliers = np.append(self.i_outliers, i_outliers)

    def sts_analysis(self, retrace="Both", smooth_type="Binomial", smooth_order=3):
        """
        Full STS analysis of the I(V) spectroscopy curves, including; (i) averaging, (ii) smoothing, 
        (iii) differentiation and (iv) variation in the dIdV curves.
        """
        # 1 - Finding the average of the selected I(V) (and I(V) squared for variance calculation)
        # 1.1 If the both traces of I(V) are to be included
        if retrace == "Both":
            # - Finding the mean of the I(V) curves
            trace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][0] for j in range(self.num_of_selected_files)]),
                axis=0)
            retrace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][1] for j in range(self.num_of_selected_files)]),
                axis=0)
            self.avg_i_data = np.mean(np.array([trace_mean, retrace_mean]), axis=0)
            # - Finding the mean of the squared I(V) curves
            trace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][0]**2 for j in range(self.num_of_selected_files)]),
                axis=0)
            retrace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][1]**2 for j in range(self.num_of_selected_files)]),
                axis=0)
            self.avgsq_i_data = np.mean(np.array([trace_mean, retrace_mean]), axis=0)
        # 1.2 If the trace I(V) curve is only selected
        elif retrace == "Trace only":
            # - Finding the mean of the trace I(V) curves
            trace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][0] for j in range(self.num_of_selected_files)]),
                axis=0)
            self.avg_i_data = trace_mean
            # - Finding the mean of the trace squared I(V) curves
            trace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][0] ** 2 for j in range(self.num_of_selected_files)]),
                axis=0)
            self.avgsq_i_data = trace_mean
        # 1.3 If the retrace I(V) curve is only selected
        elif retrace == "Retrace only":
            # - Finding the mean of the retrace I(V) curves
            retrace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][1] for j in range(self.num_of_selected_files)]),
                axis=0)
            self.avg_i_data = retrace_mean
            # - Finding the mean of the retrace squared I(V) curves
            retrace_mean = np.mean(
                np.array([self.xcrop_i_dat[j][1] ** 2 for j in range(self.num_of_selected_files)]),
                axis=0)
            self.avgsq_i_data = retrace_mean

        # 2 - Smoothing the mean and all selected I(V) curves using a chosen smoothing type
        self.smooth_i_data = list()
        # Run a for-loop over all the selected files
        for i in range(self.num_of_selected_files):
            # - If binomial smoothing is required
            if smooth_type == "Binomial":
                import scipy.ndimage.filters as smth
                # For the first run, calculate the smooth of the average I(V) curve
                if i == 0:
                    # Smoothing the average I(V) curve
                    self.smooth_avg_i_data = smth.gaussian_filter(self.avg_i_data, smooth_order)
                    # Smoothing the squared average I(V) curve
                    self.smooth_avgsq_i_data = smth.gaussian_filter(self.avgsq_i_data, smooth_order)
                if retrace == "Trace only":
                    self.smooth_i_data.append(smth.gaussian_filter(self.xcrop_i_dat[i][0], smooth_order))
                elif retrace == "Retrace only":
                    self.smooth_i_data.append(smth.gaussian_filter(self.xcrop_i_dat[i][1], smooth_order))
                elif retrace == "Both":
                    i_holder = list()
                    i_holder.append(smth.gaussian_filter(self.xcrop_i_dat[i][0], smooth_order))
                    i_holder.append(smth.gaussian_filter(self.xcrop_i_dat[i][1], smooth_order))
                    self.smooth_i_data.append(i_holder)
            # - If savgol smoothing is required
            elif smooth_type == "Savitzky-Golay":
                import scipy.signal as smth
                # For the first run, calculate the smooth of the average I(V) curve
                if i == 0:
                    # Smoothing the average I(V) curve
                    self.smooth_avg_i_data = smth.savgol_filter(self.avg_i_data, 51, smooth_order)
                    # Smoothing the squared average I(V) curve
                    self.smooth_avgsq_i_data = smth.savgol_filter(self.avgsq_i_data, 51, smooth_order)
                if retrace == "Trace only":
                    self.smooth_i_data.append(smth.savgol_filter(self.xcrop_i_dat[i][0], 51, smooth_order))
                elif retrace == "Retrace only":
                    self.smooth_i_data.append(smth.savgol_filter(self.xcrop_i_dat[i][1], 51, smooth_order))
                elif retrace == "Both":
                    i_holder = list()
                    i_holder.append(smth.savgol_filter(self.xcrop_i_dat[i][0], 51, smooth_order))
                    i_holder.append(smth.savgol_filter(self.xcrop_i_dat[i][1], 51, smooth_order))
                    self.smooth_i_data.append(i_holder)
            # - If no smoothing is required
            elif smooth_type == "None":
                # For the first run, calculate the smooth of the average I(V) curve
                if i == 0:
                    # Selecting only the average I(V) curve
                    self.smooth_avg_i_data = self.avg_i_data
                    # Smoothing the squared average I(V) curve
                    self.smooth_avgsq_i_data = self.avgsq_i_data
                # Smoothing every I(V) curve selected
                if retrace == "Trace only":
                    self.smooth_i_data.append(self.xcrop_i_dat[i][0])
                elif retrace == "Retrace only":
                    self.smooth_i_data.append(self.xcrop_i_dat[i][1])
                elif retrace == "Both":
                    i_holder = list()
                    i_holder.append(self.xcrop_i_dat[i][0])
                    i_holder.append(self.xcrop_i_dat[i][1])
                    self.smooth_i_data.append(i_holder)

        # 3 - Find the derivative of the mean and all selected I(V) curves (along with the variance)
        # - If only a single file is selected
        if self.num_of_selected_files == 1:
            # - Differentiating the averaged, smoothed I(V) curve
            self.didv_avg_data = np.diff(self.smooth_avg_i_data)
            self.didv_avg_data = self.didv_avg_data + 1.1 * abs(min(self.didv_avg_data))
            # - Differentiating the squared averaged, smoothed I(V) curve
            self.didv_avgsq_data = np.diff(self.smooth_avgsq_i_data)
            self.didv_avgsq_data = self.didv_avgsq_data + 1.1 * abs(min(self.didv_avgsq_data))
            # - Finding the variance by using: Var(X) = [ E(X)^2 - E(X^2) ]
            self.i_var = abs(self.didv_avgsq_data - self.didv_avg_data)
            # The derivative for the rest of the I(V) curves (only 1 in this case, but made in a 2D array for imshow)
            # If the both traces of I(V) are to be included
            if retrace == "Both":
                didv_trace = np.diff(self.smooth_i_data[i][0])
                didv_retrace = np.diff(self.smooth_i_data[i][1])
                didv = np.mean(np.array([didv_trace, didv_retrace]), axis=0)
                didv = didv + 1.1 * abs(min(didv))
                self.didv_data = np.array([didv])
            # If either the trace or retrace I(V) curve is selected
            else:
                didv = np.diff(self.smooth_i_data[i])
                didv = didv + 1.1 * abs(min(didv))
                self.didv_data = np.array([didv])

        # - If multiple files are selected
        else:
            for i in range(self.num_of_selected_files):
                # For the first run, calculate the derivatives of the average I(V) curve
                if i == 0:
                    # - Differentiating the averaged, smoothed I(V) curve
                    self.didv_avg_data = np.diff(self.smooth_avg_i_data)
                    self.didv_avg_data = self.didv_avg_data + 1.1*abs(min(self.didv_avg_data))
                    # - Differentiating the squared averaged, smoothed I(V) curve
                    self.didv_avgsq_data = np.diff(self.smooth_avgsq_i_data)
                    self.didv_avgsq_data = self.didv_avgsq_data + 1.1 * abs(min(self.didv_avgsq_data))
                    # - Finding the variance by using: Var(X) = [ E(X)^2 - E(X^2) ]
                    self.i_var = abs(self.didv_avgsq_data - self.didv_avg_data)
                    # - Initialising the 2D array for all dIdV curves
                    # If the both traces of I(V) are to be included
                    if retrace == "Both":
                        didv_trace = np.diff(self.smooth_i_data[i][0])
                        didv_retrace = np.diff(self.smooth_i_data[i][1])
                        didv = np.mean(np.array([didv_trace, didv_retrace]), axis=0)
                        didv = didv + 1.1 * abs(min(didv))
                        self.didv_data = didv
                    # If either the trace or retrace I(V) curve is selected
                    else:
                        didv = np.diff(self.smooth_i_data[i])
                        didv = didv + 1.1 * abs(min(didv))
                        self.didv_data = didv

                else:
                    # Calculate the derivative for all the rest of the I(V) curves selected
                    # If the both traces of I(V) are to be included
                    if retrace == "Both":
                        didv_trace = np.diff(self.smooth_i_data[i][0])
                        didv_retrace = np.diff(self.smooth_i_data[i][1])
                        didv = np.mean(np.array([didv_trace, didv_retrace]), axis=0)
                        didv = didv + 1.1 * abs(min(didv))
                        self.didv_data = np.vstack([self.didv_data, didv])
                    # If either the trace or retrace I(V) curve is selected
                    else:
                        didv = np.diff(self.smooth_i_data[i])
                        didv = didv + 1.1 * abs(min(didv))
                        self.didv_data = np.vstack([self.didv_data, didv])

    def sts_egap_finder(self, e_gap):
        """
        Determines the effective band-gap with suitable estimates on its uncertainty.
        """
        # Finding the indices for the defined voltage gap selected by the user
        lower_v_index = np.argmin(abs(self.xcrop_v_dat[0][1:] - e_gap[0]))
        upper_v_index = np.argmin(abs(self.xcrop_v_dat[0][1:] - e_gap[1]))
        # Finding the central position of the voltage gap chosen by the user
        v_mean = np.mean(self.xcrop_v_dat[0][1:][lower_v_index:upper_v_index])
        mean_v_index = np.argmin(abs(self.xcrop_v_dat[0][1:] - v_mean))
        # Cropping the voltage and dIdV domain into two halves from the average voltage gap
        # - Cropping the voltage domain
        v_lhs = self.xcrop_v_dat[0][1:][:mean_v_index]
        v_rhs = self.xcrop_v_dat[0][1:][mean_v_index:]
        # - Cropping the dIdV data
        didv_lhs = self.didv_avg_data[:mean_v_index]
        didv_rhs = self.didv_avg_data[mean_v_index:]
        # Extracting the mean dIdV value within the selected voltage gap
        didv_mean = np.mean(self.didv_avg_data[lower_v_index:upper_v_index])
        # Extracting the band-gap given the 1 sigma condition
        didv_sigma1 = didv_mean + np.std(self.didv_avg_data[lower_v_index:upper_v_index])
        # Extracting the band-gap given the 2 sigma condition
        didv_sigma2 = didv_mean + 2 * np.std(self.didv_avg_data[lower_v_index:upper_v_index])
        # If the voltage domain never reaches below the 1 sigma value of dIdV (error evasion for retracted I(V) noise)
        if len(didv_lhs < didv_sigma1) == 0:
            v_lhs_sigma1 = e_gap[0]
            v_rhs_sigma1 = e_gap[1]
            v_lhs_sigma2 = e_gap[0]
            v_rhs_sigma2 = e_gap[1]
        # If the voltage domain does reach below the 1 sigma value of dIdV (follow this path for normal I(V) curves)
        else:
            v_lhs_sigma1 = v_lhs[didv_lhs < didv_sigma1][0]
            v_rhs_sigma1 = v_rhs[didv_rhs < didv_sigma1][-1]
            v_lhs_sigma2 = v_lhs[didv_lhs < didv_sigma2][0]
            v_rhs_sigma2 = v_rhs[didv_rhs < didv_sigma2][-1]

        # Collating all of the gap information into a dictionary
        self.gap_info = {}
        self.gap_info['Egap'] = np.around(abs(e_gap[1] - e_gap[0]), 2)
        self.gap_info['Egap + 1 sigma'] = np.around(abs(v_rhs_sigma1 - v_lhs_sigma1), 2)
        self.gap_info['Egap + 2 sigma'] = np.around(abs(v_rhs_sigma2 - v_lhs_sigma2), 2)
        self.gap_info['Egap centre'] = np.around(v_mean, 2)
        self.gap_info['VBM'] = np.around(e_gap[0], 2)
        self.gap_info['VBM + 1 sigma'] = np.around(v_lhs_sigma1, 2)
        self.gap_info['VBM + 2 sigma'] = np.around(v_lhs_sigma2, 2)
        self.gap_info['CBM'] = np.around(e_gap[1], 2)
        self.gap_info['CBM + 1 sigma'] = np.around(v_rhs_sigma1, 2)
        self.gap_info['CBM + 2 sigma'] = np.around(v_rhs_sigma2, 2)
        self.gap_info['Mean dIdV'] = didv_mean
        self.gap_info['Mean dIdV + 1 sigma'] = didv_sigma1
        self.gap_info['Mean dIdV + 2 sigma'] = didv_sigma2

    def iv_plot(self, ax, retrace, axes_type, vbias_lims, i_lim):
        """
        Function to plot and format all the raw I(V) curves that have been selected.
        """
        # Formatting the I(V) curve plot
        ax.set_title("Raw I(V) curves", fontsize=20, fontweight="bold")
        ax.set_xlabel("Voltage bias [V]", fontsize=19)
        ax.set_ylabel("Current [A]", fontsize=19)
        ax.axhline(0, color='gray', linewidth=2.5)
        ax.axvline(0, color='gray', linewidth=2.5)
        ax.grid(True)
        # Plot all of the raw I(V) curves that are selected
        for i in range(self.num_of_selected_files):
            trace_alpha = 0.6
            retrace_alpha = 0.6
            if retrace == "Trace only":
                retrace_alpha = 0.05
            elif retrace == "Retrace only":
                trace_alpha = 0.05
            ax.plot(self.xcrop_v_dat[i], self.xcrop_i_dat[i][0], 'k.-', linewidth=1.0, markersize=4.5,
                    alpha=trace_alpha, label='Trace')
            ax.plot(self.xcrop_v_dat[i], self.xcrop_i_dat[i][1], 'b.-', linewidth=1.0, markersize=4.5,
                    alpha=retrace_alpha, label='Retrace')
        # If there are outliers due to cross-correlation or restricted voltage domain, show them as grey points
        if len(self.v_outliers) > 1:
            ax.plot(self.v_outliers, self.i_outliers, '.', markersize=4.5,
                    color='gray', alpha=0.2, label='Omitted')
            ax.legend(handles=list([patch.Patch(color='black', label='Trace'),
                                    patch.Patch(color='blue', label='Retrace'),
                                    patch.Patch(color='gray', label='Omitted')]),
                      loc='best', prop={'size': 12})
        # If there are no outliers, just show the default legend
        else:
            ax.legend(handles=list([patch.Patch(color='black', label='Trace'),
                                    patch.Patch(color='blue', label='Retrace')]),
                      loc='best', prop={'size': 12})
        # If axes limit is selected, plot the effects of this
        if axes_type == 'Axes limit':
            ax.set_xlim(vbias_lims[0], vbias_lims[1])
            ax.set_ylim(i_lim * -1e-9, i_lim * 1e-9)

    def iv_int_plots(self, ax1, ax2, ax3, smooth, axes_type, vbias_lims, i_lim, didv_lim):
        """
        Function to plot all the intermediate stages of the analysis.
        """
        # Set the color of the intermediate curves
        col = '#006600'
        # Plot and format the mean I(V) curve over all selected I(V) curves
        ax1.set_title("1 - Averaged", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Current [$A$]", fontsize=19)
        ax1.axhline(0, color='gray', linewidth=2.5)
        ax1.axvline(0, color='gray', linewidth=2.5)
        ax1.grid(True)
        ax1.plot(self.xcrop_v_dat[0], self.avg_i_data, '.-', linewidth=1.5, markersize=4.5, color=col)
        ax1.legend(handles=list([patch.Patch(color=col, label='Average I(V)')]),
                   loc='best', prop={'size': 10})
        # If axes limit is selected, plot the effects of this
        if axes_type == 'Axes limit':
            ax1.set_xlim(vbias_lims[0], vbias_lims[1])
            ax1.set_ylim(i_lim * -1e-9, i_lim * 1e-9)

        # Plot and format the smoothed of the average I(V) curve
        ax2.set_title("2 -" + str(smooth) + " Smoothed", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Current [$A$]", fontsize=19)
        ax2.axhline(0, color='gray', linewidth=2.5)
        ax2.axvline(0, color='gray', linewidth=2.5)
        ax2.grid(True)
        ax2.plot(self.xcrop_v_dat[0], self.smooth_avg_i_data, '.-', linewidth=1.5, markersize=4.5,
                 color=col)
        ax2.legend(handles=list([patch.Patch(color=col, label='Smoothed avgerage I(V)')]),
                   loc='best', prop={'size': 10})
        # If axes limit is selected, plot the effects of this
        if axes_type == 'Axes limit':
            ax2.set_xlim(vbias_lims[0], vbias_lims[1])
            ax2.set_ylim(i_lim * -1e-9, i_lim * 1e-9)
        # Deleting the x-axis ticks as they are all identical and shown by the bottom subplot
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # Plot and format the final differentiated I(V) curve
        ax3.set_title("3 - Differentiated", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Voltage bias [$V$]", fontsize=19)
        ax3.set_ylabel("dI/dV [$A/V$]", fontsize=19)
        ax3.axhline(0, color='gray', linewidth=2.5)
        ax3.axvline(0, color='gray', linewidth=2.5)
        ax3.grid(True)
        ax3.plot(self.xcrop_v_dat[0][1:], self.didv_avg_data, '.-', linewidth=1.5, markersize=4.5,
                 color=col)
        ax3.legend(handles=list([patch.Patch(color=col, label='dI/dV(V)')]),
                   loc='best', prop={'size': 10})
        # If axes limit is selected, plot the effects of this
        if axes_type == 'Axes limit':
            ax3.set_xlim(vbias_lims[0], vbias_lims[1])
            ax3.set_ylim(ymax=didv_lim * 1e-12)

    def didv_plot(self, ax, axes_type, vbias_lims, didv_lim, egap=False, stacked=False):
        """
        Function to plot and format the final dI/dV curve obtained from the raw I(V) curves.
        """
        # Formatting the dI/dV curve plot
        ax.set_title("Analysed dI/dV curves", fontsize=20, fontweight="bold")
        ax.set_xlabel("Voltage bias [$V$]", fontsize=19)
        ax.set_ylabel("dI/dV [$A/V$]", fontsize=19)
        ax.axvline(0, color='gray', linewidth=2.5)
        ax.grid(True, which='minor')
        ax.grid(True, which='major')
        # Plot all the selected dI/dV curves stacked on top of each other if stacked is True
        if stacked:
            for i in range(self.num_of_selected_files):
                ax.semilogy(self.xcrop_v_dat[0][1:], self.didv_data[i], '.-', linewidth=2.0, markersize=4.5, alpha=0.4,
                            color='#90046C')
                plt.legend(handles=list([patch.Patch(color='black', label='dI/dV'),
                                         patch.Patch(color=[0.3, 0.3, 0.3], alpha=0.3, label='Variance'),
                                         patch.Patch(color='#90046C', alpha=0.3, label='Selected dI/dV')]),
                           loc='best', prop={'size': 12})
        else:
            plt.legend(handles=list([patch.Patch(color='black', label='dI/dV'),
                                     patch.Patch(color=[0.3, 0.3, 0.3], alpha=0.3, label='Variance')]),
                       loc='best', prop={'size': 12})
        # Plot the dI/dV curve
        plt.semilogy(self.xcrop_v_dat[0][1:], self.didv_avg_data, 'k.-', linewidth=2.0, markersize=4.5)
        # Plotting the variance associated with dIdV
        plt.semilogy(self.xcrop_v_dat[0][1:], self.didv_avg_data + self.i_var, '-', linewidth=1.0,
                     color=[0.3, 0.3, 0.3], alpha=0.3)
        plt.fill_between(self.xcrop_v_dat[0][1:], self.didv_avg_data, self.didv_avg_data + self.i_var,
                         color=[0.6, 0.6, 0.6], alpha=0.3)
        if egap:
            # Plot the best estimates for the band-gap, VBM and CBM edges
            # - Extracting the average band-gap line
            v_gap = np.array([self.gap_info['VBM'], self.gap_info['Egap centre'], self.gap_info['CBM']])
            didv_gap = np.ones(len(v_gap)) * self.gap_info['Mean dIdV']
            # - Extracting the 1-sigma band-gap line
            v_gap_sigma1 = np.array(
                [self.gap_info['VBM + 1 sigma'], self.gap_info['Egap centre'], self.gap_info['CBM + 1 sigma']])
            didv_gap_sigma1 = np.ones(len(v_gap_sigma1)) * self.gap_info['Mean dIdV + 1 sigma']
            # - Extracting the 2-sigma band-gap line
            v_gap_sigma2 = np.array(
                [self.gap_info['VBM + 2 sigma'], self.gap_info['Egap centre'], self.gap_info['CBM + 2 sigma']])
            didv_gap_sigma2 = np.ones(len(v_gap_sigma2)) * self.gap_info['Mean dIdV + 2 sigma']
            # - Plot the band-gap lines and middle position points
            plt.plot(v_gap, didv_gap, '-', linewidth=6, markersize=10, color=[0, 0, 0.3])
            plt.plot(v_gap_sigma1, didv_gap_sigma1, '-', linewidth=5, markersize=10, color=[0, 0, 0.6])
            plt.plot(v_gap_sigma2, didv_gap_sigma2, '-', linewidth=4, markersize=8, color=[0, 0, 0.9])
            plt.plot(self.gap_info['Egap centre'], self.gap_info['Mean dIdV'], 'o', markersize=10, color=[0, 0, 0.3])
            plt.plot(self.gap_info['Egap centre'], self.gap_info['Mean dIdV + 1 sigma'], 'o', markersize=10,
                     color=[0, 0, 0.6])
            plt.plot(self.gap_info['Egap centre'], self.gap_info['Mean dIdV + 2 sigma'], 'o', markersize=10,
                     color=[0, 0, 0.9])
            # - Plot the VBM position points
            plt.plot(self.gap_info['VBM'], self.gap_info['Mean dIdV'], 'o', markersize=10, color=[0, 0.3, 0])
            plt.plot(self.gap_info['VBM + 1 sigma'], self.gap_info['Mean dIdV + 1 sigma'], 'o', markersize=10,
                     color=[0, 0.6, 0])
            plt.plot(self.gap_info['VBM + 2 sigma'], self.gap_info['Mean dIdV + 2 sigma'], 'o', markersize=10,
                     color=[0, 0.9, 0])
            # - Plot the CBM position points
            plt.plot(self.gap_info['CBM'], self.gap_info['Mean dIdV'], 'o', markersize=10, color=[0.3, 0, 0])
            plt.plot(self.gap_info['CBM + 1 sigma'], self.gap_info['Mean dIdV + 1 sigma'], 'o', markersize=10,
                     color=[0.6, 0, 0])
            plt.plot(self.gap_info['CBM + 2 sigma'], self.gap_info['Mean dIdV + 2 sigma'], 'o', markersize=10,
                     color=[0.9, 0, 0])
            # Shade in the areas between the band-gap uncertainty lines
            xshade1 = np.array([v_gap[0], v_gap_sigma1[0], v_gap_sigma1[-1], v_gap[-1]])
            yshade1 = np.array([didv_gap[0], didv_gap_sigma1[0], didv_gap_sigma1[-1], didv_gap[-1]])
            plt.fill_between(xshade1, yshade1, color=[0, 0, 0.6], alpha=0.3)
            xshade2 = np.array([v_gap_sigma1[0], v_gap_sigma2[0], v_gap_sigma2[-1], v_gap_sigma1[-1]])
            yshade2 = np.array([didv_gap_sigma1[0], didv_gap_sigma2[0], didv_gap_sigma2[-1], didv_gap_sigma1[-1]])
            plt.fill_between(xshade2, yshade2, color=[0, 0, 0.9], alpha=0.3)
        # Limit the axes if selected by the user
        if axes_type == 'Axes limit':
            plt.xlim(vbias_lims[0], vbias_lims[1])
            plt.ylim(ymax=didv_lim * 1e-12)

    def didv_image(self, ax, axes_type, vbias_lims, didv_lim):
        """
        Function to plot the mean dI/dV curve and all the stacked dI/dV curves from all the selected I(V) files.
        """
        # Formatting the dI/dV image
        ax.set_title("Train of dI/dV curves", fontsize=20, fontweight="bold")
        ax.set_ylabel("Voltage bias [$V$]", fontsize=19)
        ax.set_xlabel("Index", fontsize=19)
        ax.axhline(0, color='white', linewidth=2.5, linestyle='--')
        ax.set_yticks(np.arange(np.round(np.min(self.xcrop_v_dat[0]), 0), np.round(np.max(self.xcrop_v_dat[0]), 0),
                                0.2))
        if self.num_of_selected_files < 150:
            ax.set_xticks(np.arange(0, self.num_of_selected_files, 5))
        else:
            ax.set_xticks(np.arange(0, self.num_of_selected_files, 20))
        ax.yaxis.grid(which="major")
        img = np.matrix.transpose(self.didv_data)
        # Plotting the CITS slice from the multiple I(V) curves selected
        if axes_type == 'Axes limit' or axes_type == 'Image contrast':
            cits_slice = ax.imshow(img, cmap="viridis", aspect='auto', interpolation='gaussian', origin='lower',
                                   norm=LogNorm(vmin=1e-14, vmax=didv_lim*1e-12),
                                   extent=[0, self.num_of_selected_files, np.min(self.xcrop_v_dat[0]),
                                           np.max(self.xcrop_v_dat[0])])
        else:
            cits_slice = ax.imshow(img, cmap="viridis", aspect='auto', interpolation='gaussian', origin='lower',
                                   norm=LogNorm(vmin=1e-14, vmax=1e-11),
                                   extent=[0, self.num_of_selected_files, np.min(self.xcrop_v_dat[0]),
                                           np.max(self.xcrop_v_dat[0])])
        # Plotting the associated colorbar
        cbar = plt.colorbar(cits_slice, fraction=0.046, pad=0.01)
        cbar.ax.set_ylabel('dI/dV [A/V]', fontsize=14)
        # Shade in the areas between the band-gap uncertainty lines
        X = np.array([0, self.num_of_selected_files])
        vbm = np.array([self.gap_info['VBM'], self.gap_info['VBM']])
        vbm2s = np.array([self.gap_info['VBM + 2 sigma'], self.gap_info['VBM + 2 sigma']])
        cbm = np.array([self.gap_info['CBM'], self.gap_info['CBM']])
        cbm2s = np.array([self.gap_info['CBM + 2 sigma'], self.gap_info['CBM + 2 sigma']])
        plt.fill_between(X, vbm, vbm2s, color=[0, 0.6, 0], alpha=0.3)
        plt.fill_between(X, cbm, cbm2s, color=[0.6, 0, 0], alpha=0.3)
        # Add text for the VBM and CBM locations
        plt.text(0, self.gap_info['VBM + 2 sigma'], 'VBM 2 $\\sigma$', fontsize=14, color='white')
        plt.text(0, self.gap_info['CBM + 2 sigma'], 'CBM 2 $\\sigma$', fontsize=14, color='white')
        # Limit the axes if selected by the user
        if axes_type == 'Axes limit':
            plt.ylim(vbias_lims[0], vbias_lims[1])

    def get_widgets(self):
        """
        Creates a variety of widgets to be interacted with for the analysis of the I(V) curves.
        """
        # Select Multiple widget to select the the flat-files to be analysed
        data_select_0 = ipy.SelectMultiple(options=self.file_alias, continuous_update=False, value=[self.file_alias[0]],
                                           description="$$Raw\,I(V)\,files$$",
                                           layout=ipy.Layout(display='inline-flex', flex_flow='column',
                                                             align_items='stretch', align_content='center',
                                                             width='auto', height='100%'))

        # Toggle Buttons widget to select the type of analysis to be performed
        analysis_select_1 = ipy.ToggleButtons(options=['Intermediate plots', 'Point STS', 'Line STS'],
                                              continuous_update=False,
                                              value='Intermediate plots', description="$$I(V)\,analysis\,controls$$",
                                              layout=ipy.Layout(display='inline-flex', flex_flow='column',
                                                                align_items='center', align_content='center',
                                                                justify_content='center', height='auto'))
        # Float Range Slider widget to find an estimate of the band gap
        vbias_select_1 = ipy.FloatRangeSlider(value=[-2.5, 2.5], min=-2.5, max=2.5, step=0.01,
                                              description="Restrict: $V_{bias}$ [$V$]", continuous_update=False,
                                              layout=ipy.Layout(width='95%', height='auto', display='flex',
                                                                flex_flow='row', align_items='stretch'))
        # Toggle Buttons widget to choose between the traces of the I(V) curves
        retrace_select_1 = ipy.ToggleButtons(options=["Both", "Trace only", "Retrace only"], description="Traces: ",
                                             continuous_update=False, value="Both",
                                             layout=ipy.Layout(display='flex', flex_flow='row', align_items='stretch',
                                                               height='auto'))
        # Toggle Buttons widget to choose the type of smoothing to be performed
        smooth_select_1 = ipy.ToggleButtons(options=["None", "Binomial", "Savitzky-Golay"], description="Smoothing: ",
                                            continuous_update=False, value="Savitzky-Golay",
                                            layout=ipy.Layout(display='flex', flex_flow='row', align_items='stretch',
                                                              height='auto'))
        # Float Range Slider widget to find an estimate of the band gap
        smooth_order_select_1 = ipy.IntSlider(value=3, min=1, max=10, description="Smoothing order: ",
                                              continuous_update=False,
                                              layout=ipy.Layout(width='75%', height='auto', display='flex',
                                                                flex_flow='row', align_items='stretch'))
        # Float Range Slider widget to find an estimate of the band gap
        bandgap_select_1 = ipy.FloatRangeSlider(value=[-0.25, 0.25], min=-1.5, max=1.5, step=0.01,
                                                description="Band-gap [$V$]: ", continuous_update=False,
                                                layout=ipy.Layout(width='95%', height='auto', display='flex',
                                                                  flex_flow='row', align_items='stretch'))


        # Radio Buttons widget to allow allow autoscaling, limits and limiting crop
        limit_type_select_2 = ipy.RadioButtons(options=['Auto-scale axes', 'Axes limit', 'Image contrast'],
                                               value='Auto-scale axes', description="$$Axes\,Controls$$",
                                               continuous_update=False,
                                               layout=ipy.Layout(isplay='inline-flex', flex_flow='column',
                                                                 align_items='stretch', align_content='center',
                                                                 width='97%', height='55%'))
        # Float Range Slider widget to fix the voltage bias axes limits
        v_limits_select_2 = ipy.FloatRangeSlider(value=[-1.5, 1.5], min=-2.5, max=2.5, step=0.05,
                                                 description="$V_{bias}$ [$V$]:", continuous_update=False,
                                                 layout=ipy.Layout(width='auto', display='flex',
                                                                   flex_flow='row', align_items='stretch'))

        # Selection Slider widget to fix the current axes limits
        i_limits_select_2 = ipy.SelectionSlider(options=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3,
                                                         0.5, 0.7, 1, 30, 50, 70, 100],
                                                value=1, description='$I_{tunn}$ [$nA$]:', continuous_update=False,
                                                layout=ipy.Layout(width='97%', display='flex',
                                                                  flex_flow='row', align_items='stretch'))

        # Float Range Slider widget to fix the current axes limits
        didv_limits_select_2 = ipy.SelectionSlider(options=[0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1,
                                                            0.3, 0.5, 0.7, 1, 30, 50, 70, 100],
                                                   value=1, description='$dI/dV$ [$pA/V$]: ',
                                                   continuous_update=False,
                                                   layout=ipy.Layout(width='97%', display='flex',
                                                                     flex_flow='row', align_items='stretch'))

        # Defining a global widget box to hold all of the widgets
        self.widgets = ipy.HBox([ipy.VBox([data_select_0],
                                          layout=ipy.Layout(display='flex', width='10%',
                                                            flex_flow='column', align_items='stretch',
                                                            border='solid 0.5px')
                                          ),
                                 ipy.VBox(
                                     [analysis_select_1, vbias_select_1, retrace_select_1, smooth_select_1,
                                      smooth_order_select_1, bandgap_select_1],
                                     layout=ipy.Layout(display='flex', width='63%',
                                                       flex_flow='column', align_items='stretch',
                                                       justify_content='center')
                                     ),
                                 ipy.VBox([limit_type_select_2, v_limits_select_2, i_limits_select_2,
                                           didv_limits_select_2],
                                          layout=ipy.Layout(display='flex', width='27%',
                                                            flex_flow='column', align_items='stretch',
                                                            border='solid 0.5px')
                                          )],
                                layout=ipy.Layout(width='auto', align_items='stretch'))

    def update_function(self, chosen_data, analysis_type, vbias_crop, retrace, smooth, smooth_order, e_gap, axes_type,
                        vbias_lims, i_lim, didv_lim):
        """
        Updates the I(V) curves and analysis using the defined widgets.
        """

        # Obtain the files that have been selected by the user
        self.selected_files = chosen_data

        # Extracting the data from the files
        self.selected_data_extract()
        # Perform cross-correlation analysis between different I(V) curves
        self.selected_data_cross_correlation()
        # Perform additional data cropping over the voltage domain selected by the user
        self.selected_data_crop(vbias_crop)

        # Update the data analysis based on the user interaction
        # - If the user selects Intermediate or Point STS analysis
        self.sts_analysis(retrace, smooth, smooth_order)
        # Update the band-gap information based on the user interaction
        self.sts_egap_finder(e_gap)

        # Define a figure object with a certain size
        plt.subplots(figsize=(20, 10))

        # 1 - Defining the analysis stream when intermediate plots is selected
        if analysis_type == 'Intermediate plots':
            # - Plot the raw spectroscopy curves
            ax1 = plt.subplot(1, 3, 1)
            self.iv_plot(ax1, retrace, axes_type, vbias_lims, i_lim)
            # - Plot the intermediate analysis curves
            ax2 = plt.subplot(3, 3, 2)
            ax3 = plt.subplot(3, 3, 5, sharex=ax2, sharey=ax2)
            ax4 = plt.subplot(3, 3, 8, sharex=ax3)
            self.iv_int_plots(ax2, ax3, ax4, smooth, axes_type, vbias_lims, i_lim, didv_lim)
            # - Plot the final dIdV curve
            ax5 = plt.subplot(1, 3, 3)
            self.didv_plot(ax5, axes_type, vbias_lims, didv_lim)
            ax5.yaxis.tick_right()
            ax5.yaxis.set_label_position("right")
            ax5.set_title('Average dI/dV curve', fontsize=20, fontweight="bold")

        # 2 - Defining the analysis stream when point sts is selected
        elif analysis_type == 'Point STS':
            # - Plot the raw spectroscopy curves
            ax6 = plt.subplot(1, 2, 1)
            self.iv_plot(ax6, retrace, axes_type, vbias_lims, i_lim)
            # - Plot the final dIdV curve
            ax7 = plt.subplot(1, 2, 2)
            self.didv_plot(ax7, axes_type, vbias_lims, didv_lim, True)
            ax7.yaxis.tick_right()
            ax7.yaxis.set_label_position("right")
            ax7.set_title('Average dI/dV curve', fontsize=20, fontweight="bold")
            # - Add some text that gives the band-gap information
            plt.gcf().text(0.95, 0.85, '$E_{GAP}$ + $0\\sigma$ = ' + str(self.gap_info['Egap']) + 'V',
                           fontsize=15, color=[0, 0, 0.3])
            plt.gcf().text(0.95, 0.82, '$E_{GAP}$ + $1\\sigma$ = ' + str(self.gap_info['Egap + 1 sigma']) + 'V',
                           fontsize=15, color=[0, 0, 0.6])
            plt.gcf().text(0.95, 0.79, '$E_{GAP}$ + $2\\sigma$ = ' + str(self.gap_info['Egap + 2 sigma']) + 'V',
                           fontsize=15, color=[0, 0, 0.9])
            # - Add some text that gives the VBM information
            plt.gcf().text(0.95, 0.74, '$VBM$ = ' + str(self.gap_info['VBM']) + 'V', fontsize=15, color=[0, 0.3, 0])
            plt.gcf().text(0.95, 0.71, '$VBM$ + $1\\sigma$ = ' + str(self.gap_info['VBM + 1 sigma']) + 'V',
                           fontsize=15, color=[0, 0.6, 0])
            plt.gcf().text(0.95, 0.68, '$VBM$ + $2\\sigma$ = ' + str(self.gap_info['VBM + 2 sigma']) + 'V',
                           fontsize=15, color=[0, 0.9, 0])
            # - Add some text that gives the CBM information
            plt.gcf().text(0.95, 0.63, '$CBM$ = ' + str(self.gap_info['CBM']) + 'V', fontsize=15, color=[0.3, 0, 0])
            plt.gcf().text(0.95, 0.60, '$CBM$ + $1\\sigma$ = ' + str(self.gap_info['CBM + 1 sigma']) + 'V',
                           fontsize=15, color=[0.6, 0, 0])
            plt.gcf().text(0.95, 0.57, '$CBM$ + $2\\sigma$ = ' + str(self.gap_info['CBM + 2 sigma']) + 'V',
                           fontsize=15, color=[0.9, 0, 0])
            # - Add some text about the conductance information
            didv_avg = self.gap_info['Mean dIdV']
            didv_sigma = self.gap_info['Mean dIdV + 1 sigma'] - self.gap_info['Mean dIdV']
            plt.gcf().text(0.95, 0.52, '$dI/dV_{avg}$ = %.2e A/V' % didv_avg, fontsize=15)
            plt.gcf().text(0.95, 0.49, '$dI/dV_{\\sigma}$ = %.2e A/V' % didv_sigma, fontsize=15)

        # 3 - Defining the analysis stream when line sts is selected
        elif analysis_type == 'Line STS':
            # Plot all the final dIdV curves
            ax8 = plt.subplot(1, 2, 1)
            # - Plot the average dI/dV curve
            self.didv_plot(ax8, axes_type, vbias_lims, didv_lim, True, True)
            # Plot and format the
            ax9 = plt.subplot(1, 2, 2)
            self.didv_image(ax9, axes_type, vbias_lims, didv_lim)

        # Show the figure that has been created
        plt.show()

        return

    def user_interaction(self):
        """
        Function that allows the continuous interaction of the widgets to update the figure.
        """
        # Display the box of custom widgets
        display(self.widgets)

        # Extracting all of the necessary widgets
        chosen_data = self.widgets.children[0].children[0]
        analysis_type = self.widgets.children[1].children[0]
        bias_restrict = self.widgets.children[1].children[1]
        retrace = self.widgets.children[1].children[2]
        smooth = self.widgets.children[1].children[3]
        smooth_order = self.widgets.children[1].children[4]
        e_gap = self.widgets.children[1].children[5]
        axes_type = self.widgets.children[2].children[0]
        vbias_lims = self.widgets.children[2].children[1]
        i_lim = self.widgets.children[2].children[2]
        didv_lim = self.widgets.children[2].children[3]

        # Define the attribute to continuously update the figure, given the user interaction
        self.output = ipy.interactive(self.update_function, chosen_data=chosen_data,
                                      analysis_type=analysis_type, vbias_crop=bias_restrict, retrace=retrace,
                                      smooth=smooth, smooth_order=smooth_order, e_gap=e_gap, axes_type=axes_type,
                                      vbias_lims=vbias_lims, i_lim=i_lim, didv_lim=didv_lim)

        # Display the final output of the widget interaction
        display(self.output.children[-1])

