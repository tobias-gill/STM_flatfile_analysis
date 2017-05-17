import glob                                 # Module to load in all folder for extracting all data from folders
from copy import deepcopy                   # Module to create deep copies of flat-file instances in topography
import numpy as np                          # Standard numpy module
import matplotlib.pyplot as plt             # Standard matplotlib module in regards to plotting all figures
import matplotlib.patches as patch          # Standard matplotlib module in regards to plotting patches on figures
from matplotlib.colors import LogNorm       # Standard matplotlib module in regards to creating a log scale colorbar
from scipy.optimize import leastsq          # Standard scipy module that performs the least square operation
from scipy.ndimage.interpolation import rotate  # Standard scipy module to perform rotation of topography plots
import ipywidgets as ipy                    # Standard ipywidgets module that holds all widget functionality
from IPython.display import display         # Specific module to explicitly display the pre-defined widgets
import flatfile_3 as ff                     # Module that loads in MATRIX flat-files into python class objects

# Information about the "stm_analysis.py" module
__version__ = "2.00"
__date__ = "15th May 2017"
__status__ = "Pending"

__authors__ = "Procopi Constantinou & Tobias Gill"
__email__ = "procopios.constantinou.16@ucl.ac.uk"

# 0 - Defining some physical constants that may be used throughout the analysis
PC = {"c": 2.99792458e8, "e": 1.6021773e-19, "me": 9.109389e-31, "kB": 1.380658e-23,
      "h": 6.6260755e-34, "hbar": 1.05457e-34, "eps0": 8.85419e-12,
      "pico": 1e-12, "nano": 1e-9, "micro": 1e-6}


# 1.0 - Defining the class object to select the parent directory to browse through all the stm data
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


# 2.0 - Defining the class object that will import the '.Z_flat' files and perform all the necessary topography analysis
class STT(object):
    def __init__(self, DS):
        """
        Defines the initialisation of the class object.
        DS:     The 'DataSelection' class object.
        """
        # 2.0.1 -  Extract all the flat-files from the data directory selected
        self.flat_files = glob.glob(DS.selected_path + '*.Z_flat')       # List of all the topography flat file paths
        self.flat_files = sorted(self.flat_files, key=len)               # Sorting the list in ascending order
        self.num_of_files = len(self.flat_files)                         # Total number of flat files loaded
        self.file_alias = None                                           # List of unique identifiers to the flat files
        self.all_flatfile_extract()

        # 2.0.2 - Defining all the attributes associated with the topography file selection
        self.selected_file = None                   # String of the selected topography alias
        self.selected_pos = None                    # Integer of the array position of the topography file selected
        # - Dictionary of the scan directions
        self.scan_dict = {'up-fwd': 0, 'up-bwd': 1, 'down-fwd': 2, 'down-bwd': 3}
        # - Dictionary of inverted scan directions
        self.scan_dict_inv = {0: 'up-fwd', 1: 'up-bwd', 2: 'down-fwd', 3: 'down-bwd'}
        # - Defining all other necessary attributes
        self.selected_data = None                   # List holding the selected topography flat-file data classes
        self.scan_dir = None                        # Integer that determines the selected scan direction
        self.scan_dir_not = None                    # Array of the other scan directions that are not selected

        # 2.0.3 - Defining the leveling and image operations
        self.image_props = None                     # Dictionary that contains all the image properties
        self.leveled_data = None                    # Attribute that holds the updated level corrected data
        self.final_data = None                      # Attribute that holds the final topography data after all changes

        # 2.0.4 User interaction
        self.widgets = None                         # Widget object to hold all pre-defined widgets
        self.get_widgets()                          # Function to get all of the pre-defined widgets
        self.output = None                          # Output to the user interaction with widgets
        self.user_interaction()                     # Function to allow continuous user interaction

    def all_flatfile_extract(self):
        """
        Function to extract the file names and total number of topography flat-files within the given directory.
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
                fname = "Spectroscopy--" + str(scan_num) + "_" + str(repeat) + ".Z_flat"
                # If the file name is found, save it and add one unit to the break point
                if len([x for x in self.flat_files if fname in x]) == 1:
                    # Making the file name consistent
                    if scan_num < 10:
                        file_alias.append("topo 00" + str(scan_num) + "_" + str(repeat))
                    elif scan_num < 100:
                        file_alias.append("topo 0" + str(scan_num) + "_" + str(repeat))
                    else:
                        file_alias.append("topo " + str(scan_num) + "_" + str(repeat))
                    # Add one to the file number
                    file_num += 1
                if file_num == self.num_of_files:
                    cond = False
        # Return the unique identifiers to each I(V) flat file
        self.file_alias = file_alias

    def selected_data_extract(self, scan_dir):
        """
        Function to extract the raw data as an instance of an Omicron topography flat file. Additionally, it will return
        the x- and y-axes in terms of their real units.
        """
        # Extract the position of the topography file selected
        self.selected_pos = int(self.file_alias.index(self.selected_file))
        # Extract the topography raw data from the selected flat-file by using the flat-file load function
        self.selected_data = ff.load(self.flat_files[self.selected_pos])
        # Extract the scan-direction
        self.scan_dir = self.scan_dict[scan_dir]
        # Create an array of the minor scan directions
        if self.scan_dir == 0:
            self.scan_dir_not = np.array([1, 2, 3])
        elif self.scan_dir == 1:
            self.scan_dir_not = np.array([0, 2, 3])
        elif self.scan_dir == 2:
            self.scan_dir_not = np.array([0, 1, 3])
        elif self.scan_dir == 3:
            self.scan_dir_not = np.array([0, 1, 2])

    def nm2pnt(self, nm, flat_file, axis='x'):
        """
        Convert between nanometers and corresponding pixel number for a given Omicron flat file.

        :param nm: Nanometer value.
        :param flat_file: Instance of an Omicron flat file.
        :param axis: Plot axis of nm point. Must be either 'x' or 'y'.
        :return: Pixel number for nanometer value.
        """
        if axis == 'x':
            inc = flat_file[0].info['xinc']
        elif axis == 'y':
            inc = flat_file[0].info['yinc']

        pnt = np.int(np.round(nm / inc))

        if pnt < 0:
            pnt = 0
        if axis == 'x':
            if pnt > flat_file[0].info['xres']:
                pnt = flat_file[0].info['xres']
        elif axis == 'y':
            if pnt > flat_file[0].info['yres']:
                pnt = flat_file[0].info['yres']

        return pnt

    def topo_linewise(self, flat_file, scan_dir):
        """
        Create a copied instance of the flat file after linewise flattening an stm image by fitting lines through each 
        stm line scan, and subsequently subtracting that line from the stm scan line. This subtraction method is best 
        used for scans that are all on the same terrace.
        
        :param flat_file: Instance of an Omicron flat file.
        :param scan_dir: flat file scan direction.
        :return: the modified flat-file instance that has been line-subtracted over the scan direction.
        """
        # Create a new deep copy of the flat file
        flat_file_copy = deepcopy(flat_file)
        # Extracting the information from the flat-file
        topo_info = flat_file_copy[scan_dir].info
        # - Extract the total number of x, y pixels from the flat file (total number of points in the scan)
        x_res = topo_info['xres']
        y_res = topo_info['yres']
        # - Defining the x domain of the pixels over which the line-subtraction will be performed
        x_range = np.arange(0, x_res, 1)
        # Extracting the raw data from the flat-file instance
        topo_data = flat_file_copy[scan_dir].data
        # Executing the line-wise subtraction
        # - Define the flattened topography data array
        topo_flat_data = np.zeros((y_res, x_res))
        # - Iterate over all the y-axis pixels
        for y in range(0, y_res):
            # Finding the line of best fit through an stm scan line
            line = np.poly1d(np.polyfit(x_range, topo_data[y], 1))(x_range)
            # Appending the line subtracted data to the topo_flat_data array
            topo_flat_data[y] = topo_data[y] - line
        # Modify the copy of the flat-file instance so that the data over the scan direction is linewise subtracted
        flat_file_copy[scan_dir].data = topo_flat_data
        # Return the new amended flat file instance.
        return flat_file_copy

    def topo_localplane(self, flat_file, scan_dir, x0, x1, y0, y1):
        """
        Create a copied instance of the flat file after plane flattening an stm image, by fitting to a defined area.
        
        :param file_data: An instance of an Omicron flat file.
        :param scan_dir: flat file scan direction.
        :param x0: x-axis plane area initial co-ordinate in real units.
        :param x1: x-axis plane area final co-ordinate in real units.
        :param y0: y-axis plane area initial co-ordinate in real units.
        :param y1: y-axis plane are final co-ordinate in real units.
        :return: the modified flat-file instance that has been plane-subtracted over the scan direction and given area.
        """
        # Create a new deep copy of the flat file
        flat_file_copy = deepcopy(flat_file)
        # Extracting the information from the flat-file
        topo_info = flat_file_copy[scan_dir].info
        # - Extract the total number of x, y pixels from the flat file (total number of points in the scan)
        x_res = topo_info['xres']
        y_res = topo_info['yres']

        # Defining the function to determine the residuals of the fitted plane
        def topo_plane_residuals(param, topo_data, x0, x1, y0, y1):
            """
            Calculate the residuals between the real and fit generated data.
            :param param: List of three fit parameters for the x and y plane gradients, and z offset.
            :param topo_data: numpy array containing topography data.
            :param x0: x-axis plane area initial co-ordinate.
            :param x1: x-axis plane area final co-ordinate.
            :param y0: y-axis plane area intial co-ordinate.
            :param y1: y-axis plane area final co-ordinate.
            :return: Plane corrected data.
            """
            # Extracting the parameter information
            p_x = param[0]
            p_y = param[1]
            p_z = param[2]
            # Determination of the residuals between the real and fitted data
            diff = []
            for y in range(y0, y1):
                for x in range(x0, x1):
                    diff.append(topo_data[y, x] - (p_x * x + p_y * y + p_z))
            return diff

        # Defining the function to determine the parameters of the fitted plane
        def topo_plane_paramEval(param, x_res, y_res):
            """
            Generate a plane from given parameters.
            :param param: List of x, y gradients and z offset.
            :return: Generated plane data.
            """
            # Create an empty numpy array with the same number as pixels as the real data.
            topo_plane_fit_data = np.zeros((y_res, x_res))
            for y in range(0, y_res):  # Iterate over the y-axis pixels.
                for x in range(0, x_res):  # Iterate over the x-axis pixels.
                    topo_plane_fit_data[y, x] = param[0] * x + param[1] * y + param[2]  # Generate plane value.
            return topo_plane_fit_data  # Return entire array.

        # If the plane area is not well defined, define the starting points to be zero and end points to be the maxima
        if x0 == x1 or y0 == y1:
            x0 = self.nm2pnt(0, flat_file_copy)
            x1 = self.nm2pnt(self.selected_data[self.scan_dir].info['xreal'], flat_file_copy)
            y0 = self.nm2pnt(0, flat_file_copy, axis='y')
            y1 = self.nm2pnt(self.selected_data[self.scan_dir].info['yreal'], flat_file_copy, axis='y')
        # If the plane area is well defined, use the given points
        else:
            x0 = self.nm2pnt(x0, flat_file_copy)
            x1 = self.nm2pnt(x1, flat_file_copy)
            y0 = self.nm2pnt(y0, flat_file_copy, axis='y')
            y1 = self.nm2pnt(y1, flat_file_copy, axis='y')

        # Extracting the raw data from the flat-file instance
        topo_data = flat_file_copy[scan_dir].data
        # Initialising the parameters
        param_init = [1, 1, 1]
        # Determination of the plane-subtracted topography data
        topo_plane_lsq = leastsq(topo_plane_residuals, param_init, args=(topo_data, x0, x1, y0, y1))[0]
        topo_plane_fit = topo_plane_paramEval(topo_plane_lsq, x_res, y_res)
        topo_data_flattened = topo_data - topo_plane_fit
        topo_data_flattened = topo_data_flattened - np.amin(topo_data_flattened)
        # Modify the copy of the flat-file instance so that the data over the scan direction is plane subtracted
        flat_file_copy[scan_dir].data = topo_data_flattened
        # Return the new amended flat file instance.
        return flat_file_copy

    def topo_rotate(self, flat_file, angle):
        """
        Create a copied instance of the flat file rotated by the given angle (in degrees).

        :param flat_file: An instance of an Omicron flat file.
        :param angle: Rotation angle in degrees.
        :return: New flat file instance with rotated image data.
        """
        # Create a new deep copy of the flat file
        flat_file_copy = deepcopy(flat_file)

        # For each scan direction in the flat file rotate the data by the given angle.
        for scan_dir in flat_file_copy:
            scan_dir.data = rotate(scan_dir.data, angle)

        new_res = np.shape(flat_file_copy[0].data)  # Get the new pixel resolution from the rotated image.

        # For each scan direction amend the metadata pertinent to the new dimensions.
        for scan_dir in flat_file_copy:
            scan_dir.info['xres'] = new_res[1]  # Set new x-axis pixel resolution.
            scan_dir.info['yres'] = new_res[0]  # Set new y-axis pixel resolution.

            scan_dir.info['xreal'] = scan_dir.info['xinc'] * new_res[1]  # Set new x-axis image size.
            scan_dir.info['yreal'] = scan_dir.info['yinc'] * new_res[0]  # Set new y-axis image size.

        return flat_file_copy  # Return the new amended flat file instance.

    def topo_crop(self, flat_file, xmin, xmax, ymin, ymax):
        """
        Create a copy of the flat file, cropped by the defined pixel numbers.

        :param flat_file: An instance of an Omicron flat file.
        :param xmin: Crop x-axis initial co-ordinate in real units.
        :param xmax: Crop x-axis final co-ordinate in real units.
        :param ymin: Crop y-axis initial co-ordinate in real units.
        :param ymax: Crop y-axis final co-ordinate in real units.
        :return: New flat file instance with cropped image data.
        """
        # Converting from real units to pixel units for the image cropping operation
        xmin = self.nm2pnt(xmin, flat_file)
        xmax = self.nm2pnt(xmax, flat_file)
        ymin = self.nm2pnt(ymin, flat_file, axis='y')
        ymax = self.nm2pnt(ymax, flat_file, axis='y')

        # Create a new deep copy of the flat file
        flat_file_copy = deepcopy(flat_file)

        # For each scan direction in the flat file crop the data and amend metadata
        # - If the cropping values of the min and max are identical, avoid error and return original flat-file instance
        if xmin == xmax or ymin == ymax:
            for scan_dir in flat_file_copy:
                # - Set the minimum real value of the x- and y-axis to be zero as there is no cropping here
                scan_dir.info['xreal_min'] = 0
                scan_dir.info['yreal_min'] = 0
            # - Return new flat file instance.
            return flat_file_copy
        # - If the cropping values of the min and max are proper, perform the cropping operation
        elif xmin < xmax and ymin < ymax:
            for scan_dir in flat_file_copy:
                # - Crop the image data
                scan_dir.data = scan_dir.data[ymin:ymax, xmin:xmax]
                # - Set new x- and y-axis pixel resolution
                scan_dir.info['xres'] = xmax - xmin
                scan_dir.info['yres'] = ymax - ymin
                # - Preserve the old positions of the x- and y-axis cropping point
                scan_dir.info['xreal_min'] = scan_dir.info['xinc'] * xmin
                scan_dir.info['yreal_min'] = scan_dir.info['yinc'] * ymin
                # - Set new x- and y-axis image size
                scan_dir.info['xreal'] = scan_dir.info['xreal_min'] + scan_dir.info['xinc'] * scan_dir.info['xres']
                scan_dir.info['yreal'] = scan_dir.info['yreal_min'] + scan_dir.info['yinc'] * scan_dir.info['yres']
            # - Return new flat file instance.
            return flat_file_copy
        # - If the cropping values of the xmin and xmax are switched, but ymin and ymax are proper
        elif xmin > xmax and ymin < ymax:
            for scan_dir in flat_file_copy:
                # - Crop the image data
                scan_dir.data = scan_dir.data[ymin:ymax, xmax:xmin]
                # - Set new x- and y-axis pixel resolution
                scan_dir.info['xres'] = xmin - xmax
                scan_dir.info['yres'] = ymax - ymin
                # - Preserve the old positions of the x- and y-axis cropping point
                scan_dir.info['xreal_min'] = scan_dir.info['xinc'] * xmax
                scan_dir.info['yreal_min'] = scan_dir.info['yinc'] * ymin
                # - Set new x- and y-axis image size
                scan_dir.info['xreal'] = scan_dir.info['xreal_min'] + scan_dir.info['xinc'] * scan_dir.info['xres']
                scan_dir.info['yreal'] = scan_dir.info['yreal_min'] + scan_dir.info['yinc'] * scan_dir.info['yres']
                # - Return new flat file instance.
            return flat_file_copy
        # - If the cropping values of the ymin and ymax are switched, but xmin and xmax are proper
        elif xmin < xmax and ymin > ymax:
            for scan_dir in flat_file_copy:
                # - Crop the image data
                scan_dir.data = scan_dir.data[ymax:ymin, xmin:xmax]
                # - Set new x- and y-axis pixel resolution
                scan_dir.info['xres'] = xmax - xmin
                scan_dir.info['yres'] = ymin - ymax
                # - Preserve the old positions of the x- and y-axis cropping point
                scan_dir.info['xreal_min'] = scan_dir.info['xinc'] * xmin
                scan_dir.info['yreal_min'] = scan_dir.info['yinc'] * ymax
                # - Set new x- and y-axis image size
                scan_dir.info['xreal'] = scan_dir.info['xreal_min'] + scan_dir.info['xinc'] * scan_dir.info['xres']
                scan_dir.info['yreal'] = scan_dir.info['yreal_min'] + scan_dir.info['yinc'] * scan_dir.info['yres']
                # - Return new flat file instance.
            return flat_file_copy
        # - If the cropping values of both min and max are switched, avoid error by reversing the crop direction
        elif xmin > xmax and ymin > ymax:
            for scan_dir in flat_file_copy:
                # - Crop the image data
                scan_dir.data = scan_dir.data[ymax:ymin, xmax:xmin]
                # - Set new x- and y-axis pixel resolution
                scan_dir.info['xres'] = xmin - xmax
                scan_dir.info['yres'] = ymin - ymax
                # - Preserve the old positions of the x- and y-axis cropping point
                scan_dir.info['xreal_min'] = scan_dir.info['xinc'] * xmax
                scan_dir.info['yreal_min'] = scan_dir.info['yinc'] * ymax
                # - Set new x- and y-axis image size
                scan_dir.info['xreal'] = scan_dir.info['xreal_min'] + scan_dir.info['xinc'] * scan_dir.info['xres']
                scan_dir.info['yreal'] = scan_dir.info['yreal_min'] + scan_dir.info['yinc'] * scan_dir.info['yres']

            # Return the modified flat-file instance
            return flat_file_copy

    def minimap_crop(self, xmin, xmax, ymin, ymax, angle):
        """
        Function that determines the cropped area within the minimap of the stm topography scan. The cropping includes 
        the potential of rotation in the main topography plot and correctly rotates the cropped area within the
        minimap, with the inclusion of a vector V that demonstrates the rotation.
        
        :param xmin: Crop x-axis initial co-ordinate in real units.
        :param xmax: Crop x-axis final co-ordinate in real units.
        :param ymin: Crop y-axis initial co-ordinate in real units.
        :param ymax: Crop y-axis final co-ordinate in real units.
        :param angle: Rotation angle in degrees.
        :return: Px, Py, V which represent the vertices of the cropped rectangle after rotation and an arrow V showing 
        the rotation vector.
        """
        # TODO The rotation of the cropped region on the minimap does not work for rotations properly.
        # - Becuase you need to continuously update the axis of rotation as the figure is rotated around because the real distance is actually changing.

        # Define the rotation matrix that will rotate the points that define the cropped image
        def rot_matrix(coord, angle):
            angle = np.deg2rad(angle)
            rotMatrix = np.matrix([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
            return rotMatrix * np.matrix([[coord[0]], [coord[1]]])

        # Defining the four points of the cropped rectangles and shifting them to the origins centre of rotation
        # - Finding the position of the center of rotation
        Ox = 0.5 * self.selected_data[self.scan_dir].info['xreal']
        Oy = 0.5 * self.selected_data[self.scan_dir].info['yreal']
        # - Shifting all the points to the axis of rotation to perform the rotation
        P0 = np.array([xmin - Ox, ymin - Oy])
        P1 = np.array([xmin - Ox, ymax - Oy])
        P2 = np.array([xmax - Ox, ymax - Oy])
        P3 = np.array([xmax - Ox, ymin - Oy])

        # Applying a rotation of 'angle' degrees to the vertices of the rectangle to align it properly with the rotation
        rot_P0 = rot_matrix(P0, angle)
        rot_P1 = rot_matrix(P1, angle)
        rot_P2 = rot_matrix(P2, angle)
        rot_P3 = rot_matrix(P3, angle)

        # Shifting all the points back to their initial positions, after rotation
        rot_P0[0] = rot_P0[0] + Ox
        rot_P0[1] = rot_P0[1] + Oy
        rot_P1[0] = rot_P1[0] + Ox
        rot_P1[1] = rot_P1[1] + Oy
        rot_P2[0] = rot_P2[0] + Ox
        rot_P2[1] = rot_P2[1] + Oy
        rot_P3[0] = rot_P3[0] + Ox
        rot_P3[1] = rot_P3[1] + Oy
        # Converting from real units to pixel units for the image rotated, cropping operation
        rot_P0[0] = self.nm2pnt(rot_P0[0], self.selected_data)
        rot_P0[1] = self.nm2pnt(rot_P0[1], self.selected_data)
        rot_P1[0] = self.nm2pnt(rot_P1[0], self.selected_data)
        rot_P1[1] = self.nm2pnt(rot_P1[1], self.selected_data)
        rot_P2[0] = self.nm2pnt(rot_P2[0], self.selected_data)
        rot_P2[1] = self.nm2pnt(rot_P2[1], self.selected_data)
        rot_P3[0] = self.nm2pnt(rot_P3[0], self.selected_data)
        rot_P3[1] = self.nm2pnt(rot_P3[1], self.selected_data)
        # Extracting all the x-values for the points
        Px = np.array([rot_P0.item(0), rot_P1.item(0), rot_P2.item(0), rot_P3.item(0)])
        Py = np.array([rot_P0.item(1), rot_P1.item(1), rot_P2.item(1), rot_P3.item(1)])
        # Finding the vector arrow to represent the rotation
        Vbase = rot_P0 + 0.5*(rot_P3-rot_P0) + 0.5*(rot_P1-rot_P0)
        Vtip = 0.5*(rot_P1 - rot_P0)
        V = np.array([Vbase.item(0), Vbase.item(1), Vtip.item(0), Vtip.item(1)])
        # Return the necessary points and components
        return Px, Py, V

    def topo_flip(self, flat_file, xflip, yflip):
        """
        Create a copy of the flat file, flipped in either the left-right (x) and/or up-down (y) direction.

        :param flat_file: An instance of an Omicron flat file.
        :param xflip: Boolean as to whether a left-right flip should be performed.
        :param yflip: Boolean as to whether a up-down flip should be performed.
        :return: New flat file instance, with flipped image data if necessary.
        """
        # Create a new deep copy of the flat file
        flat_file_copy = deepcopy(flat_file)

        # For each scan direction in the flat file, perform the horizontal and vertical flips where necessary
        # - If both a horizontal (up-down) and vertical (left-right) flip are performed
        if xflip and yflip:
            for scan_dir in flat_file_copy:
                scan_dir.data = np.fliplr(scan_dir.data)
                scan_dir.data = np.flipud(scan_dir.data)
            # - Return new flat file instance.
            return flat_file_copy
        # - If just a vertical (left-right) flip is performed
        elif xflip:
            for scan_dir in flat_file_copy:
                scan_dir.data = np.fliplr(scan_dir.data)
                # - Return new flat file instance.
                return flat_file_copy
        # - If just a horizontal (up-down) flip is performed
        elif yflip:
            for scan_dir in flat_file_copy:
                scan_dir.data = np.flipud(scan_dir.data)
            # - Return new flat file instance.
            return flat_file_copy
        # - Else, if no flips are performed, return the original data back
        else:
            return flat_file_copy

    def topo_plot(self, flat_file, ax, scan_dir=0, cmap=None, vmin=None, vmax=None, smooth=None):
        """
        Function to plot the main, selected STM topographic data.

        Arguments
        :param flat_file: An instance of an Omicron topography flat file.
        :param ax:          The axes upon which to make the topography plot.
        
        Optional Arguments
        :param scan_dir:    Define which scan direction to use. fwd_up=0, bwd_up=1, fwd_dwn=2, bwd_dwn=3
        :param cmap:        Matplotlib colormap name.
        :param vmin:        Use to manually define the minimum value of the colour scale.
        :param vmax:        Use to manually define the maximum value of the colour scale.
        :param smooth:      If smoothing should be applied.
        """
        # Initialising the constants to be used for plotting
        # - Set minimum value of the topography scan to zero and convert to nanometers
        figure_data = (flat_file[scan_dir].data - np.amin(flat_file[scan_dir].data)) / PC["nano"]
        # - Only allowing four x, y and z ticks to appear
        xy_ticks = 4
        z_ticks = 4
        # - Setting the default parameters for the color-map and color-scale
        if cmap is None:
            cmap = 'hot'
        if vmin is None:
            vmin = np.amin(figure_data)
        if vmax is None:
            vmax = 1.25 * np.amax(figure_data)
            # - If no scan is performed such that vmax is globally zero, then to avoid an error, set it to unity
            if vmax == 0:
                vmax = 1

        # Plotting the topography image
        if smooth:
            cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal',
                            interpolation="gaussian")
        else:
            cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        # Defining the x- and y-axes ticks
        # - Extract the x, y units from the flat file
        xy_units = flat_file[scan_dir].info['unitxy']
        # - Extract the total number of x, y pixels from the flat file (total number of points in the scan)
        x_res = flat_file[scan_dir].info['xres']
        y_res = flat_file[scan_dir].info['yres']
        # - Extract the x, y real units from the flat file (maximum size of the scan in real, integer units)
        x_max = flat_file[scan_dir].info['xreal']
        y_max = flat_file[scan_dir].info['yreal']
        x_min = flat_file[scan_dir].info['xreal_min']
        y_min = flat_file[scan_dir].info['yreal_min']
        # - Setting the x-ticks locations by input
        ax.set_xticks([x for x in np.arange(0, x_res + 1, x_res / xy_ticks)])
        # - Setting the x-tick labels by rounding the numbers to one decimal place
        ax.set_xticklabels(
            [str(np.round(x, 1)) for x in np.arange(x_min, x_max + 1, (x_max-x_min) / xy_ticks)], fontsize=13)
        # - Setting the y-ticks locations by input
        ax.set_yticks([y for y in np.arange(0, y_res + 1, y_res / xy_ticks)])
        # - Setting the y-tick labels by rounding the numbers to one decimal place
        ax.set_yticklabels(
            [str(np.round(y, 1)) for y in np.arange(y_min, y_max + 1, (y_max-y_min) / xy_ticks)], fontsize=13)
        # Labelling the x- and y-axes with the units given from the flat file
        ax.set_xlabel('x /' + xy_units, size=18, weight='bold')
        ax.set_ylabel('y /' + xy_units, size=18, weight='bold')
        # Adding a title to the graph
        ax.set_title(flat_file[scan_dir].info['runcycle'][:-1] + ' : ' + self.scan_dict_inv[scan_dir],
                     fontsize=18, weight='bold')
        # Setting the scale-bar properties
        # - Defining the size and location of the scale bar
        sbar_xloc_max = x_res - 0.5 * (x_res / 10)
        sbar_xloc_min = x_res - 1.5 * (x_res / 10)
        sbar_loc_text = sbar_xloc_max - 0.5 * (x_res / 10)
        # - Plotting the scale-bar and its unit text
        ax.plot([sbar_xloc_min, sbar_xloc_max], [0.02 * y_res, 0.02 * y_res], 'k-', linewidth=5)
        ax.text(sbar_loc_text, 0.03 * y_res, str(np.round(x_max / 10, 2)) + xy_units, weight='bold', ha='center')
        # Setting the colorbar properties
        # - Define the colorbar ticks
        cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]
        # - Add labels to the colorbar ticks
        cbar_ticklabels = [str(np.round(z, 2)) for z in
                           np.arange(vmin, vmax + 1, vmax / z_ticks)]
        # - Create the colorbar next to the primary topography image
        cbar = plt.colorbar(cax, ticks=cbar_ticks, fraction=0.025, pad=0.01)
        cbar.ax.set_yticklabels(cbar_ticklabels, size=16)                       # Set colorbar tick labels
        cbar.set_label('Height [nm]', size=18, weight='bold')                   # Set colorbar label

    def minimap_topo_plot(self, flat_file, ax, scan_dir=0, cmap=None, vmin=None, vmax=None):
        """
        Function to plot the minimap version of the selected STM topographic data to show the area's over which 
        cropping and plane-subtraction has been peformed.

        Arguments
        :param flat_file: An instance of an Omicron topography flat file.
        :param ax:          The axes upon which to make the topography plot.
        
        Optional Arguments
        :param scan_dir:    Define which scan direction to use. fwd_up=0, bwd_up=1, fwd_dwn=2, bwd_dwn=3
        :param cmap:        Matplotlib colormap name.
        :param vmin:        Use to manually define the minimum value of the colour scale.
        :param vmax:        Use to manually define the maximum value of the colour scale.
        """
        # Set minimum value of the topography scan to zero and convert to nanometers
        figure_data = (flat_file[scan_dir].data - np.amin(flat_file[scan_dir].data)) / PC["nano"]
        # - Only allowing four x, y and z ticks to appear
        xy_ticks = 4
        z_ticks = 4
        # Setting the default parameters for the color-map and color-scale
        if cmap is None:
            cmap = 'hot'
        if vmin is None:
            vmin = np.amin(figure_data)
        if vmax is None:
            vmax = 1.25 * np.amax(figure_data)
            # - If no scan is peformed such that vmax is zero, then to avoid an error, set it to one
            if vmax == 0:
                vmax = 1
        # Plotting the topography image
        cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        # Extract the x, y units from the flat file
        xy_units = flat_file[scan_dir].info['unitxy']
        # Extract the total number of x, y pixels from the flat file (total number of points in the scan)
        x_res = flat_file[scan_dir].info['xres']
        y_res = flat_file[scan_dir].info['yres']
        # Extract the x, y real units from the flat file (maximum size of the scan in real, integer units)
        x_max = int(flat_file[scan_dir].info['xreal'])
        y_max = int(flat_file[scan_dir].info['yreal'])
        # - Setting the x-ticks locations by input
        ax.set_xticks([x for x in np.arange(0, x_res + 1, x_res / xy_ticks)])
        # - Setting the x-tick labels by rounding the numbers to one decimal place
        ax.set_xticklabels(
            [str(np.round(x, 1)) for x in np.arange(0, x_max + 1, x_max / xy_ticks)], fontsize=10)
        # - Setting the y-ticks locations by input
        ax.set_yticks([y for y in np.arange(0, y_res + 1, y_res / xy_ticks)])
        # - Setting the y-tick labels by rounding the numbers to one decimal place
        ax.set_yticklabels(
            [str(np.round(y, 1)) for y in np.arange(0, y_max + 1, y_max / xy_ticks)], fontsize=10)
        # Labelling the x- and y-axes with the units given from the flat file
        ax.set_xlabel('x /' + xy_units, size=10, weight='bold')
        ax.set_ylabel('y /' + xy_units, size=10, weight='bold')
        # Setting the scale-bar properties
        # - Defining the size and location of the scale bar
        sbar_xloc_max = x_res - 0.5 * (x_res / 10)
        sbar_xloc_min = x_res - 1.5 * (x_res / 10)
        sbar_loc_text = sbar_xloc_max - 0.5 * (x_res / 10)
        # - Plotting the scale-bar and its unit text
        ax.plot([sbar_xloc_min, sbar_xloc_max], [0.02 * y_res, 0.02 * y_res], 'k-', linewidth=5)
        ax.text(sbar_loc_text, 0.04 * y_res, str(np.round(x_max / 10, 2)) + xy_units, weight='bold', ha='center')
        # Setting the colorbar properties
        # - Define the colorbar ticks
        cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]
        # - Add labels to the colorbar ticks
        cbar_ticklabels = [str(np.round(z, 2)) for z in
                           np.arange(vmin, vmax + 1, vmax / z_ticks)]
        # - Create the colorbar next to the primary topography image
        cbar = plt.colorbar(cax, ticks=cbar_ticks, fraction=0.025, pad=0.00, orientation="vertical")
        cbar.ax.set_yticklabels(cbar_ticklabels, size=10)
        # Set the x-and y-limits to be equal to the size of the image
        ax.set_xlim(0, x_res)
        ax.set_ylim(0, y_res)
        # Adding a legend to show the color of the plane and cropped polygons
        ax.legend(handles=list([patch.Patch(color='blue', label='plane'),
                                patch.Patch(color='green', label='crop')]),
                  loc='best', prop={'size': 8}, frameon=False)
        # Adding a grid to the minimap
        ax.grid(True, color='gray')

        #  Add text to the plot for all the important information
        plt.gcf().text(0.35, 0.86, flat_file[scan_dir].info['runcycle'][:-1] + ' : ' +
                       flat_file[scan_dir].info['direction'], fontsize=14, weight='bold')
        plt.gcf().text(0.35, 0.84, flat_file[scan_dir].info['date'], fontsize=14, weight='bold')

        plt.gcf().text(0.35, 0.82, 'Comments: ' + flat_file[scan_dir].info['comment'], fontsize=14)
        plt.gcf().text(0.35, 0.75, 'Current set-point: ' + str(flat_file[scan_dir].info['current']) + str('A'),
                       fontsize=14)
        plt.gcf().text(0.35, 0.73, 'Voltage bias: ' + str(np.round(flat_file[scan_dir].info['vgap'], 2)) + str('V'),
                       fontsize=14)
        plt.gcf().text(0.35, 0.71, '[' + str(np.int(x_res)) + 'x' + str(np.int(y_res)) + '] $pts$',
                       fontsize=14)
        plt.gcf().text(0.35, 0.69, '[' + str(np.round(x_max, 1)) + 'x' + str(np.round(y_max, 1)) + '] $' + xy_units
                       + '^2$', fontsize=14)

    def other_topo_plots(self, flat_file, ax, scan_dir=0, cmap=None, vmin=None, vmax=None):
        """
        Function to plot all the other STM topographic data that has not been selected.

        Arguments
        :param flat_file: An instance of an Omicron topography flat file.
        :param ax:          The axes upon which to make the topography plot.
        
        Optional Arguments
        :param scan_dir:    Define which scan direction to use. fwd_up=0, bwd_up=1, fwd_dwn=2, bwd_dwn=3
        :param cmap:        Matplotlib colormap name.
        :param vmin:        Use to manually define the minimum value of the colour scale.
        :param vmax:        Use to manually define the maximum value of the colour scale.
        """
        # Set minimum value of the topography scan to zero and convert to nanometers
        figure_data = (flat_file[scan_dir].data - np.amin(flat_file[scan_dir].data)) / PC["nano"]
        # Setting the default parameters for the color-map and color-scale
        if cmap is None:
            cmap = 'hot'
        if vmin is None:
            vmin = np.amin(figure_data)
        if vmax is None:
            vmax = 1.25 * np.amax(figure_data)
            # - If no scan is peformed such that vmax is zero, then to avoid an error, set it to one
            if vmax == 0:
                vmax = 1
        # Plotting the topography image
        cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        # Removing all the x- and y-axes ticks
        ax.axis("off")
        # Extract the total number of x, y pixels from the flat file (total number of points in the scan)
        x_res = flat_file[scan_dir].info['xres']
        y_res = flat_file[scan_dir].info['yres']
        # Adding a title to the graph
        ax.set_title(flat_file[scan_dir].info['runcycle'][:-1] + ' : ' + self.scan_dict_inv[scan_dir],
                     fontsize=8, weight='bold')
        # Setting the scale-bar properties
        # - Defining the size and location of the scale bar
        sbar_xloc_max = x_res - 0.5 * int(x_res / 10)
        sbar_xloc_min = x_res - 1.5 * int(x_res / 10)
        # - Plotting the scale-bar and its unit text
        ax.plot([sbar_xloc_min, sbar_xloc_max], [0.02 * y_res, 0.02 * y_res], 'k-', linewidth=3)
        # If there is no scan performed, add text to say this
        if np.sum(np.sum(figure_data)) == 0:
            ax.text(0.5*x_res, 0.5*y_res, 'NO SCAN TAKEN', fontsize=15, color=[1, 1, 1], weight='bold', rotation=45,
                    ha='center', va='center')

    def get_widgets(self):
        """
        Creates a variety of widgets to be interacted with for the analysis of the topography images.
        """
        # Select Dropdown widget to select the the flat-files to be analysed
        data_select_0 = ipy.Dropdown(options=self.file_alias, description="$$Raw\,Topo\,files$$",
                                     continuous_update=False,
                                     layout=ipy.Layout(display='inline-flex', flex_flow='column',
                                                       align_items='stretch', align_content='stretch',
                                                       justify_content='center', height='70px', width="95%"))
        # Toggle Buttons widget to select the type of analysis to be performed
        scan_type_0 = ipy.ToggleButtons(options=['up-fwd', 'up-bwd', 'down-fwd', 'down-bwd'], value='up-fwd',
                                        description="$$Scan\,type$$", continuous_update=False,
                                        layout=ipy.Layout(display='inline-flex', flex_flow='column',
                                                          align_items='stretch', align_content='stretch',
                                                          justify_content='center', height='90%', width="95%"))

        # Toggle Buttons widget to select the type of analysis to be performed
        level_type_1 = ipy.ToggleButtons(options=['None', 'Line-wise', 'Local plane'], value='None',
                                         description='$$Leveling:$$', continuous_update=False,
                                         layout=ipy.Layout(display='inline-flex', flex_flow='row',
                                                           align_items='stretch', align_content='stretch',
                                                           height='50%', width="100%"))
        # Float text widgets to choose the x and y points for the local plane subtraction
        # - Defining all the x and y co-ordinate pairs for local-place selection
        x0_coord_1 = ipy.FloatSlider(value=0, min=0, max=500, description="$x_0$", color='black',
                                     continuous_update=False,
                                     layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))
        y0_coord_1 = ipy.FloatSlider(value=0, min=0, max=500, description="$y_0$", color='black',
                                     continuous_update=False,
                                     layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))
        x1_coord_1 = ipy.FloatSlider(value=500, min=0, max=500, description="$x_1$", color='black',
                                     continuous_update=False,
                                     layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))
        y1_coord_1 = ipy.FloatSlider(value=500, min=0, max=500, description="$y_1$", color='black',
                                     continuous_update=False,
                                     layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))

        # Add a label for image cropping
        label_2 = ipy.Label(value='$$Image\,Crop:$$ ', layout=ipy.Layout(width='95%', height='auto', display='flex',
                                                                         flex_flow='row', align_items='stretch'))
        # - Defining all the x and y co-ordinate pairs for local-place selection
        x0_crop_2 = ipy.FloatSlider(value=0, min=0, max=500, description="$x_0$", color='black',
                                    continuous_update=False,
                                    layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                      align_items='stretch'))
        y0_crop_2 = ipy.FloatSlider(value=0, min=0, max=500, description="$y_0$", color='black',
                                    continuous_update=False,
                                    layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                      align_items='stretch'))
        x1_crop_2 = ipy.FloatSlider(value=0, min=0, max=500, description="$x_1$", color='black',
                                    continuous_update=False,
                                    layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                      align_items='stretch'))
        y1_crop_2 = ipy.FloatSlider(value=0, min=0, max=500, description="$y_1$", color='black',
                                    continuous_update=False,
                                    layout=ipy.Layout(width='50%', height='', display='flex', flex_flow='row',
                                                      align_items='stretch'))

        # Float Slider widget to control the rotation from 0 - 359
        rotation_2 = ipy.FloatSlider(value=0, min=0, max=359, step=0.5, description="$Rotation$: ",
                                     continuous_update=False,
                                     layout=ipy.Layout(width='50%', height='auto', display='flex',
                                                       flex_flow='row', align_items='stretch'))

        # Checkbox widget to flip along x or y axes
        xflip_2 = ipy.Checkbox(description="$x$-flip: ")
        yflip_2 = ipy.Checkbox(description="$y$-flip: ")
        # Textbox widget that control the colormap being used
        col_text_2 = ipy.Text(value='hot', description="$$Color-map: $$", color='black', continuous_update=False,
                              layout=ipy.Layout(width='30%', height='', display='flex', flex_flow='row',
                                                align_items='stretch'))
        autocontrast_2 = ipy.Checkbox(description="$$Lock\,con.:$$", value=True,
                                      layout=ipy.Layout(width='95%', height='auto', display='flex',
                                                        flex_flow='row', align_items='stretch'))
        # Float Range widget that controls the minimum and maximum contrast
        coarse_cont_2 = ipy.FloatSlider(value=10, min=0, max=100, step=1, description="$$Coarse\,con.:$$",
                                        continuous_update=False,
                                        layout=ipy.Layout(width='50%', height='auto', display='flex',
                                                          flex_flow='row', align_items='stretch'))
        # Float Range widget that controls the minimum and maximum contrast
        fine_cont_2 = ipy.FloatSlider(value=0, min=0, max=1, step=0.01, description="$$Fine\,con.:$$",
                                      continuous_update=False,
                                      layout=ipy.Layout(width='50%', height='auto', display='flex',
                                                        flex_flow='row', align_items='stretch'))

        # Checkbox widget to determine if there should be smoothing
        smooth_2 = ipy.Checkbox(description="$$Smooth:$$", value=False,
                                layout=ipy.Layout(width='95%', height='auto', display='flex',
                                                  flex_flow='row', align_items='stretch'))

        # Creating a tab widget to hold all the operational information
        all_tabs = ipy.Tab([ipy.VBox([level_type_1,
                                      ipy.HBox([x0_coord_1, y0_coord_1]),
                                      ipy.HBox([x1_coord_1, y1_coord_1])]
                                     ),
                            ipy.VBox([label_2,
                                      ipy.HBox([x0_crop_2, y0_crop_2]),
                                      ipy.HBox([x1_crop_2, y1_crop_2]),
                                      rotation_2,
                                      ipy.HBox([xflip_2, yflip_2]),
                                      smooth_2, col_text_2, autocontrast_2, coarse_cont_2, fine_cont_2])
                            ],
                           layout=ipy.Layout(display='inline-flex', flex_flow='column', align_items='stretch',
                                             width='100%', height='100%'))
        all_tabs.set_title(0, 'Level Operations')
        all_tabs.set_title(1, 'Image Operations')

        # Defining a global widget box to hold all of the widgets
        self.widgets = ipy.HBox([ipy.VBox([data_select_0, scan_type_0],
                                          layout=ipy.Layout(display='inline-flex', flex_flow='column',
                                                            border='solid 0.5px', align_items='stretch',
                                                            width='15.5%', height='100%')),
                                 all_tabs])

    def update_function(self, chosen_data, scan_dir, level_type, p_x0, p_x1, p_y0, p_y1,
                        c_x0, c_x1, c_y0, c_y1,
                        rot, xflip, yflip, smooth, colormap, autocontrast, coarse_cont, fine_cont):
        """
        Updates the topography scans and analysis using the defined widgets.
        """
        # Obtain the file name selected by the user
        self.selected_file = chosen_data

        # Extracting the flat-file instances of the selected file and it's 'scan_dir' parameter
        self.selected_data_extract(scan_dir)

        # Defining all the properties for the main topography plot
        # - Converting from real units to pixel units for the local plane subtraction operation
        pix_p_x0 = self.nm2pnt(p_x0, self.selected_data)
        pix_p_x1 = self.nm2pnt(p_x1, self.selected_data)
        pix_p_y0 = self.nm2pnt(p_y0, self.selected_data, axis='y')
        pix_p_y1 = self.nm2pnt(p_y1, self.selected_data, axis='y')

        # - Define a dictionary of all the image properties
        self.image_props = {'leveling': level_type, "real plane": np.array([p_x0, p_x1, p_y0, p_y1]),
                            "real crop": np.array([c_x0, c_x1, c_y0, c_y1]),
                            "rotation": rot, "x flip": xflip, "y flip": yflip, 'smooth': smooth,
                            "colormap": colormap, "auto contrast": autocontrast,
                            "contrast": float(coarse_cont + fine_cont)}

        # Writing labels to specify what analysis has been performed
        analysis_string = ''
        if level_type == 'Line-wise':
            analysis_string += 'LW, \n'
        elif level_type == 'Local plane':
            analysis_string += 'LP, \n'
        if rot != 0:
            analysis_string += 'R{' + str(rot) + 'deg}, \n'
        if xflip:
            analysis_string += 'LR flip, \n'
        if yflip:
            analysis_string += 'UD flip, \n'
        if smooth:
            analysis_string += 'Smthd.'

        # Defining all the properties for the minimap topography plot
        # - Extracting the vertices of the rectangle if the image is rotated and then cropped
        Px, Py, V = self.minimap_crop(c_x0, c_x1, c_y0, c_y1, rot)

        # If no background subtraction is performed
        if level_type == 'None':
            # Executing the level operations
            self.leveled_data = self.selected_data

            # Executing the image operations
            topo_f = self.topo_flip(self.leveled_data, xflip, yflip)
            topo_fr = self.topo_rotate(topo_f, rot)
            topo_frc = self.topo_crop(topo_fr, c_x0, c_x1, c_y0, c_y1)
            self.final_data = topo_frc

            # Plotting the main topography scan selected
            plt.subplots(figsize=(22, 10))
            ax1 = plt.subplot(1, 2, 2)
            if autocontrast:
                self.topo_plot(self.final_data, ax1, self.scan_dir, colormap, None, None, smooth)
            else:
                self.topo_plot(self.final_data, ax1, self.scan_dir, colormap, None, self.image_props["contrast"],
                               smooth)

            # Plotting the minimap of the topography scan selected
            ax2 = plt.subplot(2, 4, 6)
            self.minimap_topo_plot(self.leveled_data, ax2, self.scan_dir, colormap)
            # - Plotting the area over which the cropping is performed
            if rot == 0 or rot == 90 or rot == 180 or rot == 270:
                Px = np.append(Px, Px[0])
                Py = np.append(Py, Py[0])
                ax2.plot(Px, Py, 'go-', alpha=1, markersize=4, linewidth=2)
                ax2.arrow(V[0], V[1], V[2], V[3], head_width=7, head_length=5, fc='g', ec='g', linewidth=3)
                ax2.fill_between(Px, Py, color='green', alpha=0.4)
            # Adding the analysis information
            plt.gcf().text(0.35, 0.62, analysis_string, fontsize=12, va='top', ha='left')

            # Plotting all the other topography scans
            plt.subplots(figsize=(10, 3))
            ax3 = plt.subplot(1, 3, 1)
            self.other_topo_plots(self.leveled_data, ax3, self.scan_dir_not[0], colormap)
            ax4 = plt.subplot(1, 3, 2)
            self.other_topo_plots(self.leveled_data, ax4, self.scan_dir_not[1], colormap)
            ax5 = plt.subplot(1, 3, 3)
            self.other_topo_plots(self.leveled_data, ax5, self.scan_dir_not[2], colormap)

        # If line-wise background subtraction is performed
        elif level_type == "Line-wise":
            # Executing the level operations
            self.leveled_data = self.topo_linewise(self.selected_data, self.scan_dir)

            # Executing the image operations
            topo_f = self.topo_flip(self.leveled_data, xflip, yflip)
            topo_fr = self.topo_rotate(topo_f, rot)
            topo_frc = self.topo_crop(topo_fr, c_x0, c_x1, c_y0, c_y1)
            self.final_data = topo_frc

            # Plotting the main topography scan selected
            plt.subplots(figsize=(22, 10))
            ax1 = plt.subplot(1, 2, 2)
            if autocontrast:
                self.topo_plot(self.final_data, ax1, self.scan_dir, colormap, None, None, smooth)
            else:
                self.topo_plot(self.final_data, ax1, self.scan_dir, colormap, None, self.image_props["contrast"],
                               smooth)

            # Plotting the minimap of the topography scan selected
            ax2 = plt.subplot(2, 4, 6)
            self.minimap_topo_plot(self.leveled_data, ax2, self.scan_dir, colormap)
            # - Plotting the area over which the cropping is performed
            if rot == 0 or rot == 90 or rot == 180 or rot == 270:
                Px = np.append(Px, Px[0])
                Py = np.append(Py, Py[0])
                ax2.plot(Px, Py, 'go-', alpha=1, markersize=4, linewidth=2)
                ax2.arrow(V[0], V[1], V[2], V[3], head_width=7, head_length=5, fc='g', ec='g', linewidth=3)
                ax2.fill_between(Px, Py, color='green', alpha=0.4)
            # Adding the analysis information
            plt.gcf().text(0.35, 0.62, analysis_string, fontsize=12, va='top', ha='left')

            # Plotting all the other topography scans
            plt.subplots(figsize=(10, 3))
            ax3 = plt.subplot(1, 3, 1)
            self.other_topo_plots(self.leveled_data, ax3, self.scan_dir_not[0], colormap)
            ax4 = plt.subplot(1, 3, 2)
            self.other_topo_plots(self.leveled_data, ax4, self.scan_dir_not[1], colormap)
            ax5 = plt.subplot(1, 3, 3)
            self.other_topo_plots(self.leveled_data, ax5, self.scan_dir_not[2], colormap)

        # If local plane background subtraction is performed
        elif level_type == "Local plane":
            # Executing the level operations
            self.leveled_data = self.topo_localplane(self.selected_data, self.scan_dir, p_x0, p_x1, p_y0, p_y1)

            # Executing the image operations
            topo_f = self.topo_flip(self.leveled_data, xflip, yflip)
            topo_fr = self.topo_rotate(topo_f, rot)
            topo_frc = self.topo_crop(topo_fr, c_x0, c_x1, c_y0, c_y1)
            self.final_data = topo_frc

            # Plotting the main topography scan selected
            plt.subplots(figsize=(22, 10))
            ax1 = plt.subplot(1, 2, 2)
            if autocontrast:
                self.topo_plot(self.final_data, ax1, self.scan_dir, colormap, None, None, smooth)
            else:
                self.topo_plot(self.final_data, ax1, self.scan_dir, colormap, None, self.image_props["contrast"],
                               smooth)

            # Plotting the minimap of the topography scan selected
            ax2 = plt.subplot(2, 4, 6)
            self.minimap_topo_plot(self.leveled_data, ax2, self.scan_dir, colormap)
            # - Plotting the area over which the plane subtraction is performed
            ax2.plot([pix_p_x0, pix_p_x1, pix_p_x1, pix_p_x0], [pix_p_y0, pix_p_y0, pix_p_y1, pix_p_y1], 'bo',
                     alpha=0.4)
            ax2.fill_between([pix_p_x0, pix_p_x1], [pix_p_y0, pix_p_y0], [pix_p_y1, pix_p_y1], color='blue',
                             alpha=0.4)
            # - Plotting the area over which the cropping is performed
            if rot == 0 or rot == 90 or rot == 180 or rot == 270:
                Px = np.append(Px, Px[0])
                Py = np.append(Py, Py[0])
                ax2.plot(Px, Py, 'go-', alpha=1, markersize=4, linewidth=2)
                ax2.arrow(V[0], V[1], V[2], V[3], head_width=7, head_length=5, fc='g', ec='g', linewidth=3)
                ax2.fill_between(Px, Py, color='green', alpha=0.4)
            # Adding the analysis information
            plt.gcf().text(0.35, 0.62, analysis_string, fontsize=12, va='top', ha='left')

            # Plotting all the other topography scans
            plt.subplots(figsize=(10, 3))
            ax3 = plt.subplot(1, 3, 1)
            self.other_topo_plots(self.leveled_data, ax3, self.scan_dir_not[0], colormap)
            ax4 = plt.subplot(1, 3, 2, sharey=ax3)
            self.other_topo_plots(self.leveled_data, ax4, self.scan_dir_not[1], colormap)
            ax5 = plt.subplot(1, 3, 3, sharey=ax4)
            self.other_topo_plots(self.leveled_data, ax5, self.scan_dir_not[2], colormap)

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
        # - LHS widgets
        chosen_data = self.widgets.children[0].children[0]
        scan_dir = self.widgets.children[0].children[1]
        # - LEVEL OPERATION widgets
        level_type = self.widgets.children[1].children[0].children[0]
        p_x0 = self.widgets.children[1].children[0].children[1].children[0]
        p_y0 = self.widgets.children[1].children[0].children[1].children[1]
        p_x1 = self.widgets.children[1].children[0].children[2].children[0]
        p_y1 = self.widgets.children[1].children[0].children[2].children[1]
        # - IMAGE OPERATIONS widgets
        c_x0 = self.widgets.children[1].children[1].children[1].children[0]
        c_y0 = self.widgets.children[1].children[1].children[1].children[1]
        c_x1 = self.widgets.children[1].children[1].children[2].children[0]
        c_y1 = self.widgets.children[1].children[1].children[2].children[1]
        rot = self.widgets.children[1].children[1].children[3]
        xflip = self.widgets.children[1].children[1].children[4].children[0]
        yflip = self.widgets.children[1].children[1].children[4].children[1]
        smooth = self.widgets.children[1].children[1].children[5]
        colormap = self.widgets.children[1].children[1].children[6]
        autocontrast = self.widgets.children[1].children[1].children[7]
        coarse_cont = self.widgets.children[1].children[1].children[8]
        fine_cont = self.widgets.children[1].children[1].children[9]

        # Define the attribute to continuously update the figure, given the user interaction
        self.output = ipy.interactive(self.update_function, chosen_data=chosen_data, scan_dir=scan_dir,
                                      level_type=level_type, p_x0=p_x0, p_x1=p_x1, p_y0=p_y0, p_y1=p_y1,
                                      c_x0=c_x0, c_x1=c_x1, c_y0=c_y0, c_y1=c_y1,
                                      rot=rot, xflip=xflip, yflip=yflip, smooth=smooth, colormap=colormap,
                                      autocontrast=autocontrast, coarse_cont=coarse_cont, fine_cont=fine_cont,
                                      continous_update=False)

        # Display the final output of the widget interaction
        display(self.output.children[-1])


# 2.1 - Defining the class object that will use the analysed topography scan to look at 1D line profiles
class STT_lineprof(object):
    def __init__(self, topo_data):
        """
        Defines the initialisation of the class object.
        topo_data:     The 'STT' class object of the selected, analysed and final topography data.
        """
        # 2.1.1 -  Extract the final stm topography image from the first stage of the analysis
        self.topo_data = topo_data
        self.topo_scan_dir = self.topo_data.scan_dict_inv[self.topo_data.scan_dir]
        self.topo_data = self.topo_data.final_data[self.topo_data.scan_dir]
        self.topo_xmin = self.topo_data.info['xreal_min']
        self.topo_xmax = self.topo_data.info['xreal']
        self.topo_ymin = self.topo_data.info['yreal_min']
        self.topo_ymax = self.topo_data.info['yreal']

        # 2.1.2 - Line profile information
        self.line_points = None         # Defining the real points of where the line-profile is taken
        self.line_pix_points = None     # Defining the pixel points of where the line-profile is taken
        self.line_prof_len = None       # Defining the real length of the line-profile that is taken
        self.line_prof_x = None         # Defining the x-domain of the line-profile taken
        self.line_prof_y = None         # Defining the y-domain of the line-profile taken

        # 2.1.3 - User interaction
        self.widgets = None         # Widget object to hold all pre-defined widgets
        self.get_widgets()          # Function to get all of the pre-defined widgets
        self.output = None          # Output to the user interaction with widgets
        self.user_interaction()     # Function to allow continuous user interaction

    def nm2pnt(self, nm, flat_file, axis='x'):
        """
        Convert between nanometers and corresponding pixel number for a given Omicron flat file.

        :param nm: Nanometer value.
        :param flat_file: Instance of the selected Omicron flat file.
        :param axis: Plot axis of nm point. Must be either 'x' or 'y'.
        :return: Pixel number for nanometer value.
        """
        if axis == 'x':
            inc = flat_file.info['xinc']
        elif axis == 'y':
            inc = flat_file.info['yinc']

        pnt = np.int(np.round(nm / inc))

        if pnt < 0:
            pnt = 0
        if axis == 'x':
            if pnt > flat_file.info['xres']:
                pnt = flat_file.info['xres']
        elif axis == 'y':
            if pnt > flat_file.info['yres']:
                pnt = flat_file.info['yres']

        return pnt

    def profile(self, points, flat_file, num_points=1000):
        """
        Extract a line profile from the given flat file and list of x, y co-ordinates.

        Arguments
        :param points: List of x, y co-ordinate pairs that define the line profile in real units.
        :param flat_file: An instance of a Omicron flat file.

        Optional Arguments
        :param num_points: Number of points in the line profile data.
        :return: Line profile z-data, Line profile distance-data.
        """
        # Finding the total length of the line profile selected by the user
        length = 0
        for p in range(len(points) - 1):
            length += np.sqrt((points[p + 1, 0] - points[p, 0]) ** 2 + (points[p + 1, 1] - points[p, 1]) ** 2)
        # Finding the total number of pixels in the x- and y-direction
        x_len = len(flat_file.data[0])
        y_len = len(flat_file.data)
        # Finding the points in terms of the x- and y-pixels
        for point in range(len(points)):
            points[point][0] = self.nm2pnt(points[point][0] - self.topo_xmin, flat_file, axis='x')
            points[point][1] = self.nm2pnt(points[point][1] - self.topo_ymin, flat_file, axis='y')
            if points[point][0] >= x_len:
                points[point][0] = x_len - 1
            if points[point][1] >= y_len:
                points[point][1] = y_len - 1
        # Defining a function that will determine the value of z for a given co-ordinate (x, y)
        def line(coords, flat_file):

            x0, y0 = coords[0]
            x1, y1 = coords[1]
            num = num_points
            x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

            zi = flat_file.data[y.astype(np.int), x.astype(np.int)]

            return zi
        # Determination of the line profile data
        profile_data = np.array([])
        for pair in range(len(points) - 1):
            profile_data = np.append(profile_data, line([points[pair], points[pair + 1]], flat_file))
        # Return the line profile data and its total length in real units
        return profile_data, length

    def profile_plot(self, ax, profile_data, length):
        """
        Create a plot of the given line profile data.

        Arguments
        :param ax: The axes upon which to make the topography plot.
        :param profile_data: List of line profile data.
        :param length: Nanometer length of line profile.
        

        Optional Arguments
        :param xticks: Number of x-axis ticks.
        :param yticks: Number of y-axis ticks.
        :return:
        """
        # Converting the apparent height in terms of nano-meters
        profile_data = profile_data / PC['nano']
        # Plot the line profile data
        ax.plot(profile_data, 'ko-', markersize=5, linewidth=2)

        # - Only allowing four x and y ticks to appear
        x_ticks = 5
        y_ticks = 5
        # Set the x-axis ticks from the number defined
        ax.set_xticks([x for x in np.arange(0, len(profile_data) + 10 * 10 ** -10, len(profile_data) / x_ticks)])
        # Set the x-axis tick labels from the given profile length.
        ax.set_xticklabels([str(np.round(x, 2)) for x in np.arange(0, length + 1, length / x_ticks)], size=15)
        # Set the x-axis label.
        ax.set_xlabel('L / nm', size=18, weight='bold')

        # Set the y-axis ticks from the number defined
        ax.set_yticks([y for y in np.arange(0, 1.2 * np.max(profile_data), np.max(profile_data) / y_ticks)])
        # Set the y-axis tick labels from the range of the profile data
        ax.set_yticklabels(
            [str(np.round(y, 2)) for y in np.arange(0, 1.2 * np.max(profile_data), np.max(profile_data) / y_ticks)],
            size=15)
        # Set the y-axis label
        ax.set_ylabel('Apparent height / nm', size=18, weight='bold')

        # Add horizontal and vertical axes lines
        ax.axhline(0, color='black', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=1.5)
        # Adding a legend to show the color of the plane and cropped polygons
        ax.legend(handles=list([patch.Patch(color='black', label='Line-profile')]),
                  loc='best', prop={'size': 15}, frameon=False)
        # Adding a grid
        ax.grid(True, color='gray')

    def topo_profile_plot(self, ax, flat_file, points, cmap=None, vmin=None, vmax=None):
        """
        Create a plot of the topography image, with the given line profile locations overlaid.

        Arguments
        :param ax: The axes upon which to make the topography plot.
        :param flat_file: An instance of an Omicron flat file.
        :param points: List of x, y co-ordinate pairs that construct the line profile, in real units.

        Optional Arguments
        :param scan_dir: Scan direction of the flat file.
        :param cmap: Pyplot color scheme to use.
        :param vmin: Z-axis minimum value.
        :param vmax: Z-axis maximum value.
        :param xy_ticks: Number of x-, y-axis ticks.
        :param z_ticks: Number of z-axis ticks.
        :return:
        """

        # Initialising the constants to be used for plotting
        # - Set minimum value of the topography scan to zero and convert to nanometers
        figure_data = (flat_file.data - np.amin(flat_file.data)) / PC["nano"]
        # - Only allowing four x, y and z ticks to appear
        xy_ticks = 4
        z_ticks = 4
        # - Setting the default parameters for the color-map and color-scale
        if cmap is None:
            cmap = 'hot'
        if vmin is None:
            vmin = np.amin(figure_data)
        if vmax is None:
            vmax = 1.25 * np.amax(figure_data)
            # - If no scan is performed such that vmax is globally zero, then to avoid an error, set it to unity
            if vmax == 0:
                vmax = 1

        # Plotting the topography image
        cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        # Plot the line profile points on the axis.
        ax.plot(points[:, 0], points[:, 1], 'bo-', markersize=8, linewidth=2.5)

        # Defining the x- and y-axes ticks
        # - Extract the x, y units from the flat file
        xy_units = flat_file.info['unitxy']
        # - Extract the total number of x, y pixels from the flat file (total number of points in the scan)
        x_res = flat_file.info['xres']
        y_res = flat_file.info['yres']
        # - Extract the x, y real units from the flat file (maximum size of the scan in real, integer units)
        x_max = flat_file.info['xreal']
        y_max = flat_file.info['yreal']
        x_min = flat_file.info['xreal_min']
        y_min = flat_file.info['yreal_min']
        # - Setting the x-ticks locations by input
        ax.set_xticks([x for x in np.arange(0, x_res + 1, x_res / xy_ticks)])
        # - Setting the x-tick labels by rounding the numbers to one decimal place
        ax.set_xticklabels(
            [str(np.round(x, 1)) for x in np.arange(x_min, x_max + 1, (x_max - x_min) / xy_ticks)], fontsize=13)
        # - Setting the y-ticks locations by input
        ax.set_yticks([y for y in np.arange(0, y_res + 1, y_res / xy_ticks)])
        # - Setting the y-tick labels by rounding the numbers to one decimal place
        ax.set_yticklabels(
            [str(np.round(y, 1)) for y in np.arange(y_min, y_max + 1, (y_max - y_min) / xy_ticks)], fontsize=13)
        # Labelling the x- and y-axes with the units given from the flat file
        ax.set_xlabel('x /' + xy_units, size=18, weight='bold')
        ax.set_ylabel('y /' + xy_units, size=18, weight='bold')
        # Adding a title to the graph
        ax.set_title(flat_file.info['runcycle'][:-1] + ' : ' + self.topo_scan_dir,
                     fontsize=18, weight='bold')
        # Setting the scale-bar properties
        # - Defining the size and location of the scale bar
        sbar_xloc_max = x_res - 0.5 * (x_res / 10)
        sbar_xloc_min = x_res - 1.5 * (x_res / 10)
        sbar_loc_text = sbar_xloc_max - 0.5 * (x_res / 10)
        # - Plotting the scale-bar and its unit text
        ax.plot([sbar_xloc_min, sbar_xloc_max], [0.02 * y_res, 0.02 * y_res], 'k-', linewidth=5)
        ax.text(sbar_loc_text, 0.03 * y_res, str(np.round(x_max / 10, 2)) + xy_units, weight='bold', ha='center')
        # Setting the colorbar properties
        # - Define the colorbar ticks
        cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]
        # - Add labels to the colorbar ticks
        cbar_ticklabels = [str(np.round(z, 2)) for z in
                           np.arange(vmin, vmax + 1, vmax / z_ticks)]
        # - Create the colorbar next to the primary topography image
        cbar = plt.colorbar(cax, ticks=cbar_ticks, fraction=0.025, pad=0.01)
        cbar.ax.set_yticklabels(cbar_ticklabels, size=13)  # Set colorbar tick labels
        cbar.set_label('Height [nm]', size=13, weight='bold')  # Set colorbar label

        # Define the limits of the plot
        ax.set_xlim([0, x_res])
        ax.set_ylim([0, y_res])

        # Adding a legend to show the color of the plane and cropped polygons
        ax.legend(handles=list([patch.Patch(color='blue', label='Line-profile')]),
                  loc='best', prop={'size': 15}, frameon=False)
        # Adding a grid
        ax.grid(True, color='gray', alpha=0.6)

    def get_widgets(self):
        """
        Creates a variety of widgets to be interacted with for the analysis of the topography images.
        """
        # Float text widgets to choose the x and y points for the line-profile
        # - Defining all the x and y co-ordinate pairs
        x0_coord_1 = ipy.FloatSlider(value=0, min=self.topo_xmin, max=self.topo_xmax, description="$x_0$",
                                     color='black', continuous_update=False,
                                     layout=ipy.Layout(width='100%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))
        y0_coord_1 = ipy.FloatSlider(value=0, min=self.topo_ymin, max=self.topo_ymax, description="$y_0$",
                                     color='black', continuous_update=False,
                                     layout=ipy.Layout(width='100%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))
        x1_coord_1 = ipy.FloatSlider(value=0, min=self.topo_xmin, max=self.topo_xmax, description="$x_1$",
                                     color='black', continuous_update=False,
                                     layout=ipy.Layout(width='100%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))
        y1_coord_1 = ipy.FloatSlider(value=0, min=self.topo_ymin, max=self.topo_ymax, description="$y_1$",
                                     color='black', continuous_update=False,
                                     layout=ipy.Layout(width='100%', height='', display='flex', flex_flow='row',
                                                       align_items='stretch'))

        # all_tabs.set_title(1, 'Sinusoidal fitting')
        # all_tabs.set_title(0, 'Single peak fitting')
        # all_tabs.set_title(1, 'Double peak fitting')

        # Defining a global widget box to hold all of the widgets
        self.widgets = ipy.HBox([ipy.VBox([ipy.HBox([x0_coord_1, y0_coord_1]),
                                           ipy.HBox([x1_coord_1, y1_coord_1])])],
                                layout=ipy.Layout(display='inline-flex', flex_flow='column', align_items='stretch',
                                                  width='60%', height=''))

    def update_function(self, l_x0, l_y0, l_x1, l_y1):
        """
        Update the line-profile figure using the interactive widgets.
        """

        # Defining the real x-, y-coords where the line profile will be taken
        points = np.array([[l_x0, l_y0],
                           [l_x1, l_y1]])
        self.line_points = points

        # Extracting the line profile, line profile domain, length and pixel points over the defined points
        self.line_prof_y, self.line_prof_len = self.profile(points, self.topo_data)
        self.line_pix_points = points
        self.line_prof_x = np.linspace(0, self.line_prof_len, len(self.line_prof_y))

        # Defining the figure size
        plt.subplots(figsize=(22, 10))

        # Plotting the main topography scan selected
        ax1 = plt.subplot(1, 2, 1)
        self.topo_profile_plot(ax1, self.topo_data, self.line_pix_points)

        # If the line points are not well defined, ignore the profile plots
        if l_x0 == l_x1 and l_y0 == l_y1:
            # Show the figure that has been created
            plt.show()
        # If the line points are well defined, plot the line profile
        else:
            # Plotting the main topography scan selected
            ax2 = plt.subplot(2, 2, 4)
            self.profile_plot(ax2, self.line_prof_y, self.line_prof_len)
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
        l_x0 = self.widgets.children[0].children[0].children[0]
        l_y0 = self.widgets.children[0].children[0].children[1]
        l_x1 = self.widgets.children[0].children[1].children[0]
        l_y1 = self.widgets.children[0].children[1].children[1]

        # Define the attribute to continuously update the figure, given the user interaction
        self.output = ipy.interactive(self.update_function, l_x0=l_x0, l_y0=l_y0, l_x1=l_x1, l_y1=l_y1)

        # Display the final output of the widget interaction
        display(self.output.children[-1])


# TODO: Change the STS analysis so it is consistent with topography, in regards to using the raw flat file data, not extracting it
# 3.0 - Defining the class object that will import the '.I(V)_flat' files and perform necessary spectroscopy analysis
class STS(object):
    def __init__(self, DS):
        """
        Defines the initialisation of the class object.
        DS:     The 'DataSelection' class object.
        """
        # 3.1 -  Extract all the flat-files from the data directory selected
        self.flat_files = glob.glob(DS.selected_path + '*.I(V)_flat')   # List of all the I(V) flat file paths
        self.flat_files = sorted(self.flat_files, key=len)              # Sorting the list in ascending order
        self.num_of_files = len(self.flat_files)                        # Total number of I(V) flat files loaded
        self.file_alias = None                                          # List of unique identifiers to I(V) flat files
        self.all_flatfile_extract()

        # 3.2 - Defining all the attributes associated with the I(V) file selection
        self.selected_files = None                          # List of the selected I(V) aliases
        self.num_of_selected_files = None                   # Total number of I(V) flat files selected
        self.selected_pos = None                            # List of the array positions of the I(V) files
        self.selected_data = None                           # List of the selected I(V) flat file classes
        self.selected_v_dat = None                          # List of the selected I(V) voltage data domains
        self.selected_i_dat = None                          # List of the selected I(V) current data ranges

        # 3.3 - Passing the selected files through the sts analysis functions
        # 3.3.1 Cross-correlation analysis attributes
        self.xcorr_info = None                              # Dictionary with all the cross-correlation info
        self.xcorr_v_dat = None                             # List of cross-correlated I(V) voltages
        self.xcorr_i_dat = None                             # List of cross-correlated I(V) currents
        self.v_outliers = None                              # 1D array of the outlying voltage points
        self.i_outliers = None                              # 1D array of the outlying current points
        # 3.3.2 Cropped voltage domain attributes
        self.xcrop_v_dat = None                             # Cross-correlated, cropped I(V) voltage list
        self.xcrop_i_dat = None                             # Cross-correlated, cropped  I(V) current list
        # 3.3.3 STS analysis attributes
        self.avg_i_data = None                              # Average I(V) curve over all selected files
        self.avgsq_i_data = None                            # Average of the squared I(V) curves over all selected files
        self.smooth_i_data = None                           # Smoothed I(V) curves of all selected files
        self.smooth_avg_i_data = None                       # Smoothed version of the average I(V) curve
        self.smooth_avgsq_i_data = None                     # Smoothed version of the average of the squares I(V) curve
        self.didv_data = None                               # Derivative of all smoothed I(V) curves in a 2D array form
        self.didv_avg_data = None                           # Derivative of the average I(V) curve
        self.didv_avgsq_data = None                         # Derivative of the average of the squares I(V) curve
        self.i_var = None                                   # Variance/uncertainty in the best estimation of dI/dV
        # 3.3.4 User interaction
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
        # Int Slider widget to find an estimate of the band gap
        smooth_order_select_1 = ipy.IntSlider(value=3, min=1, max=10, description="Smoothing order: ",
                                              continuous_update=False,
                                              layout=ipy.Layout(width='75%', height='auto', display='flex',
                                                                flex_flow='row', align_items='stretch'))
        # Float Range Slider widget to find an estimate of the band gap
        bandgap_select_1 = ipy.FloatRangeSlider(value=[-0.25, 0.25], min=-1.5, max=1.5, step=0.01,
                                                description="Band-gap [$V$]: ", continuous_update=False,
                                                layout=ipy.Layout(width='95%', height='auto', display='flex',
                                                                  flex_flow='row', align_items='stretch'))

        # Toggle Buttons widget to allow allow autoscaling, limits and limiting crop
        limit_type_select_2 = ipy.ToggleButtons(options=['Auto-scale axes', 'Axes limit', 'Image contrast'],
                                                value='Auto-scale axes', description="$$Axes\,Contr.$$",
                                                continuous_update=False,
                                                layout=ipy.Layout(display='inline-flex', flex_flow='column',
                                                                  align_items='center', align_content='center',
                                                                  justify_content='center', height='auto', width='50%'))
        # Float Range Slider widget to fix the voltage bias axes limits
        v_limits_select_2 = ipy.FloatRangeSlider(value=[-1.5, 1.5], min=-2.5, max=2.5, step=0.05,
                                                 description="$V_{bias}$ [$V$]:", continuous_update=False,
                                                 layout=ipy.Layout(width='97%', display='flex',
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
                                                            flex_flow='column', align_items='center')
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


# 4.0 - Defining the class object that will import the '.I(Z)_flat' files and perform all the necessary I(Z) analysis
class STZ(object):
    def __init__(self, DS):
        """
        Defines the initialisation of the class object.
        DS:     The 'DataSelection' class object.
        """
        # 4.1 -  Extract all the flat-files from the data directory selected
        self.flat_files = glob.glob(DS.selected_path + '*.I(Z)_flat')    # List of all the I(Z) flat file paths
        self.num_of_files = len(self.flat_files)                         # Total number of I(Z) flat files loaded
        self.file_alias = None                                           # List of unique identifiers to the flat files
        self.all_flatfile_extract()

        # 4.2 - Defining all the attributes associated with the I(Z) file selection


# 5.0 - Defining the class object that will import the '.I(V)_flat' files associated with CITS maps
class CITS(object):
    def __init__(self, DS):
        """
        Defines the initialisation of the class object.
        DS:     The 'DataSelection' class object.
        """
        # 5.1 -  Extract all the flat-files from the data directory selected
        self.flat_files = glob.glob(DS.selected_path + '*.I(Z)_flat')    # List of all the I(Z) flat file paths
        self.num_of_files = len(self.flat_files)                         # Total number of I(Z) flat files loaded
        self.file_alias = None                                           # List of unique identifiers to the flat files
        self.all_flatfile_extract()

        #.data[0].data gives the CITS data formation


        # 5.2 - Defining all the attributes associated with the CITS file selection
