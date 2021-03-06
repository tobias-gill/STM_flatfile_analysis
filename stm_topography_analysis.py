from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from copy import deepcopy
import ipywidgets as widgets
from IPython.display import clear_output
from IPython.display import display

class local_plane():

    def __init__(self, file_data, x0, x1, y0, y1, scan_dir=0):
        """
        Plane flatten an stm image by fitting to a defined area.

        Arguments
        :param file_data: An instance of an Omicron flat file.
        :param x0: x-axis plane area initial co-ordinate.
        :param x1: x-axis plane area final co-ordinate.
        :param y0: y-axis plane area initial co-ordinate.
        :param y1: y-axis plane are final co-ordinate.

        Optional Arguments
        :param scan_dir: flat file scan direction.
        """

        x0 = self.nm2pnt(x0, file_data)
        x1 = self.nm2pnt(x1, file_data)
        y0 = self.nm2pnt(y0, file_data, axis='y')
        y1 = self.nm2pnt(y1, file_data, axis='y')

        self.topo_info = file_data[scan_dir].info
        self.x_res = self.topo_info['xres']
        self.y_res = self.topo_info['yres']

        self.x_range = np.arange(0, self.x_res, 1)
        self.y_range = np.arange(0, self.y_res, 1)

        self.topo_data = file_data[scan_dir].data

        self.param_init = [1, 1, 1]

        self.topo_plane_lsq = leastsq(self.topo_plane_residuals, self.param_init,
                                      args=(self.topo_data, x0, x1, y0, y1))[0]
        self.topo_plane_fit = self.topo_plane_paramEval(self.topo_plane_lsq)
        self.topo_data_flattened = self.topo_data - self.topo_plane_fit
        self.topo_data_flattened = self.topo_data_flattened - np.amin(self.topo_data_flattened)

        self.get_data()

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

    def get_data(self):
        """
        Get the plane flattened data.
        :return: Return plane flattened data.
        """
        return self.topo_data_flattened

    def topo_plane_residuals(self, param, topo_data, x0, x1, y0, y1):
        """
        Calculate the residuals between the real and fit generated data.

        Arguments
        :param param: List of three fit parameters for the x and y plane gradients, and z offset.
        :param topo_data: numpy array containing topography data.
        :param x0: x-axis plane area initial co-ordinate.
        :param x1: x-axis plane area final co-ordinate.
        :param y0: y-axis plane area intial co-ordinate.
        :param y1: y-axis plane area final co-ordinate.
        :return: Plane corrected data.
        """
        self.p_x = param[0]
        self.p_y = param[1]
        self.p_z = param[2]

        self.diff = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                self.diff.append(topo_data[y, x] - (self.p_x*x + self.p_y*y + self.p_z))
        return self.diff

    def topo_plane_paramEval(self, param):
        """
        Generate a plane from given parameters.
        :param param: List of x, y gradients and z offset.
        :return: Generated plane data.
        """
        # Create an empty numpy array with the same number as pixels as the real data.
        self.topo_plane_fit_data = np.zeros((self.y_res, self.x_res))
        for y in range(0, self.y_res):  # Iterate over the y-axis pixels.
            for x in range(0, self.x_res):  # Iterate over the x-axis pixels.
                self.topo_plane_fit_data[y, x] = param[0]*x + param[1]*y + param[2]  # Generate plane value.
        return self.topo_plane_fit_data  # Return entire array.


def stm_plot(flat_file, scan_dir=0, cmap=None, vmin=None, vmax=None, xy_ticks=4, z_ticks=4):
    """
    Function to plot STM topographic data.

    Arguments
    :param flat_file: An instance of an Omicron topography flat file.

    Optional Arguments
    :param scan_dir: Define which scane direction to use. fwd_up=0, bwd_up=1, fwd_dwn=2, bwd_dwn=3
    :param cmap: Matplotlib colormap name.
    :param vmin: Use to manually define the minimum value of the colour scale.
    :param vmax: Use to manually define the maximum value of the colour scale.
    :param xy_ticks: Use to manually define the number of ticks on the x and y axis.
    :param z_ticks: Use to manually define the number of ticks on the colour bar.
    """
    nm = 10 ** -9

    fig, ax = plt.subplots()

    # Set minimum value to zero and convert to nanometers.
    figure_data = (flat_file[scan_dir].data - np.amin(flat_file[scan_dir].data)) / nm

    if cmap is None:
        cmap = 'hot'

    if vmin is None:
        vmin = np.amin(figure_data)
    if vmax is None:
        vmax = 1.25 * np.amax(figure_data)

    cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    xy_units = flat_file[scan_dir].info['unitxy']  # Get x, y units from flat file.

    x_res = flat_file[scan_dir].info['xres']  # Get number of x pixels from flat file.
    y_res = flat_file[scan_dir].info['yres']  # Get number of y pixels from flat file.

    x_max = flat_file[scan_dir].info['xreal']  # Get x size in real units from flat file.
    y_max = flat_file[scan_dir].info['yreal']  # Get y size in real units from flat file.

    ax.set_xticks([x for x in np.arange(0, x_res + 1, x_res / xy_ticks)])  # Set xticks by input.
    ax.set_xticklabels(
        [str(np.round(x, 1)) for x in np.arange(0, x_max + 1, x_max / xy_ticks)])  # Set tick labels with rounding.

    ax.set_yticks([y for y in np.arange(0, y_res + 1, y_res / xy_ticks)])  # Set yticks by input.
    ax.set_yticklabels(
        [str(np.round(y, 1)) for y in np.arange(0, y_max + 1, y_max / xy_ticks)])  # Set tick labels with rounding.

    ax.set_xlabel(xy_units, size=16, weight='bold')  # Label x axis with units from flat file.
    ax.set_ylabel(xy_units, size=16, weight='bold')  # Label y axis with units from flat file.

    ax.set_title('Set-Points: {voltage} V, {current} pA'.format(voltage=flat_file[scan_dir].info['vgap'],
                                                                current=np.round(
                                                                    flat_file[scan_dir].info['current'] * 10 ** 12)))

    cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]  # Define colorbar ticks.
    cbar_ticklabels = [str(np.round(z, 1)) for z in np.arange(vmin, vmax + 1, vmax / z_ticks)]  # Label colorbar ticks.
    cbar = fig.colorbar(cax, ticks=cbar_ticks)  # Create colorbar.
    cbar.ax.set_yticklabels(cbar_ticklabels, size=16)  # Set colorbar tick labels.
    cbar.set_label('Height [nm]', size=18, weight='bold')  # Set colorbar label.

    plt.show()


def nm2pnt(nm, flat_file, axis='x'):
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


def stm_rotate(flat_file, angle):
    """
    Create a copied instance of the flat file rotated by the given angle.

    :param flat_file: An instance of an Omicron flat file.
    :param angle: Rotation angle in degrees.
    :return: New flat file instance with rotated image data.
    """
    flat_file_copy = deepcopy(flat_file)  # Create a deep copy of the flat file.

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


def stm_crop(flat_file, xmin, xmax, ymin, ymax):
    """
    Create a copy of the flat file, cropped by the defined pixel numbers.

    :param flat_file: An instance of an Omicron flat file.
    :param xmin: Crop x-axis initial co-ordinate.
    :param xmax: Crop x-axis final co-ordinate.
    :param ymin: Crop y-axis initial co-ordinate.
    :param ymax: Crop y-axis final co-ordinate.
    :return: New flat file instance with cropped image data.
    """

    xmin = nm2pnt(xmin, flat_file)
    xmax = nm2pnt(xmax, flat_file)
    ymin = nm2pnt(ymin, flat_file, axis='y')
    ymax = nm2pnt(ymax, flat_file, axis='y')

    flat_file_copy = deepcopy(flat_file)  # Create a new deep copy of the flat file.

    # For each scan direction in the flat file crop the data and amend metadata.
    for scan_dir in flat_file_copy:
        scan_dir.data = scan_dir.data[ymin:ymax, xmin:xmax]  # Crop image data.

        scan_dir.info['xres'] = xmax - xmin  # Set new x-axis pixel resolution.
        scan_dir.info['yres'] = ymax - ymin  # Set new y-axis pixel resolution.

        scan_dir.info['xreal'] = scan_dir.info['xinc'] * scan_dir.info['xres']  # Set new x-axis image size.
        scan_dir.info['yreal'] = scan_dir.info['yinc'] * scan_dir.info['yres']  # Set new y-axis image size.

    return flat_file_copy  # Return new flat file instance.


def profile(points, flat_file, num_points=100, scan_dir=0):
    """
    Extract a line profile from the given falt file and list of x, y co-ordinates.

    Arguments
    :param points: List of x, y co-ordinate pairs that define the line profile.
    :param flat_file: An instance of a Omicron flat file.

    Optional Arguments
    :param num_points: Number of points in the line profile data.
    :param scan_dir: Scan direction from which to take the line profile data.
    :return: Line profile z-data, Line profile distance-data.
    """
    length = 0
    for p in range(len(points) - 1):
        length += np.sqrt((points[p + 1, 0] - points[p, 0]) ** 2 + (points[p + 1, 1] - points[p, 0]) ** 2)

    x_len = len(flat_file[scan_dir].data[0])
    y_len = len(flat_file[scan_dir].data)

    for point in range(len(points)):
        points[point][0] = nm2pnt(points[point][0], flat_file, axis='x')
        points[point][1] = nm2pnt(points[point][1], flat_file, axis='y')
        if points[point][0] >= x_len:
            points[point][0] = x_len - 1
        if points[point][1] >= y_len:
            points[point][1] = y_len - 1

    def line(coords, flat_file):

        x0, y0 = coords[0]
        x1, y1 = coords[1]
        num = num_points
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

        zi = flat_file[scan_dir].data[y.astype(np.int), x.astype(np.int)]

        return zi

    profile_data = np.array([])
    for pair in range(len(points) - 1):
        profile_data = np.append(profile_data, line([points[pair], points[pair + 1]], flat_file))

    return profile_data, length


def profile_plot(profile_data, length, xticks=5, yticks=5):
    """
    Create a plot of the given line profile data.

    Arguments
    :param profile_data: List of line profile data.
    :param length: Nanometer length of line profile.

    Optional Arguments
    :param xticks: Number of x-axis ticks.
    :param yticks: Number of y-axis ticks.
    :return:
    """
    fig, ax = plt.subplots()  # Create a pyplot subplot figure and axis instance.

    ax.plot(profile_data)  # Plot the line profile data.

    # Set the x-axis ticks from the number defined.
    ax.set_xticks([x for x in np.arange(0, len(profile_data) + 10 * 10 ** -10, len(profile_data) / xticks)])
    # Set the x-axis tick labels from the given profile length.
    ax.set_xticklabels([str(np.round(x, 1)) for x in np.arange(0, length + 1, length / xticks)], size=12, weight='bold')
    # Set the x-axis label.
    ax.set_xlabel('[nm]', size=14, weight='bold')

    # Set the y-axis ticks from the number defined.
    ax.set_yticks([y for y in np.arange(0, 1.2 * np.max(profile_data), np.max(profile_data) / yticks)])
    # Set the y-axis tick labels from the range of the profile data.
    ax.set_yticklabels(
        [str(np.round(y, 10)) for y in np.arange(0, 1.2 * np.max(profile_data), np.max(profile_data) / yticks)],
        size=12, weight='bold')
    # Set the y-axis label.
    ax.set_ylabel('Height [m]', size=14, weight='bold')

    plt.show()  # Show the plot.


def stm_profile_plot(flat_file, points, scan_dir=0, cmap=None, vmin=None, vmax=None, xy_ticks=4, z_ticks=4):
    """
    Create a plot of the stm image, with the given line profile locations overlaid.

    Arguments
    :param flat_file: An instance of an Omicron flat file.
    :param points: List of x, y co-ordinate pairs that construct the line profile.

    Optional Arguments
    :param scan_dir: Scan direction of the flat file.
    :param cmap: Pyplot color scheme to use.
    :param vmin: Z-axis minimum value.
    :param vmax: Z-axis maximum value.
    :param xy_ticks: Number of x-, y-axis ticks.
    :param z_ticks: Number of z-axis ticks.
    :return:
    """
    nm = 10 ** -9  # Define the nanometer to meter conversion.

    fig, ax = plt.subplots()  # Create an instance of a pyplot figure and axis.

    # Set the minimum of the scan data to zero.
    figure_data = (flat_file[scan_dir].data - np.amin(flat_file[scan_dir].data)) / nm

    if cmap is None:  # If no color scheme is given use hot as default.
        cmap = 'hot'

    if vmin is None:  # If no z-axis minimum is given use minimum of the image data.
        vmin = np.amin(figure_data)
    if vmax is None:  # If no z-axis maxmimum is given use 125% of the maximum in the image data.
        vmax = 1.25 * np.amax(figure_data)

    # Add image plot to the axis and define it so that the color map can be generated.
    cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # Convert nanometer values into pixel numbers.
    for point in range(len(points)):
        points[point][0] = nm2pnt(points[point][0], flat_file, axis='x')
        points[point][1] = nm2pnt(points[point][1], flat_file, axis='y')

    # Plot the line profile points on the axis.
    ax.plot(points[:, 0], points[:, 1], 'bo-')

    xy_units = flat_file[scan_dir].info['unitxy']  # Get xy units.

    x_res = flat_file[scan_dir].info['xres']  # Get number of x-axis pixels.
    y_res = flat_file[scan_dir].info['yres']  # Get number of y-axis pixels.

    x_max = flat_file[scan_dir].info['xreal']  # Get x-axis image size.
    y_max = flat_file[scan_dir].info['yreal']  # get y-axis image size.

    # Set the x-axis ticks from number given.
    ax.set_xticks([x for x in np.arange(0, x_res + 1, x_res / xy_ticks)])
    # Set the x-axis tick labels from image size.
    ax.set_xticklabels([str(np.round(x, 1)) for x in np.arange(0, x_max + 1, x_max / xy_ticks)])

    # Set the y-axis ticks from number given
    ax.set_yticks([y for y in np.arange(0, y_res + 1, y_res / xy_ticks)])
    # Set the y-axis tick labels from image size.
    ax.set_yticklabels([str(np.round(y, 1)) for y in np.arange(0, y_max + 1, y_max / xy_ticks)])

    # Set the x- and y-axis labels.
    ax.set_xlabel(xy_units, size=16, weight='bold')
    ax.set_ylabel(xy_units, size=16, weight='bold')

    # Define the limits of the plot.
    ax.set_xlim([0, x_res])
    ax.set_ylim([0, y_res])

    # St the plot title with the image setpoint parameters.
    ax.set_title('Set-Points: {voltage} V, {current} pA'.format(voltage=flat_file[scan_dir].info['vgap'],
                                                                current=np.round(
                                                                    flat_file[scan_dir].info['current']*10**12)))

    # Define list containing the z-axis ticks from number given.
    cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]
    # Define the z-axis tick labels.
    cbar_ticklabels = [str(np.round(z, 1)) for z in np.arange(vmin, vmax + 1, vmax / z_ticks)]
    # Create color bar.
    cbar = fig.colorbar(cax, ticks=cbar_ticks)
    # Set the color bar tick labels.
    cbar.ax.set_yticklabels(cbar_ticklabels, size=16)
    # Set color bar label.
    cbar.set_label('Height [' + xy_units + ']', size=18, weight='bold')

    plt.show()


class FFTPlot(object):
    def __init__(self, flat_file):
        super().__init__()

        self.ff = flat_file

        # Calculate FFT
        self.perform_fft()

        # Construct Widgets
        self.create_sliders()
        self.create_checkboxes()

        # Create Layout
        sliders = [self.contrast_slider, self.x_crop_slider, self.y_crop_slider]
        sliders_vbox = widgets.VBox(sliders)

        checkboxes = [self.y_crop]
        checkboxes_vbox = widgets.VBox(checkboxes)

        controls = [sliders_vbox, checkboxes_vbox]
        controls_hbox = widgets.HBox(controls)

        self.plot()

        layout = [controls_hbox]
        self.layout_vbox = widgets.VBox(layout)

        self.contrast_slider.observe(self.change_contrast)
        self.x_crop_slider.observe(self.change_zoom)
        self.y_crop_slider.observe(self.change_zoom)

        self.y_crop.observe(self.toggle_y_zoom)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.marker = None

        display(self.layout_vbox)

    def get_widget(self):
        return self.layout_vbox

    def perform_fft(self, use_flat=True):
        """
        Calculates the 2D discrete Fourier Transform on the image data. By default will use flattened data if it exists, if not
        resorts to using the raw data.

        Args:
            use_flat (bool): Allows the user to overide the default behaviour of using flattened data.

        Returns:
            None
        """
        for scan_dir in range(len(self.ff)):
            if use_flat and 'data_flat' in self.ff[scan_dir].__dict__.keys():
                self.ff[scan_dir].data_fft = np.fft.fftshift(np.fft.fft2(self.ff[scan_dir].data_flat))
            else:
                self.ff[scan_dir].data_fft = np.fft.fftshift(np.fft.fft2(self.ff[scan_dir].data))

    def create_sliders(self):
        """
        Creates all of the control sliders.

        Returns:
            None
        """
        self.create_contrast_slider()
        self.create_crop_sliders()

    def create_checkboxes(self):
        """
        Creates all of the control checkboxes.

        Returns:
            None
        """
        self.create_y_crop_box()

    # Sliders
    # =======

    def create_contrast_slider(self):
        """
        Creates an IntSlider to change the contrast of the FFT plot. maximum valuesa taken from the data.

        Returns:
            None
        """
        contrast_max = 0
        for scan_dir in self.ff:
            temp_max = np.max(np.abs(scan_dir.data_fft))
            if temp_max > contrast_max:
                contrast_max = temp_max

        self.contrast_slider = widgets.IntSlider(min=0, max=100, step=1, value=50, continuous_update=False,
                                                 description='Contrast', readout=False)

    def create_crop_sliders(self):
        """
        Creates an IntSlider to symmetrically crop the FFT plot, to effectively zoom in.

        Returns:
            None
        """
        x_range = int(self.ff[0].info['xres'] / 2)
        y_range = int(self.ff[0].info['yres'] / 2)

        self.x_crop_slider = widgets.IntSlider(min=0, max=x_range, step=1, value=0, continuous_update=True,
                                               description='Zoom', readout=False)
        self.y_crop_slider = widgets.IntSlider(min=0, max=y_range, step=1, value=0, continuous_update=True,
                                               disabled=True,
                                               description='Y - Zoom', readout=False)

    # Checkboxes
    # ==========

    def create_y_crop_box(self):
        """
        Creates a CheckBox widget to toggle independent x and y zooming.

        Returns:
            None
        """
        self.y_crop = widgets.Checkbox(value=False, description='y zoom')

    # Plotting
    # ========

    def plot(self):

        x_mid = int(self.ff[0].info['xres'] / 2)
        y_mid = int(self.ff[0].info['yres'] / 2)

        x = x_mid - self.x_crop_slider.value

        if self.y_crop.value is True:
            y = y_mid - self.y_crop_slider.value
        else:
            y = y_mid - self.x_crop_slider.value

        x0 = x_mid - x
        x1 = x_mid + x
        y0 = y_mid - y
        y1 = y_mid + y

        self.x_range = [x0, x1]
        self.y_range = [y0, y1]

        plot_data = np.abs(self.ff[0].data_fft[y0:y1, x0:x1])

        self.v_range = np.max(plot_data) / 100

        v_max = self.contrast_slider.value * self.v_range

        cm2inch = 0.393701

        self.fig, self.ax = plt.subplots(figsize=(15 * cm2inch, 15 * cm2inch))
        self.fft_plot = self.ax.imshow(plot_data, origin='lower', cmap='hot', clim=(0, v_max), interpolation='none')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def change_contrast(self, b):
        """
        Connected to the contrast slider in the FFT widget. When the slider is moved the plot colour map scale will be adjusted
        to the range percentage value of the slider.

        Args:
            b: Widget

        Returns:
            None
        """
        self.fft_plot.set_clim(0, self.contrast_slider.value * self.v_range)
        clear_output()
        display(self.fig)

    def change_zoom(self, b):
        """
        Connected to the zoom slider of the FFT Widget. Sets the x and y limits of the plot to effectively zoom.

        Args:
            b: Widget

        Returns:
            None
        """

        x_mid = int(self.ff[0].info['xres'] / 2)
        y_mid = int(self.ff[0].info['yres'] / 2)

        x = x_mid - self.x_crop_slider.value

        if self.y_crop.value is True:
            y = y_mid - self.y_crop_slider.value
        else:
            y = y_mid - self.x_crop_slider.value

        x0 = x_mid - x
        x1 = x_mid + x
        y0 = y_mid - y
        y1 = y_mid + y

        self.x_range = [x0, x1]
        self.y_range = [y0, y1]

        self.ax.set_xlim([x0, x1])
        self.ax.set_ylim([y0, y1])

    def toggle_y_zoom(self, b):
        """
        Connected to the y zoom checkbox. Toggles whether to use seperate x and y zoom scales.

        Args:
            b: Widget

        Returns:
            None
        """
        if self.y_crop_slider.disabled is True:
            self.y_crop_slider.disabled = False
        elif self.y_crop_slider.disabled is False:
            self.y_crop_slider.disabled = True

    # Reciprocal measurements

    def k_dist(self, px, py):
        """
        Calculates the reciprocal, and real space distances from a pair of x and y coordinates passed to the function. The

        Args:
            px (int): x direction pixel number of FFT image.
            py (int): y direction pixel number of FFT image.

        Returns:
            k (float): Reciprocal distance of the given pixel.
            r (float): Real space distance corresponding to the given pixel.
        """

        mid_x, mid_y = int(np.round(self.ff[0].info['xres'] / 2)), int(np.round(self.ff[0].info['yres'] / 2))

        x_real, y_real = self.ff[0].info['xreal'], self.ff[0].info['yreal']
        kx_real, ky_real = 2 * np.pi / x_real, 2 * np.pi / y_real

        x = np.abs(px - mid_x)
        y = np.abs(py - mid_y)

        theta = np.degrees(np.arctan(y/x))

        kx = kx_real - np.sum([kx_real for i in range(x + 1)])
        ky = ky_real - np.sum([ky_real for i in range(y + 1)])

        k = np.sqrt(kx ** 2 + ky ** 2)
        r = 2 * np.pi / k

        k = np.round(k, 2)
        r = np.round(r, 4)
        kx = np.round(kx, 2)
        ky = np.round(ky, 2)
        theta = np.round(theta, 2)

        return (k, (kx, ky), theta), r

    # Mouse Events

    def onclick(self, event):
        """
        Called my a mouse click event within the plot axes. Will plot a marker pair that identifies the distance being calculated
        by the k_dist() function. Also adds the recirpocal and real space distances as a title to the plot.

        Args:
            event (mpl event): Mouse click event within the plot axes.

        Returns:
            None
        """

        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        x_mid = int(np.round(self.ff[0].info['xres'] / 2))
        y_mid = int(np.round(self.ff[0].info['yres'] / 2))

        (k, (kx, ky), theta), r = self.k_dist(x, y)

        if self.marker is None and event.button == 1:
            self.marker, = self.ax.plot([x_mid, x], [y_mid, y], 'o-c', markersize=10, alpha=0.5)
            self.ax.set_xlim(self.x_range)
            self.ax.set_ylim(self.y_range)
            self.ax.set_title("|k| = {k} {unit}^-1, r = {r} {unit}".format(k=k, unit=self.ff[0].info['unitxy'], r=r))
            self.ax.set_xlabel("kx = {kx}, ky = {ky}, angle = {theta}".format(kx=kx, ky=ky, theta=theta))
        elif event.button == 1:
            self.marker.set_xdata([x_mid, x])
            self.marker.set_ydata([y_mid, y])
            self.ax.set_xlim(self.x_range)
            self.ax.set_ylim(self.y_range)
            self.ax.set_title("|k| = {k} {unit}^-1, r = {r} {unit}".format(k=k, unit=self.ff[0].info['unitxy'], r=r))
            self.ax.set_xlabel("kx = {kx}, ky = {ky}, angle = {theta}".format(kx=kx, ky=ky, theta=theta))
        elif event.button == 3:
            self.marker.set_xdata([])
            self.marker.set_ydata([])
            self.ax.set_xlim(self.x_range)
            self.ax.set_ylim(self.y_range)
            self.ax.set_title('')

