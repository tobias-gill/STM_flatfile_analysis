from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import rotate
from copy import deepcopy

class local_plane():

    def __init__(self, file_data, x0, x1, y0, y1, scan_dir=0):

        self.topo_info = file_data[scan_dir].info
        self.x_res = self.topo_info['xres']
        self.y_res = self.topo_info['yres']

        self.x_range = np.arange(0, self.x_res, 1)
        self.y_range = np.arange(0, self.y_res, 1)

        self.topo_data = file_data[scan_dir].data

        self.param_init = [1, 1, 1]

        self.topo_plane_lsq = leastsq(self.topo_plane_residuals, self.param_init, args=(self.topo_data, x0, x1, y0, y1))[0]
        self.topo_plane_fit = self.topo_plane_paramEval(self.topo_plane_lsq)
        self.topo_data_flattened = self.topo_data - self.topo_plane_fit
        self.topo_data_flattened = self.topo_data_flattened - np.amin(self.topo_data_flattened)

        self.get_data()

    def get_data(self):
        return self.topo_data_flattened

    def topo_plane_residuals(self, param, topo_data, x0, x1, y0, y1):
        self.p_x = param[0]
        self.p_y = param[1]
        self.p_z = param[2]

        self.diff = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                self.diff.append(topo_data[y, x] - (self.p_x*x + self.p_y*y + self.p_z))
        return self.diff

    def topo_plane_paramEval(self, param):
        self.topo_plane_fit_data = np.zeros((self.y_res, self.x_res))
        for y in range(0, self.y_res):
            for x in range(0, self.x_res):
                self.topo_plane_fit_data[y, x] = param[0]*x + param[1]*y + param[2]
        return self.topo_plane_fit_data


def stm_plot(flat_file, scan_dir=0, cmap=None, vmin=None, vmax=None, xy_ticks=4, z_ticks=4):
    """
    Function to plot STM topographic data.

    :param flat_file: An instance of an Omicron topography flat file.
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

    cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]  # Define colorbar ticks.
    cbar_ticklabels = [str(np.round(z, 1)) for z in np.arange(vmin, vmax + 1, vmax / z_ticks)]  # Label colorbar ticks.
    cbar = fig.colorbar(cax, ticks=cbar_ticks)  # Create colorbar.
    cbar.ax.set_yticklabels(cbar_ticklabels, size=16)  # Set colorbar tick labels.
    cbar.set_label('Height [nm]', size=18, weight='bold')  # Set colorbar label.

    plt.show()


def nm2pnt(nm, flat_file, axis='x'):
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
    flat_file_copy = deepcopy(flat_file)
    for scan_dir in flat_file_copy:
        scan_dir.data = rotate(scan_dir.data, angle)

    new_res = np.shape(flat_file_copy[0].data)

    for scan_dir in flat_file_copy:
        scan_dir.info['xres'] = new_res[1]
        scan_dir.info['yres'] = new_res[0]

        scan_dir.info['xreal'] = scan_dir.info['xinc'] * new_res[1]
        scan_dir.info['yreal'] = scan_dir.info['yinc'] * new_res[0]

    return flat_file_copy


def stm_crop(flat_file, xmin, xmax, ymin, ymax):
    flat_file_copy = deepcopy(flat_file)

    x_res = flat_file_copy[0].info['xres']
    y_res = flat_file_copy[0].info['yres']

    for scan_dir in flat_file_copy:
        scan_dir.data = scan_dir.data[ymin:ymax, xmin:xmax]

        scan_dir.info['xres'] = xmax - xmin
        scan_dir.info['yres'] = ymax - ymin

        scan_dir.info['xreal'] = scan_dir.info['xinc'] * scan_dir.info['xres']
        scan_dir.info['yreal'] = scan_dir.info['yinc'] * scan_dir.info['yres']

    return flat_file_copy


def profile(points, flat_file, num_points=100, scan_dir=0):
    length = 0
    for p in range(len(points) - 1):
        length += np.sqrt((points[p + 1, 0] - points[p, 0]) ** 2 + (points[p + 1, 1] - points[p, 0]) ** 2)

    for point in range(len(points)):
        points[point][0] = nm2pnt(points[point][0], flat_file, axis='x')
        points[point][1] = nm2pnt(points[point][1], flat_file, axis='y')

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


def profile_plot(profile_data, length, flat_file, xticks=5, yticks=5):
    fig, ax = plt.subplots()

    ax.plot(profile_data)

    ax.set_xticks([x for x in np.arange(0, len(profile_data) + 10 * 10 ** -10, len(profile_data) / xticks)])
    ax.set_xticklabels([str(np.round(x, 1)) for x in np.arange(0, length + 1, length / xticks)], size=12, weight='bold')
    ax.set_xlabel('[nm]', size=14, weight='bold')

    ax.set_yticks([y for y in np.arange(0, 1.2 * np.max(profile_data), 1.2 * np.max(profile_data) / yticks)])
    ax.set_yticklabels(
        [str(np.round(y, 10)) for y in np.arange(0, 1.2 * np.max(profile_data), 1.2 * np.max(profile_data) / yticks)],
        size=12, weight='bold')
    ax.set_ylabel('Height [m]', size=14, weight='bold')

    plt.show()


def stm_profile_plot(flat_file, points, scan_dir=0, cmap=None, vmin=None, vmax=None, xy_ticks=4, z_ticks=4):
    nm = 10 ** -9

    fig, ax = plt.subplots()

    figure_data = (flat_file[scan_dir].data - np.amin(flat_file[scan_dir].data)) / nm

    if cmap is None:
        cmap = 'hot'

    if vmin is None:
        vmin = np.amin(figure_data)
    if vmax is None:
        vmax = 1.25 * np.amax(figure_data)

    cax = ax.imshow(figure_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    for point in range(len(points)):
        points[point][0] = nm2pnt(points[point][0], flat_file, axis='x')
        points[point][1] = nm2pnt(points[point][1], flat_file, axis='y')

    ax.plot(points[:, 0], points[:, 1], 'bo-')

    xy_units = flat_file[scan_dir].info['unitxy']

    x_res = flat_file[scan_dir].info['xres']
    y_res = flat_file[scan_dir].info['yres']

    x_max = flat_file[scan_dir].info['xreal']
    y_max = flat_file[scan_dir].info['yreal']

    ax.set_xticks([x for x in np.arange(0, x_res + 1, x_res / xy_ticks)])
    ax.set_xticklabels([str(np.round(x, 1)) for x in np.arange(0, x_max + 1, x_max / xy_ticks)])

    ax.set_yticks([y for y in np.arange(0, y_res + 1, y_res / xy_ticks)])
    ax.set_yticklabels([str(np.round(y, 1)) for y in np.arange(0, y_max + 1, y_max / xy_ticks)])

    ax.set_xlabel(xy_units, size=16, weight='bold')
    ax.set_ylabel(xy_units, size=16, weight='bold')

    ax.set_xlim([0, x_res])
    ax.set_ylim([0, y_res])

    cbar_ticks = [z for z in np.arange(vmin, vmax * 1.01, vmax / z_ticks)]
    cbar_ticklabels = [str(np.round(z, 1)) for z in np.arange(vmin, vmax + 1, vmax / z_ticks)]
    cbar = fig.colorbar(cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticklabels, size=16)
    cbar.set_label('Height [' + xy_units + ']', size=18, weight='bold')

    plt.show()
