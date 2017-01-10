from scipy.optimize import leastsq
import numpy as np


class topo_local_plane():

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