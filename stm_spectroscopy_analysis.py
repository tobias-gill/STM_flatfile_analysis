import bqplot as bq
from IPython.display import display
from ipywidgets import *
from traitlets import *

class spec_plot(object):
    def __init__(self, flat_files):

        self.flat_files = flat_files

        self.set_scales()
        self.set_axis()

        self.init_data()

        self.line = bq.Lines(x=self.v_dat, y=self.y_dat, scales={'x': self.sc_x, 'y': self.sc_y})
        self.fig = bq.Figure(axes=[self.x_ax, self.y_ax], marks=[self.line])

        self.repeats_dropdown()
        self.show_retrace()

        display(self.fig, self.rep, self.retrace)

    def set_scales(self):
        """
        Create the x and y scales for the spectroscopy plot

        Returns:
            Nothing
        """
        self.sc_x = bq.LinearScale()
        self.sc_y = bq.LinearScale()

    def set_axis(self):
        """
        Create the x and y axes

        Returns:
            Nothing
        """
        x_label = self.flat_files[0][0].info['unitv']
        y_label = self.flat_files[0][0].info['unit']

        if 'Aux' in self.flat_files[0][0].info['filename']:
            y_type = 'dI/dV (V) '
        else:
            y_type = 'I(V) '

        self.x_ax = bq.Axis(label='Voltage [' + x_label + ']',
                            scale=self.sc_x,
                            gridlines='dashed')
        self.y_ax = bq.Axis(label=y_type + '[' + y_label + ']',
                            scale=self.sc_y,
                            orientation='vertical',
                            gridlines='dashed')

    def init_data(self):
        """
        Extract voltage data and initial y data
        """
        v_start = self.flat_files[0][0].info['vstart']
        v_res = self.flat_files[0][0].info['vres']
        v_inc = self.flat_files[0][0].info['vinc']
        v_end = v_start + v_res * v_inc

        self.v_dat = np.arange(v_start, v_end, v_inc)
        self.y_dat = self.flat_files[0][0].data

    def calc_mean(self, retrace=False):

        if retrace:
            tr_dat = np.mean(np.array([self.flat_files[i][0].data for i in range(len(self.flat_files))]), axis=0)
            rtr_dat = np.mean(np.array([self.flat_files[i][1].data for i in range(len(self.flat_files))]), axis=0)
            return np.mean(np.array([tr_dat, rtr_dat]), axis=0)

        else:
            return np.mean(np.array([self.flat_files[i][0].data for i in range(len(self.flat_files))]), axis=0)

    def update_graph(self, repeat):
        """
        Updates the current flat file to visualise
        """
        if repeat == 'all':
            self.line.y = np.array([self.flat_files[i][0].data for i in range(len(self.flat_files))])
        elif repeat == 'avg':
            self.line.y = self.calc_mean()
        else:
            self.line.y = [self.flat_files[repeat][i].data for i in range(len(self.flat_files[repeat]))]

    def repeats_dropdown(self):
        """
        Creates the dropdown to select which flat file to visualise
        """
        opts = {}
        for i in np.arange(0, len(self.flat_files)):
            opts[i + 1] = i
        opts['all'] = 'all'
        opts['avg'] = 'avg'

        self.rep = interact(self.update_graph, repeat=opts)

    def show_retrace(self):
        """
        Toggle for showing retrace, if there is one
        """
        self.retrace = ToggleButton(value=False,
                                    description='Show retrace',
                                    disabled=True,
                                    button_style='',
                                    tooltop='Use retrace?')
        if len(self.flat_files[0]) > 1:
            self.retrace.value = True
            self.retrace.disabled = False