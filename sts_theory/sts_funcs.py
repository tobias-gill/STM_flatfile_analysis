import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sint
import scipy.interpolate as spol
import scipy.optimize as sopt
import ipywidgets as ipy
from IPython.display import display

__version__ = "1.0"
__date__ = "2017-03-29"
__status__ = "Complete"

__author__ = "Procopi Constantinou"
__email__ = "procopios.constantinou.16@ucl.ac.uk"

constants = {"c": 2.99792458e8, "e": 1.6021773e-19, "me": 9.109389e-31, "kB": 1.380658e-23,
             "h": 6.6260755e-34, "hbar": 1.05457e-34, "eps0": 8.85419e-12}


def gauss(x, mu=0, sigma=0.2):
    """ 
    Define the gaussian function to be used for the theoretical density of states.
    Mandatory arguments:\n
    :param x: domain of the Gaussian distribution.
    Optional arguments:\n
    :param mu: peak position of the Gaussian.
    :param sigma: peak width of the Gaussian.
    Output:\n
    :return: The Gaussian distribution.
    """
    y = np.exp(-(x - mu) ** 2 / sigma)
    return y


def fermi(x, flevel=0, temp=300):
    """
    Define the fermi-dirac distribution to be used when defining the density of states.
    Mandatory arguments:\n
    :param x: [eV]: domain of the Fermi-Dirac distribution.
    Optional arguments:\n
    :param flevel: [eV]: position of the Fermi-level as a float {for single biases} or np.ndarray {for multiple biases}.
    :param temp: [K]: temperature of the system.
    Output:\n
    :return The Fermi-Dirac distribution.
    """
    if np.size(flevel) == 1:
        y = 1 / (1 + np.exp(((constants["e"] * (x - flevel)) / (constants["kB"] * temp))))
    else:
        y = np.zeros((len(flevel), len(x)))
        for i in np.arange(0, len(flevel)):
            y[i] = 1 / (1 + np.exp(((constants["e"] * (x - flevel[i])) / (constants["kB"] * temp))))
    return y


def iv(system, tip, sample, tunneling_matrix):
    """
    Defing a function that determines the I(V) curves, given the tip, sample and tunneling matrix element.
    Mandatory arguments:\n
    :param system: 
    :param tip: class object containing all the tip information (ensure tip.DoS is defined).
    :param sample: class object containing all the sample information (ensure sample.DoS is defined).
    :param tunneling_matrix: class object of the matrix elements (ensure tunneling_matrix.tunn_element is defined).
    Output:\n
    :return The tunneling current as a function of the voltage bias (I(V) curve).
    """
    current = np.zeros(len(system.bias))
    for i in np.arange(0, len(system.bias)):
        current[i] = sint.simps(tip.DoSbiased[i] * sample.DoS * tunneling_matrix.tunn_element[i])
    return current


def theory_plot(x: np.ndarray, y: np.ndarray, info: dict = None):
    """    
    Defining a function to plot y(x) as a matplotlib figure.
    Mandatory arguments:\n
    :param x: np.ndarray of the independent variable x.
    :param y: np.ndarray of the dependent variable y.
    Optional arguments:\n
    :param info: dictionary with one or more the following keywords {"title":, "xlabel":, "ylabel":, "xlim":, "ylim":}
    Output:\n
    :return: A plot of the data passed through the function.
    """
    # Defining a figure and axes object
    fig, ax = plt.subplots()
    # Plotting the figure
    ax.plot(x, y, 'b-', linewidth=3.0)
    ax.fill_between(x, y, color='blue', alpha=0.1)
    # Formatting the figure
    ax.axvline(x=0, color="k", linestyle="-", linewidth=1)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=1)
    ax.grid(True)
    if info is None:
        plt.xlim(np.array([np.min(x), np.max(x)]))
        plt.ylim(np.array([0.95 * np.min(y), 1.05 * np.max(y)]))
        plt.show()
        return None
    else:
        if info.get("title") is not None:
            plt.title(info["title"], fontsize=18)
        if info.get("xlabel") is not None:
            plt.xlabel(info["xlabel"], fontsize=14)
        if info.get("ylabel") is not None:
            plt.ylabel(info["ylabel"], fontsize=14)
        if info.get("xlim") is not None:
            plt.xlim(info["xlim"])
        else:
            plt.xlim(np.array([np.min(x), np.max(x)]))
        if info.get("ylim") is not None:
            plt.ylim(info["ylim"])
        else:
            plt.ylim(np.array([0.9 * np.min(y), 1.1 * np.max(y)]))
        plt.show()
        return None


class System(object):
    """
    Define a system class that yields all the akin information about the tip-sample system.\n
    Mandatory attributes:\n
    :param .en: [eV] np.ndarray that yields the domain of the electron energy.
    :param .bias: [V] np.ndarray that yields the bias range between the tip-sample.
    :param .z0: [nm] float that yields the tip-sample distance.
    :param .temp: [K] float that yields the temperature of the tip-sample system.
    Optional attributes:\n
    :param .de: [eV] float that yields the thermal broadening due to the finite temperature.
    :param .di: [pA] float that yields the uncertainty associated with the tunneling current.
    :param .iz_params: dictionary of the best fit parameters to I(z) data {"I0":, "dI0":, "kappa":, "dkappa":}.
    :param .wfunc_param: [eV] float that yields the experimental value of the work function given I(z) data.
    """

    def __init__(self, en, bias, z0=0.8, temp=77):
        self.ground = 0
        self.bias = bias
        self.en = en
        self.temp = temp
        self.z0 = z0
        self.de = (2 * constants["kB"] * self.temp) / constants["e"]
        self.di = 1
        self.iz_params = None
        self.wfunc_param = None

    def update_de(self):
        """
        Updates the thermal broadening if the temperature of the system is changed.
        """
        self.de = (2 * constants["kB"] * self.temp) / constants["e"]

    def iz(self, i_tunneling, z_distance):
        """
        Extracting the information about the decay constant and work function, given an I(z) data-set.\n
        Mandatory arguments:\n
        :param i_tunneling: [A] np.ndarray of the data obtained for the tunneling current vs tip-sample distance.
        :param z_distance: [m] np.ndarray of the data obtained for the tip-sample distance.
        Output:\n
        :param .iz_params: dictionary of the best fit parameters to I(z) data {"I0":, "dI0":, "kappa":, "dkappa":}.
        :param .wfunc_param: [eV] float that yields the experimental value of the work function given I(z) data.
        """

        # Calculating the best fit parameters to the data
        def f_iz(x, a, b):
            return a * np.exp(-2 * b * x)

        popt, pcov = sopt.curve_fit(f_iz, i_tunneling, z_distance)
        # Assigning attributes to the best fit parameters obtained
        self.iz_params = {"I0": popt[0], "dI0": pcov[0][0], "kappa": popt[1], "dkappa": pcov[1][1]}
        self.wfunc_param = (popt[1] * constants["hbar"]) ** 2 / (2 * constants["me"])


class Tip(object):
    """
    Define a tip class that yields all the necessary information about the tip being used.\n
    Mandatory attributes:\n
    :param .material: str that describes the material used in the tip.
    :param .wfunc: [eV] float that yields the work-function of the material.
    :param .eoffset: [eV] float that yields the zero energy-offset from the centre of the band-gap.
    Optional attributes:\n
    :param .DoS: [eV^-1] np.ndarray that yields the density of states as a function of the electron energy.
    :param .RoC: [m^-1] float that yields the radius of curvature of the tip, given field emission data.
    """

    def __init__(self, material, wfunc):
        self.material = material
        self.wfunc = wfunc
        self.DoSground = None
        self.DoSbiased = None
        self.RoC = None

    def dos_linear(self, system, gradient=0, intercept=1):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param gradient: float that yields the gradient of the straight line.
        :param intercept: float that yields the y-intercept of the straight line.
        Output:\n
        :return .DoS: Metallic; linear density of states.
        """
        # - Finding the unbiased, grounded density of states
        dos = abs(gradient * system.en + intercept)
        self.DoSground = dos / sint.simps(dos)
        # - Finding the biased density of states
        DOSbiased = np.zeros((len(system.bias), len(system.en)))
        for i in np.arange(0, len(system.bias)):
            dos = abs(gradient * (system.en - system.bias[i]) + intercept)
            DOSbiased[i] = dos / sint.simps(dos)
        self.DoSbiased = DOSbiased

    def dos_gauss(self, system, mu=0, sigma=10):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param mu: float that yields the peak position of the Gaussian.
        :param sigma: float that yields the width of the Gaussian peak.
        Output:\n
        :return .DoS: Metallic; gaussian density of states.
        """
        # - Finding the unbiased, grounded density of states
        dos = gauss(system.en, mu, sigma)
        self.DoSground = dos / sint.simps(dos)
        # - Finding the biased density of states
        DOSbiased = np.zeros((len(system.bias), len(system.en)))
        for i in np.arange(0, len(system.bias)):
            dos = gauss(system.en + system.bias[i], mu, sigma)
            DOSbiased[i] = dos / sint.simps(dos)
        self.DoSbiased = DOSbiased

    def dos_linewgauss(self, system, gradient=0, intercept=1, mu=0, sigma=0.2):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param gradient: float that yields the gradient of the straight line.
        :param intercept: float that yields the y-intercept of the straight line.
        :param mu: float that yields the peak position of the Gaussian.
        :param sigma: float that yields the width of the Gaussian peak.
        Output:\n
        :return .DoS: Metallic; linear and gaussian convoluted density of states.
        """
        # - Finding the unbiased, grounded density of states
        dos_linear = abs(gradient * system.en + intercept)
        dos_gauss = gauss(system.en, mu, sigma)
        dos = dos_gauss + dos_linear
        self.DoSground = dos / sint.simps(dos)
        # - Finding the biased density of states
        DOSbiased = np.zeros((len(system.bias), len(system.en)))
        for i in np.arange(0, len(system.bias)):
            dos_linear = abs(gradient * (system.en - system.bias[i]) + intercept)
            dos_gauss = gauss(system.en - system.bias[i], mu, sigma)
            dos = dos_gauss + dos_linear
            DOSbiased[i] = dos / sint.simps(dos)
        self.DoSbiased = DOSbiased

    def dos_data(self, system, x, y):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.    
        :param x: np.ndarray of the domain of the density of states data.
        :param y: np.ndarray of the density of states data.
        Output:\n
        :return .DoS: density of states based on a data-set.
        """
        f = spol.interp1d(x, y, kind="cubic")
        self.DoS = f(system.en) / sint.simps(f(system.en))

    def roc_data(self, v_bias, i_emission):
        """
        Mandatory arguments:\n
        :param v_bias: [V] np.ndarray of the voltage domain used during field emission.
        :param i_emission: [A] np.ndarray of the field emission current.
        Output:\n
        :return .RoC [m^-1]: Radius of curvature of the tip.
        """
        # Defining the constants to be used
        k = 5  # geometrical factor
        xi = 0.4  # correction factor
        # Fitting a linear equation to the given variables
        y = np.log(i_emission / v_bias ** 2)
        x = 1 / v_bias
        [_, grad] = np.polyfit(x, y, 1)
        self.RoC = grad / (-6.8e9 * xi * k * self.wfunc ** (3 / 2.))


class Sample(object):
    """
    Define a sample class that yields all the necessary information about the sample being used.\n
    Mandatory attributes:\n
    :param .material: str that describes the chemical-composition of the sample.
    :param .wfunc: [eV]: float that yields the work-function of the sample.
    :param .egap: [eV]: float that yields the band-gap of the sample.
    :param .eoffset [eV]: float that yields the zero energy-offset from the centre of the band-gap.
    Optional attributes:\n
    :param .corrugation_hor [nm]: float that yields the corrugation in the horizontal direction of the sample surface.
    :param .corrugation_ver [nm]: float that yields the corrugation in the vertical direction of the sample surface.
    :param .DoS [eV^-1]: np.ndarray that yields the density of states as a function of the electron energy.
    """

    def __init__(self, material, wfunc, egap=1.1, eoffset=0):
        """    
        Initialisation attributes:\n
        :param .material: str that describes the chemical-composition of the sample.
        :param .wfunc [eV]: float that yields the work-function of the sample.
        :param .egap [eV]: float that yields the band-gap of the sample.
        :param .eoffset [eV]: float that yields the zero energy-offset from the centre of the band-gap.
        """
        self.material = material
        self.wfunc = wfunc
        self.egap = egap
        self.eoffset = eoffset
        self.corrugation_hor = None
        self.corrugation_ver = None
        self.DoS = None

    def dos_linear(self, system, gradient=0, intercept=1):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param gradient: float that yields the gradient of the straight line.
        :param intercept: float that yields the y-intercept of the straight line.
        Output:\n
        :return .DoS: Metallic; linear density of states.
        """
        dos = abs(gradient * system.en + intercept)
        self.DoS = dos / sint.simps(dos)

    def dos_gauss(self, system, mu=0., sigma=10):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param mu: float that yields the peak position of the Gaussian.
        :param sigma: float that yields the width of the Gaussian peak.
        Output:\n
        :return .DoS: Metallic; gaussian density of states.
        """
        dos = gauss(system.en, mu, sigma)
        self.DoS = dos / sint.simps(dos)

    def dos_step(self, system):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Output:\n
        :return .DoS: Semiconductor; an inverted top-hat function with a band-gap at its centre
        """
        egap_lhs = +1 * (self.egap / 2) - self.eoffset
        egap_rhs = -1 * (self.egap / 2) - self.eoffset
        dos = (-1 * np.sign(system.en + egap_lhs) + 1) + (np.sign(system.en + egap_rhs) + 1)
        self.DoS = dos / sint.simps(dos)

    def dos_para(self, system, grad_lhs=1, grad_rhs=1):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param grad_lhs: float that yields the curvature of the LHS parabolic density of states.
        :param grad_rhs: float that yields the curvature of the RHS parabolic density of states.
        Output:\n
        :return .DoS: Semiconductor; density of states with a band-gap and parabolic dispersion.
        """
        # Splitting the energy domain into three regions, with respect to the band-gap
        # - Finding the band-gap edge on the left and right hand side
        egap_lhs = (-1 * self.egap / 2) + self.eoffset
        egap_rhs = (+1 * self.egap / 2) + self.eoffset
        arg_lhs = np.argmax(system.en[system.en <= egap_lhs])
        arg_rhs = np.argmax(system.en[system.en <= egap_rhs])
        # - Extracting the domains over each region
        x_lhs = system.en[0:arg_lhs + 1]
        x_mid = system.en[arg_lhs + 1:arg_rhs + 1]
        x_rhs = system.en[arg_rhs + 1:]
        # Finding the density of states in each region
        dos_lhs = grad_lhs * np.sqrt(np.abs(x_lhs + (self.egap / 2) - self.eoffset))
        dos_mid = np.zeros(len(x_mid))
        dos_rhs = grad_rhs * np.sqrt(np.abs(x_rhs - (self.egap / 2) - self.eoffset))
        # Appending all the density of states into a single array
        dos = np.append(np.append(dos_lhs, dos_mid), dos_rhs)
        self.DoS = dos / sint.simps(dos)

    def dos_parass(self, system, grad_lhs=1, grad_rhs=1, ss_params=None):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Optional arguments:\n
        :param grad_lhs: float that yields the curvature of the LHS parabolic density of states.
        :param grad_rhs: float that yields the curvature of the RHS parabolic density of states.
        :param ss_params: np.array([[mu1, sigma1],[]...]) that yields the position and spread of the surface states.
        Output:\n
        :return .DoS: Semiconductor; density of states with a band-gap, parabolic dispersion and surface states.
        """
        # Splitting the energy domain into three regions, with respect to the band-gap
        # - Finding the band-gap edge on the left and right hand side
        egap_lhs = (-1 * self.egap / 2) + self.eoffset
        egap_rhs = (+1 * self.egap / 2) + self.eoffset
        arg_lhs = np.argmax(system.en[system.en <= egap_lhs])
        arg_rhs = np.argmax(system.en[system.en <= egap_rhs])
        # - Extracting the domains over each region
        x_lhs = system.en[0:arg_lhs + 1]
        x_mid = system.en[arg_lhs + 1:arg_rhs + 1]
        x_rhs = system.en[arg_rhs + 1:]
        # Finding the density of states in each region
        dos_lhs = grad_lhs * np.sqrt(np.abs(x_lhs + (self.egap / 2) - self.eoffset))
        dos_mid = np.zeros(len(x_mid))
        dos_rhs = grad_rhs * np.sqrt(np.abs(x_rhs - (self.egap / 2) - self.eoffset))
        # Appending all the density of states into a single array
        dos_int = np.append(np.append(dos_lhs, dos_mid), dos_rhs)
        # Finding the density of states for each surface-state
        if ss_params is None:
            dos_ss1 = gauss(system.en, 0.60 + self.eoffset, 0.01)
            dos_ss2 = gauss(system.en, -0.60 + self.eoffset, 0.01)
            dos_ss = dos_ss1 + dos_ss2
        else:
            dos_allss = np.zeros((len(ss_params), len(system.en)))
            for i in range(0, len(ss_params)):
                dos_allss[i] = gauss(system.en, ss_params[i][0], ss_params[i][1])
            dos_ss = np.sum(dos_allss, axis=0)
        # Linear superposition of all density of states elements
        dos = dos_int + dos_ss
        self.DoS = dos / sint.simps(dos)

    def dos_data(self, system, x, y):
        """    
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Mandatory arguments:\n
        :param x: np.ndarray of the domain of the density of states data.
        :param y: np.ndarray of the density of states data.
        Output:\n
        :return .DoS: density of states based on a data-set.
        """
        f = spol.interp1d(x, y, kind="cubic")
        self.DoS = f(system.en) / sint.simps(f(system.en))


class TunnMatrix(object):
    """
    Define a tunneling matrix element class that yields the necessary information of the matrix element.\n
    Optional attributes:\n
    :param .fermi_ground: np.ndarray that yields the Fermi-Dirac distribution for the ground voltage.
    :param .fermi_bias: np.ndarray that yields the Fermi-Dirac distribution for the given voltage bias.
    :param .tunn_element: np.ndarray that yields the tunneling matrix element as a function of the electron energy.
    """

    def __init__(self, system):
        self.fermi_ground = fermi(system.en, system.ground, system.temp)
        self.fermi_bias = fermi(system.en, system.bias, system.temp)
        self.tunn_element = None

    def tunn_const(self, system):
        """
        Mandatory arguments:\n
        :param system: class object that contains all the akin information about the tip-sample system.
        Output:\n
        :return .tunn_element: constant potential well modulated by the Fermi-Dirac distributions over all biases.
        """
        # Determination of the potential well as a constant
        well = np.ones(len(system.en))
        # Modulating the potential well with the Fermi-Dirac distribtuion
        tunn_matrix = list()
        for i in np.arange(0, len(system.bias)):
            tme = (self.fermi_bias[i] - self.fermi_ground) * well
            tunn_matrix.append(tme)
        self.tunn_element = tunn_matrix

    def tunn_wkb(self, tip, sample, system):
        """
        Mandatory arguments:\n
        :param tip: class object that contains all the tip information.
        :param sample: class object that contains all the sample information.
        :param system: class object that contains all the akin information about the tip-sample system.
        Output:\n
        :return .tunn_element: WKB approximated potential well modulated by Fermi-Dirac distributions over all biases.
        """
        # Finding the tunneling matrix element over all the biases defined
        tunn_matrix = list()
        for i in np.arange(0, len(system.bias)):
            # Determination of the potential well using the WKB approximation
            coeff = (-1 * (2 * np.sqrt(2 * constants["me"])) / (constants["hbar"])) * system.z0 * 1e-9
            arg1 = (0.5 * constants["e"] * (tip.wfunc + sample.wfunc + system.bias[i]))
            arg2 = -1 * (constants["e"] * system.en)
            arg = np.sqrt(arg1 + arg2)
            well = np.exp(coeff * arg)
            # Modulating the potential well with the Fermi-Dirac distribution
            tme = (self.fermi_bias[i] - self.fermi_ground) * well
            tunn_matrix.append(tme)
        self.tunn_element = tunn_matrix


def float_value_widget(label, mini, maxi, step, default=None):
    if default is None:
        slider = ipy.FloatSlider(value=0.25*maxi, min=mini, max=maxi, step=step, description=label,
                                 continuous_update=True)
        display(slider)
    else:
        slider = ipy.FloatSlider(value=default, min=mini, max=maxi, step=step, description=label,
                                 continuous_update=True)
        display(slider)
    return slider


def float_range_widget(label, mini, maxi, step):
    slider = ipy.FloatRangeSlider(value=[0.75 * mini, 0.75 * maxi], min=mini, max=maxi, step=step, description=label,
                                  continuous_update=True)
    display(slider)
    return slider


def buttons_widget(label, options, default=None):
    if default is None:
        button = ipy.ToggleButtons(options=options, value=options[0], description=label, continuous_update=True)
        display(button)
    else:
        button = ipy.ToggleButtons(options=options, value=default, description=label, continuous_update=True)
        display(button)
    return button


def textbox_widget():
    textbox = ipy.Text(value="Si", description="$Sample$", continuous_update=True)
    display(textbox)
    return textbox


def tip_dos_plot(system, tip_material, tip_dos):
    # Defining the tip object from the button selection [tip_material]
    if tip_material.value == "W-tip":
        tip = Tip("W-tip", 4.5)
    if tip_material.value == "PtIr-tip":
        tip = Tip("PtIr-tip", 5.4)

    # Extracting the tip density of states based on the button selection [tip_dos]
    # - For a metallic tip with a linear density of states
    if tip_dos.value == "Metal; Linear":
        def f(gradient):
            tip.dos_linear(system, gradient)
            plot_info = {"title": tip.material+" DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                 "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.005]}
            theory_plot(system.en, tip.DoSground, plot_info)
            return
        ipy.interact(f, gradient=ipy.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0, description='$Gradient:$'),
                     continuous_update=True)
        return tip
    # - For a metallic tip with a gaussian distribution form
    if tip_dos.value == "Metal; Gaussian":
        def f(peak_pos, peak_width):
            tip.dos_gauss(system, peak_pos, peak_width)
            plot_info = {"title": tip.material+" DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                         "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.02]}
            theory_plot(system.en, tip.DoSground, plot_info)
            return
        ipy.interact(f, peak_pos=ipy.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0.0,
                                                 description='$Peak$ $position:$'),
                     peak_width=ipy.FloatSlider(min=0.10, max=2.00, step=0.1, value=0.5,
                                                description='$Peak$ $width:$'),
                     continuous_update=True)
        return tip
    # - For a metallic tip with a superposition of linear and gaussian form
    if tip_dos.value == "Metal; Linear + Gaussian":
        def f(gradient, intercept, peak_pos, peak_width):
            tip.dos_linewgauss(system, gradient, intercept, peak_pos, peak_width)
            plot_info = {"title": tip.material+" DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                         "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.02]}
            theory_plot(system.en, tip.DoSground, plot_info)
            return
        ipy.interact(f, gradient = ipy.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0,
                                                   description='$Gradient:$'),
                     intercept = ipy.FloatSlider(min=0, max=5, step=0.1, value=0.2,
                                                 description='$Intercept:$'),
                     peak_pos = ipy.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0.0,
                                                description='$Peak$ $position:$'),
                     peak_width = ipy.FloatSlider(min=0.10, max=2.00, step=0.1, value=0.2,
                                                  description='$Peak$ $width:$'),
                     continuous_update=True)
        return tip


def sample_dos_plot(system, sample_name, sample_dos, sample_Phi, sample_egap, sample_eoffset):
    # Defining the sample object from the selections made
    sample = Sample(sample_name.value, sample_Phi.value, sample_egap.value, sample_eoffset.value)

    # Extracting the sample density of states based on the button selection [sample_dos]
    # - For a metallic sample with a linear density of states
    if sample_dos.value == "Metal; Linear":
        def f(gradient):
            sample.dos_linear(system, gradient)
            plot_info = {"title": sample_name.value + " DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                         "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.005]}
            theory_plot(system.en, sample.DoS, plot_info)
            return
        ipy.interact(f, gradient=ipy.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0,
                                                 description='$Gradient:$'), continuous_update=True)
        return sample

    # - For a metallic sample with a gaussian distribution form
    if sample_dos.value == "Metal; Gaussian":
        def f(peak_pos, peak_width):
            sample.dos_gauss(system, peak_pos, peak_width)
            plot_info = {"title": sample_name.value + " DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                         "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.020]}
            theory_plot(system.en, sample.DoS, plot_info)
            return
        ipy.interact(f,
                     peak_pos=ipy.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0.0,
                                              description='$Peak$ $position:$'),
                     peak_width=ipy.FloatSlider(min=0.10, max=2.00, step=0.1, value=0.5,
                                                description='$Peak$ $width:$'),
                     continuous_update=True)
        return sample

    # - For a semiconductor sample with a band-gap and step-like dispersion
    if sample_dos.value == "Semi; Gap + Step":
        sample.dos_step(system)
        plot_info = {"title": sample_name.value + " DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                     "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.005]}
        theory_plot(system.en, sample.DoS, plot_info)
        return sample

    # - For a semiconductor sample with a band-gap and parabolic dispersion
    if sample_dos.value == "Semi; Gap + Para":
        def f(curvature):
            sample.dos_para(system, curvature)
            plot_info = {"title": sample_name.value + " DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                         "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.005]}
            theory_plot(system.en, sample.DoS, plot_info)
            return
        ipy.interact(f, curvature=ipy.FloatSlider(min=0.1, max=4.0, step=0.1, value=1.0,
                                                  description='$Curvature:$'), continuous_update=True)
        return sample

    # - For a semiconductor sample with a band-gap, parabolic dispersion and surface states
    if sample_dos.value == "Semi; Gap + Para + SS":
        def f(curvature, ss1_peak_pos, ss1_peak_width, ss2_peak_pos, ss2_peak_width):
            surf_states = np.array([[ss1_peak_pos, ss1_peak_width], [ss2_peak_pos, ss2_peak_width]])
            sample.dos_parass(system, curvature, 1, surf_states)
            plot_info = {"title": sample_name.value + " DOS", "xlabel": "$Energy$ $/$ $(eV)$",
                         "ylabel": "$Normalised$ $DOS$ $/$ $(eV^{-1})$", "ylim": [0, 0.006]}
            theory_plot(system.en, sample.DoS, plot_info)
            return
        ipy.interact(f, curvature=ipy.FloatSlider(min=0.1, max=4.0, step=0.05, value=1.0, description='$Curvature:$'),
                     ss1_peak_pos=ipy.FloatSlider(min=-2.0, max=2.0, step=0.01, value=0.4 + sample_eoffset.value,
                                                  description='$RHS$ $Peak$ $position:$'),
                     ss1_peak_width=ipy.FloatSlider(min=0.001, max=0.050, step=0.001, value=0.02,
                                                    description='$RHS$ $Peak$ $width:$'),
                     ss2_peak_pos=ipy.FloatSlider(min=-2.0, max=2.0, step=0.01, value=-0.4 + sample_eoffset.value,
                                                  description='$LHS$ $Peak$ $position:$'),
                     ss2_peak_width=ipy.FloatSlider(min=0.001, max=0.050, step=0.001, value=0.02,
                                                    description='$LHS$ $Peak$ $width:$'),
                     continuous_update=True)
        return sample


def tunneling_plot(system, tip, sample, tunneling_form):
    # Defining the tunelling matrix element
    tme = TunnMatrix(system)

    # Defining the tunneling matrix element object from the button selection [tunneling_form] for a given bias [vbias]
    def f(vbias):
        # - If a constant matrix element form is selected
        if tunneling_form.value == "Constant":
            tme.tunn_const(system)
        # - If a WKB matrix element form is selected
        if tunneling_form.value == "WKB approx.":
            tme.tunn_wkb(tip, sample, system)
        # - Plotting the tunneling matrix element for a given bias [vbias]
        global_max = 1.05*np.max(tme.tunn_element)
        global_min = 1.05*np.min(tme.tunn_element)
        pos = (np.arange(0, len(system.bias))[vbias == system.bias])[0]
        if system.bias[pos] > 0:
            plot_info = {"title": "Transmission probability", "xlabel": "$Tip$ $bias$ $/$ $(V)$",
                         "ylabel": "$Transmission$ $probability$", "ylim": [0, global_max]}
            theory_plot(system.en, tme.tunn_element[pos], plot_info)
            return
        else:
            plot_info = {"title": "Transmission probability", "xlabel": "$Tip$ $bias$ $/$ $(V)$",
                         "ylabel": "$Transmission$ $probability$", "ylim": [global_min, 0]}
            theory_plot(system.en, tme.tunn_element[pos], plot_info)
            return
    # Directly interacting with the voltage bias
    bias_step = np.around(abs(system.bias[1] - system.bias[0]), 2)
    ipy.interact(f, vbias=ipy.FloatSlider(min=system.bias[0], max=system.bias[-1], step=bias_step, value=system.bias[0],
                                          description='$Tip$ $bias[V]:$'), continuous_update=True)
    return tme

def summary_plot(system, tip, sample, tme, iv, didv):
    def f(vbias):
        pos = (np.arange(0, len(system.bias))[vbias == system.bias])[0]
        zeropos = (np.arange(0, len(system.bias)))[system.bias == 0][0]
        fig, ax = plt.subplots(figsize=(20, 10))

        # Plot the static, grounded Fermi distribution
        plt.subplot(2, 5, 1)
        plt.plot(tme.fermi_ground, system.en, 'b-', linewidth=3.0)
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="g", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.grid(True)
        plt.title("$f_{GROUND}[E]$")
        plt.ylim([np.min(system.en) * 0.95, np.max(system.en) * 1.05])

        # Plot the tunneling matrix element as a function of voltage bias
        plt.subplot(2, 5, 2)
        plt.plot(tme.tunn_element[pos], system.en, 'b-', linewidth=3.0)
        plt.fill_between(tme.tunn_element[pos], system.en, tme.tunn_element[pos], color='blue', alpha='0.3')
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="g", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.axhline(y=system.bias[pos], color="r", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.grid(True)
        plt.title("$f_{GROUND}[E].f_{BIAS}[E].T[E]$")
        plt.ylim([np.min(system.en) * 0.95, np.max(system.en) * 1.05])
        plt.xlim([-3e-7, 3e-7])

        # Plotting the changing, biased Fermi distribution
        plt.subplot(2, 5, 3)
        plt.plot(tme.fermi_bias[pos], system.en, 'b-', linewidth=3.0)
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="g", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.axhline(y=system.bias[pos], color="r", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.grid(True)
        plt.title("$f_{BIAS}[E]$")
        plt.ylim([np.min(system.en) * 0.95, np.max(system.en) * 1.05])

        # Plot the tip DOS as a function of the voltage bias
        plt.subplot(2, 5, 4)
        plt.plot(tip.DoSbiased[pos], system.en, 'b-', linewidth=3.0)
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="g", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.axhline(y=system.bias[pos], color="r", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.grid(True)
        plt.title("$Tip$ $DOS$")
        plt.ylim([np.min(system.en) * 0.95, np.max(system.en) * 1.05])

        # Plot the static, grounded sample DOS
        plt.subplot(2, 5, 5)
        plt.plot(sample.DoS, system.en, 'b-', linewidth=3.0)
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="g", linestyle="-", linewidth=3.5, alpha=0.75)
        plt.grid(True)
        plt.title("$Sample$ $DOS$")
        plt.ylim([np.min(system.en) * 0.95, np.max(system.en) * 1.05])

        # Plot the I-V
        plt.subplot(2, 2, 3)
        plt.plot(system.bias, iv, 'b-', linewidth=3.0)
        plt.fill_between(system.bias, iv, color='blue', alpha='0.3')
        plt.plot(system.bias[zeropos], iv[zeropos], 'go', markersize=10.0, alpha=0.8)
        plt.plot(system.bias[pos], iv[pos], 'ro', markersize=10.0, alpha=0.8)
        plt.plot(np.array([system.bias[pos], system.bias[pos]]), np.array([0, iv[pos]]), 'r-', linewidth=1.2,
                 alpha=0.8)
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.grid(True)
        plt.title("$I(V)$")
        plt.xlim([np.min(system.bias), np.max(system.bias)])

        # Plot the dI/dV
        plt.subplot(2, 2, 4)
        plt.plot(system.bias, didv, 'b-', linewidth=3.0)
        plt.fill_between(system.bias, didv, color='blue', alpha='0.3')
        plt.plot(system.bias[zeropos], didv[zeropos], 'go', markersize=10.0, alpha=0.8)
        plt.plot(system.bias[pos], didv[pos], 'ro', markersize=10.0, alpha=0.8)
        plt.plot(np.array([system.bias[pos], system.bias[pos]]), np.array([0, didv[pos]]), 'r-',
                 linewidth=1.2, alpha=0.8)
        compare_sample_dos = (sample.DoS/max(sample.DoS))*max(didv)
        plt.plot(system.en, compare_sample_dos, 'k--',linewidth=1.5)
        # Formatting the figure
        plt.axvline(x=0, color="k", linestyle="-", linewidth=1)
        plt.axhline(y=0, color="k", linestyle="-", linewidth=1)
        plt.grid(True)
        plt.title("$dI/dV(V)$")
        plt.xlim([np.min(system.bias), np.max(system.bias)])

        # Show the plots
        plt.show()
        return

    # Directly interacting with the voltage bias
    bias_step = np.around(abs(system.bias[1] - system.bias[0]), 2)
    ipy.interact(f, vbias=ipy.FloatSlider(min=system.bias[0], max=system.bias[-2], step=bias_step,
                                          value=system.bias[0], description='$Tip$ $bias[V]:$'),
                 continuous_update=True)
    return


