from cherab.metis import read_hdf5
from cherab.core.utility import RecursiveDict

from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, interp1d

import numpy as np

class METISModel():

    def __init__(self, filepath = None):

        self._flush_properties()

        if filepath is not None:
            self.filepath = filepath


    def _flush_properties(self):
        """
        reset vaues connected to data
        :return:
        """
        self._zerod_data = {}
        self._profile0d_data = {}

        self._zerod_interpolator = RecursiveDict() # holds interpolator for zerod dataset
        self._profile1d_interpolator = RecursiveDict() #holds interpolators for profile1d dataset
        self._profile_shape = None
        self._time_shape = None

    def _get_nearest_index(self, value, vector):
        """
        Get index of the nearest vector value to the value
        :param value: float, value to find the nearest index for
        :param vector: array with values to search in
        :return:
        """
        vals = np.abs(vector - value)
        argmin = vals.argmin()

        return argmin


    def update_data(self):
        """
        Reads data from metis file and uptades the object.
        :return:
        """

        if self._filepath is None:
            raise ValueError('Self.filepath has to be specified')

        zerod, profil0d = read_hdf5(self._filepath)

        self._flush_properties()

        self._zerod_data = zerod
        self._profile0d_data = profil0d
        self._time = zerod["temps"]

        # calculate normalized poloidal flux and add it to profiles
        profil0d["psin"] = np.zeros_like(profil0d["psi"])
        for index, value in enumerate(profil0d["psi"].T):

            profil0d["psin"][:, index] = (value - value.min())/(value.max() - value.min())

        self._time_shape = profil0d["psi"].shape[1]
        self._profile_shape = profil0d["psi"].shape[0]
        self._zerod_data = zerod
        self._profile0d_data = profil0d

    @property
    def zerod_list(self):
        """
        List of quantities in zerod dataset.
        :return:
        """
        return list(self._zerod_data.keys())

    @property
    def profile1d_list(self):
        """
        List of quantities in profile1d dataset.
        :return:
        """
        return list(self._profile0d_data.keys())

    def zerod(self, quantity):
        """
        Returns array of values for the specified quantity from zerod dataset.
        :param quantity: Name of physical quantity from zerod dataset
        :return:
        """
        if not quantity in list(self._zerod_data.keys()):
            raise ValueError("quantity {0}, not in zerod METIS group: {1}".format(quantity, self._zerod_data.keys()))

        return self._zerod_data[quantity]

    def profile1d(self, quantity, time=None):
        """
        Returns array of values for the specified quantity from profiles1d dataset. If time is not specified, 2D
        array with all time slices is returned. If time is specified a the nearest time slice is returned as 1D array.
        :param quantity: Name of physical quantity from profiles1d dataset
        :param time: Time setting the nearest time silece to be returned.
        :return:
        """


        if not quantity in list(self._profile0d_data.keys()):
            raise ValueError("quantity {0}, not in zerod METIS group: {1}".format(quantity, self._profile0d_data.keys()))


        if time:
            time_arg = self._get_nearest_index(time, self.time)
            return self._profile0d_data[quantity][:, time_arg]
        else:
            return self._profile0d_data[quantity]

    @property
    def filepath(self):
        """
        Path to the METIS file.
        :return:
        """
        return self._filepath

    @filepath.setter
    def filepath(self, filepath):
        """
        if filepath is changed, the object is reset and data are loaded
        :param filepath:
        :return:
        """
        self._filepath = filepath
        self.update_data()

    @property
    def time(self):
        """
        time vector of the timeslices
        :return:
        """
        return self._zerod_data["temps"]

    def interpolate_zerod(self, quantity, time, kind="cubic"):
        """
        Interpolates quantity for given time value. The quantity name has to be a valid METIS 0D quantity.
        :param quantity: str, a valid zerod quantity
        :param time: time within the time frame of the METIS simulation
        :param kind: 'cubic' or 'linear', default='cubic', degree of the interpolator
        :return: float
        """
        #check validity of the quantity requested
        if not quantity in list(self._zerod_data.keys()):
            raise ValueError("{0} passed as quantity, but has to be one of: {1}".format(quantity, self._zerod_data.keys()))

        #return interpolated values and create interpolator if missing
        try:
            value = self._zerod_interpolator[quantity][kind](time)
        except TypeError:
            #construct the right interpolator
            if kind == "cubic" or kind == "linear":
                self._zerod_interpolator[quantity][kind] = interp1d(self.time, self._zerod_data[quantity], kind=kind)
            else:
                raise ValueError("degree can be only 'linear' or 'cubic' but '{0}' passed.".format(kind))

            value = self._zerod_interpolator[quantity][kind](time)

        return value

    def interpolate_profile1d(self, quantity, time, kind ="cubic",**free_variable ):
        """
        Attempts to interpolate requested quantity for the specified free variable and time.
        :param quantity: str, a valid profile1d quantity
        :param time: time within the simulation time frame
        :param kind: kind of the interpolation. Can be cubic or linear
        :param free_variable: Free variable to interpolate on. The name of the free_variable passed will determine the
               name of the quantity from the profile1d dataset. If no free_variable is specified, psin and the default
               values will be used
        :return:
        """

        #check validity of the quantity requested
        if not quantity in list(self._profile0d_data.keys()):
            raise ValueError("quantity {0}, not in zerod METIS group: {1}".format(quantity, self._profile0d_data.keys()))

        #prepare free variable and check validity
        if not free_variable:
            free_variable_name = "psin"
            free_variable_data = np.linspace(0, 1, self._profile_shape, endpoint=True)
        else:
            free_variable_name = list(free_variable.keys())[0]
            free_variable_data = free_variable[free_variable_name]

        if not free_variable_name in list(self._profile0d_data.keys()):
            raise ValueError("quantity {0}, not in zerod METIS group: {1}".format(free_variable_name, self._profile0d_data.keys()))

        #return interpolated values and create interpolator if missing
        try:
            value = self._profile1d_interpolator[quantity][free_variable_name][kind](time, free_variable_data).squeeze()
        except TypeError:
            #prepare values for the interpolator initialization
            time_vector = np.tile(self.time, self._profile_shape)[:, np.newaxis]
            fv_vector = self._profile0d_data[free_variable_name].flatten()[:, np.newaxis]
            v_vector = self._profile0d_data[quantity].flatten()
            pnts = np.concatenate((time_vector, fv_vector), axis=1)
            #construct the right interpolator
            if kind == "cubic":
                self._profile1d_interpolator[quantity][free_variable_name][kind] = CloughTocher2DInterpolator(pnts, v_vector, rescale=True)
            elif kind == "linear":
                self._profile1d_interpolator[quantity][free_variable_name][kind] = LinearNDInterpolator(pnts, v_vector, rescale=True)
            #evaluate it
            value = self._profile1d_interpolator[quantity][free_variable_name][kind](time, free_variable_data).squeeze()

        return value
