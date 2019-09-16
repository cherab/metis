from raysect.core import Vector3D

from cherab.core import Plasma, Species, Maxwellian
from cherab.core.atomic.elements import lookup_element, lookup_isotope, helium
from cherab.core.math import Constant3D
from cherab.core.utility import RecursiveDict
from cherab.metis import read_hdf5
from cherab.openadas import OpenADAS
from cherab.tools.equilibrium.efit import EFITEquilibrium
from cherab.tools.plasmas.ionisation_balance import from_elementdensity
from cherab.core.math import Interpolate1DCubic, ClampOutput1D
from cherab.core.math.constant import ConstantVector3D

from scipy.constants import electron_mass, atomic_mass
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, interp1d

import numpy as np


class METISModel:

    def __init__(self, filepath=None):

        self._flush_properties()

        if filepath is not None:
            self.filepath = filepath

        self._equilibrium = None

    def _flush_properties(self):
        """
        reset vaues connected to data
        :return:
        """
        self._zerod_data = {}
        self._profile0d_data = {}

        self._zerod_interpolator = RecursiveDict()  # holds interpolator for zerod dataset
        self._profile1d_interpolator = RecursiveDict()  # holds interpolators for profile1d dataset

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
        profil0d["psin"] = np.divide((profil0d["psi"] - profil0d["psi"].min(axis=0)),
                                     profil0d["psi"].max(axis=0) - profil0d["psi"].min(axis=0))

        self._zerod_data = zerod
        self._profile0d_data = profil0d

    @property
    def equilibrium(self):
        return self._equilibrium

    @equilibrium.setter
    def equilibrium(self, value):
        if not isinstance(value, EFITEquilibrium):
            raise TypeError("Value has to be EFITEquilibrium, but {0} passed.".format(value))

        self._equilibrium = value

    @property
    def _time_shape(self):
        return self._zerod_data["temps"].shape[0]

    @property
    def _profile_shape(self):
        return self._profile0d_data["psi"].shape[0]

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

    def zerod(self, quantity, time=None):
        """
        Returns array of values for the specified quantity from zerod dataset.
        :param quantity: Name of physical quantity from zerod dataset
        :return:
        """
        if not quantity in list(self._zerod_data.keys()):
            raise ValueError("quantity {0}, not in zerod METIS group: {1}".format(quantity, self._zerod_data.keys()))

        if time is None:
            return self._zerod_data[quantity]
        else:
            index = self._get_nearest_index(time, self.time)
            return self._zerod_data[quantity][index]

    def profile1d(self, quantity, time=None):
        """
        Returns array of values for the specified quantity from profiles1d dataset. If time is not specified, 2D
        array with all time slices is returned. If time is specified a the nearest time slice is returned as 1D array.
        :param quantity: Name of physical quantity from profiles1d dataset
        :param time: Time setting the nearest time silece to be returned.
        :return: np.ndarray object
        """

        if not quantity in list(self._profile0d_data.keys()):
            raise ValueError(
                "quantity {0}, not in zerod METIS group: {1}".format(quantity, self._profile0d_data.keys()))

        if quantity == "xli" and time:
            return self._profile0d_data["xli"][:, 0]

        if quantity == "xli" and not time:
            return self._profile0d_data["xli"]

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

    def zerod_interpolate(self, quantity, time, kind="cubic"):
        """
        Interpolates quantity for given time value. The quantity name has to be a valid METIS 0D quantity.
        :param quantity: str, a valid zerod quantity
        :param time: time within the time frame of the METIS simulation
        :param kind: 'cubic' or 'linear', default='cubic', degree of the interpolator
        :return: float
        """
        # check validity of the quantity requested
        if not quantity in list(self._zerod_data.keys()):
            raise ValueError(
                "{0} passed as quantity, but has to be one of: {1}".format(quantity, self._zerod_data.keys()))

        # return interpolated values and create interpolator if missing
        try:
            value = self._zerod_interpolator[quantity][kind](time)
        except TypeError:
            # construct the right interpolator
            if kind == "cubic" or kind == "linear":
                self._zerod_interpolator[quantity][kind] = interp1d(self.time, self._zerod_data[quantity], kind=kind)
            else:
                raise ValueError("degree can be only 'linear' or 'cubic' but '{0}' passed.".format(kind))

            value = self._zerod_interpolator[quantity][kind](time)

        return value

    def profile1d_interpolate(self, quantity, time, kind="cubic", **free_variable):
        """
        Attempts to interpolate requested quantity for the specified free variable and time.
        :param quantity: str, a valid profile1d quantity
        :param time: time within the simulation time frame
        :param kind: kind of the interpolation. Can be cubic or linear
        :param free_variable: Free variable to interpolate on. The name of the free_variable passed will determine the
               name of the quantity from the profile1d dataset. If no free_variable is specified, psin and the default
               values will be used
        :return: np.ndarray object
        """

        # check validity of the quantity requested
        if not quantity in list(self._profile0d_data.keys()):
            raise ValueError(
                "quantity {0}, not in zerod METIS group: {1}".format(quantity, self._profile0d_data.keys()))

        # prepare free variable and check validity
        if not free_variable:
            free_variable_name = "psin"
            free_variable_data = np.linspace(0, 1, self._profile_shape, endpoint=True)
        else:
            free_variable_name = list(free_variable.keys())[0]
            free_variable_data = free_variable[free_variable_name]

        if not free_variable_name in list(self._profile0d_data.keys()):
            raise ValueError(
                "quantity {0}, not in profil0d METIS group: {1}".format(free_variable_name,
                                                                        self._profile0d_data.keys()))

        # return interpolated values and create interpolator if missing
        try:
            value = self._profile1d_interpolator[quantity][free_variable_name][kind](time, free_variable_data).squeeze()
        except TypeError:
            # prepare values for the interpolator initialization
            time_vector = np.tile(self.time, self._profile_shape)[:, np.newaxis]

            # construct free variable 1D array of values
            if not free_variable_name == "xli":  # get time-space profiles into 1D free variable array
                fv_vector = self._profile0d_data[free_variable_name].flatten()[:, np.newaxis]
            else:  # xli is only 1D array and has to be expanded to fit rest of the interpolation
                fv_vector = np.repeat(self._profile0d_data["xli"][:, 0], self._time_shape)[:, np.newaxis]

            if not quantity == "xli":  # get time-space profiles into 1D free variable array
                v_vector = self._profile0d_data[quantity].flatten()
            else:
                v_vector = np.repeat(self._profile0d_data["xli"][:, 0], self._time_shape)[:, np.newaxis]

            pnts = np.concatenate((time_vector, fv_vector), axis=1)
            # construct the right interpolator
            if kind == "cubic":
                self._profile1d_interpolator[quantity][free_variable_name][kind] = CloughTocher2DInterpolator(pnts,
                                                                                                              v_vector,
                                                                                                              rescale=True)
            elif kind == "linear":
                self._profile1d_interpolator[quantity][free_variable_name][kind] = LinearNDInterpolator(pnts, v_vector,
                                                                                                        rescale=True)
            # evaluate it
            value = self._profile1d_interpolator[quantity][free_variable_name][kind](time, free_variable_data).squeeze()

        return value

    def interpolator1d(self, quantity, time, kind="cubic", **free_variable):
                # check validity of the quantity requested
        if not quantity in list(self._profile0d_data.keys()):
            raise ValueError(
                "quantity {0}, not in zerod METIS group: {1}".format(quantity, self._profile0d_data.keys()))

        # prepare free variable and check validity
        if not free_variable:
            free_variable_name = "psin"
            free_variable_data = np.linspace(0, 1, self._profile_shape, endpoint=True)
        else:
            free_variable_name = list(free_variable.keys())[0]
            free_variable_data = free_variable[free_variable_name]

        if not free_variable_name in list(self._profile0d_data.keys()):
            raise ValueError(
                "quantity {0}, not in profil0d METIS group: {1}".format(free_variable_name,
                                                                        self._profile0d_data.keys()))

        profile = self.profile1d_interpolate(quantity, time, kind=kind, **{free_variable_name: free_variable_data})

        interpolator = Interpolate1DCubic(free_variable_data, profile, extrapolate=True, extrapolation_type="nearest")

        return interpolator

    def equilibrium_map2d(self, quantity, time, value_outside_lcfs=0.0, interpolate=True):
        """
        Generated 2d interpolator using EFITEquilibrium.map2d function

        :param quantity: Physical quantity from profiles1d provided by METIS
        :param time: Time to interpolate the profile for or to search for the nearest simulation time-slice. Depends on value of interpolate parameter
        :param value_outside_lcfs: Value to return if point is outside lcfs
        :param interpolate: Defaut True, uses interpolation to obtain quantity values of given time and normalized poloidal flux.
        If false the quantity values from the nearest simulated time slice are used.
        :return: Function2D object.
        """

        # obtain values for interpolation either diectly by taking a time-slice profile or by interpolation
        if interpolate:
            psin = self.profile1d_interpolate("psin", time, xli=self._profile0d_data["xli"][:, 0])
            values = self.profile1d_interpolate(quantity, time)
        else:
            values = self.profile1d(quantity, time)
            psin = self.profile1d("psin", time)

        profile = Interpolate1DCubic(psin, values, extrapolate=True, extrapolation_type="nearest")
        return self._equilibrium.map2d(profile, value_outside_lcfs)

    def equilibrium_map3d(self, quantity, time, value_outside_lcfs=0.0, interpolate=True):
        """
        Generate 3d interpolator using EFITEquilibrium.map3d function

        :param quantity: Physical quantity from profiles1d provided by METIS
        :param time:Time to interpolate the profile for or to search for the nearest simulation time-slice. Depends on value of interpolate parameter
        :param value_outside_lcfs: Value to return if point is outside lcfs
        :param interpolate: Defaut True, uses interpolation to obtain quantity values of given time and normalized poloidal flux.
        If false the quantity values from the nearest simulated time slice are used.
        :return: Function3D object.
        """

        # obtain values for interpolation either diectly by taking a time-slice profile or by interpolation
        if interpolate:
            psin = self.profile1d_interpolate("psin", time, xli=self._profile0d_data["xli"][:, 0])
            values = self.profile1d_interpolate(quantity, time)
        else:
            psin = self.profile1d("psin", time)
            values = self.profile1d(quantity, time)

        profile = Interpolate1DCubic(psin, values)
        return self._equilibrium.map3d(profile, value_outside_lcfs)

    def equilibrium_map_vector2d(self, toroidal_quantity=None, poloidal_quantity=None, normal_quantity=None, time=None,
                                 value_outside_lcfs=0.0, interpolate=True):
        """
        Uses EFITEquilibrium.map_vector2d to create 2D vector field interpolator.
        :param toroidal_quantity: Toroidal vector component
        :param poloidal_quantity: Pooidal vector component
        :param normal_quantity: Component parallel toflux surface normal
        :param time: Time to interpolate the profile for or to search for the nearest simulation time-slice. Depends on value of interpolate parameter
        :param value_outside_lcfs: Value to return if point is outside lcfs
        :param interpolate: Defaut True, uses interpolation to obtain quantity values of given time and normalized poloidal flux.
        If false the quantity values from the nearest simulated time slice are used.
        :return: VectorFunction2D object.
        """

        if time is None:
            time = self.time[0]

        # obtain values for interpolation either diectly by taking a time-slice profile or by interpolation
        if interpolate:  # iterpolate psin and values
            psin = self.profile1d_interpolate("psin", time, xli=self._profile0d_data["xli"][:, 0])

            # pass zeros if not specified
            if toroidal_quantity is None:
                toroidal_values = np.zeros_like(psin)
            else:
                toroidal_values = self.profile1d_interpolate(toroidal_quantity, time)

            if poloidal_quantity is None:
                poloidal_values = np.zeros_like(psin)
            else:
                poloidal_values = self.profile1d_interpolate(poloidal_quantity, time)

            if normal_quantity is None:
                normal_values = np.zeros_like(psin)
            else:
                normal_values = self.profile1d_interpolate(normal_quantity, time)
        else:
            psin = self.profile1d("psin", time)

            if toroidal_quantity is None:
                toroidal_values = np.zeros_like(psin)
            else:
                toroidal_values = self.profile1d(toroidal_quantity, time)

            if poloidal_quantity is None:
                poloidal_values = np.zeros_like(psin)
            else:
                poloidal_values = self.profile1d(poloidal_quantity, time)

            if normal_quantity is None:
                normal_values = np.zeros_like(psin)
            else:
                normal_values = self.profile1d(poloidal_quantity, time)

        profile_toroidal = Interpolate1DCubic(psin, toroidal_values, extrapolate=True, extrapolation_type="nearest")
        profile_poloidal = Interpolate1DCubic(psin, toroidal_values, extrapolate=True, extrapolation_type="nearest")
        profile_normal = Interpolate1DCubic(psin, toroidal_values, extrapolate=True, extrapolation_type="nearest")

        return self._equilibrium.map_vector2d((psin, toroidal_values), (psin, poloidal_values), (psin, normal_values))

    def equilibrium_map_vector3d(self, toroidal_quantity=None, poloidal_quantity=None, normal_quantity=None,
                                 time=None, value_outside_lcfs=0.0, interpolate=True):
        """
        Uses EFITEquilibrium.map_vector2d to create 2D vector field interpolator.
        :param toroidal_quantity: Toroidal vector component
        :param poloidal_quantity: Pooidal vector component
        :param normal_quantity: Component parallel toflux surface normal
        :param time: Time to interpolate the profile for or to search for the nearest simulation time-slice. Depends on value of interpolate parameter
        :param value_outside_lcfs: Value to return if point is outside lcfs
        :param interpolate: Defaut True, uses interpolation to obtain quantity values of given time and normalized poloidal flux.
        If false the quantity values from the nearest simulated time slice are used.
        :return: VectorFunction3D object.
        """

        if time is None:
            time = self.time[0]

        if interpolate:  # iterpolate psin and values
            psin = self.profile1d_interpolate("psin", time, xli=self._profile0d_data["xli"][:, 0])

            # pass zeros if not specified
            if toroidal_quantity is None:
                toroidal_values = np.zeros_like(psin)
            else:
                toroidal_values = self.profile1d_interpolate(toroidal_quantity, time)

            if poloidal_quantity is None:
                poloidal_values = np.zeros_like(psin)
            else:
                poloidal_values = self.profile1d_interpolate(poloidal_quantity, time)

            if normal_quantity is None:
                normal_values = np.zeros_like(psin)
            else:
                normal_values = self.profile1d_interpolate(normal_quantity, time)
        else:
            psin = self.profile1d("psin", time)

            if toroidal_quantity is None:
                toroidal_values = np.zeros_like(psin)
            else:
                toroidal_values = self.profile1d(toroidal_quantity, time)

            if poloidal_quantity is None:
                poloidal_values = np.zeros_like(psin)
            else:
                poloidal_values = self.profile1d(poloidal_quantity, time)

            if normal_quantity is None:
                normal_values = np.zeros_like(psin)
            else:
                normal_values = self.profile1d(poloidal_quantity, time)

        return self._equilibrium.map_vector3d((psin, toroidal_values), (psin, poloidal_values), (psin, normal_values))

    def create_plasma(self, time, plasma: Plasma = None, interpolate=False, hydrogen_isotope_number=2, t_cold_neutrals=1,
                      atomic_data=None, main_impurity=None):

        if atomic_data is None:
            atomic_data = OpenADAS()

        if plasma is None:
            plasma = Plasma()

        plasma_rotation = self.equilibrium_map_vector3d(toroidal_quantity="vtor", poloidal_quantity="vtheta", time=time,
                                                        interpolate=interpolate)
        # populate electrons
        if interpolate:
            n_e = self.profile1d_interpolate("nep", time)
            t_e = self.profile1d_interpolate("tep", time)
            n_h0_cold = self.profile1d_interpolate("n0m", time)
            n_h0_hot = self.profile1d_interpolate("n0", time)
        else:
            n_e = self.profile1d("nep", time)
            t_e = self.profile1d("tep", time)
            n_h0_cold = self.profile1d("n0m", time)
            n_h0_hot = self.profile1d("n0", time)

        electron_density = self.equilibrium_map3d("nep", time, interpolate=interpolate)
        electron_temperature = self.equilibrium_map3d("tep", time, interpolate=interpolate)

        plasma.electron_distribution = Maxwellian(electron_density, electron_temperature, plasma_rotation,
                                                  electron_mass)

        # populate hydrogen ions
        hydrogen_isotope = lookup_isotope("hydrogen", number=hydrogen_isotope_number)

        t_i = self.equilibrium_map3d("tip", time, interpolate=interpolate)

        #todo:uncomment when more distributions of a same species can be put into plasma composition
        #h0_hot_density = self.equilibrium_map3d("n0", time, interpolate=interpolate)
        #h0_hot_distribution = Maxwellian(h0_hot_density, t_i, plasma_rotation,
        #                                 hydrogen_isotope.atomic_weight * atomic_mass)
        #plasma.composition.add(Species(hydrogen_isotope, 0, h0_hot_distribution))

        h0_cold_density = self.equilibrium_map3d("n0m", time, interpolate=interpolate)
        h0_cold_temperature = Constant3D(0)
        h0_cold_distribution = Maxwellian(h0_cold_density, h0_cold_temperature, plasma_rotation,
                                          hydrogen_isotope.atomic_weight * atomic_mass)
        plasma.composition.add(Species(hydrogen_isotope, 0, h0_cold_distribution))

        h1_density = self.equilibrium_map3d("n1p", time, interpolate=interpolate)
        h1_distribution = Maxwellian(h1_density, t_i, plasma_rotation, hydrogen_isotope.atomic_weight * atomic_mass)
        plasma.composition.add(Species(hydrogen_isotope, 1, h1_distribution))

        # add helium species if present
        if interpolate:
            he_density = self.profile1d_interpolate("nhep", time)
        else:
            he_density = self.profile1d("nhep", time)

        if np.any(he_density > 0):
            helium = lookup_element("helium")

            helium_balance = from_elementdensity(atomic_data, helium, he_density, n_e, t_e,
                                                 tcx_donor=lookup_element("hydrogen"), tcx_donor_charge=0,
                                                 tcx_donor_n=n_h0_cold)

            for charge, dens in helium_balance.items():
                dist = Maxwellian(dens, t_i, plasma_rotation, helium.atomic_weight * atomic_mass)
                plasma.composition.add(Species(helium, charge, dist))

        if main_impurity is not None:
            if interpolate:
                main_density = self.profile1d_interpolate("nzp", time)
            else:
                main_density = self.profile1d("nzp", time)

            balance = from_elementdensity(atomic_data, main_impurity, main_density, n_e, t_e,
                                          tcx_donor=lookup_element("hydrogen"), tcx_donor_charge=0,
                                          tcx_donor_n=n_h0_cold)

            for charge, dens in balance.items():
                dist = Maxwellian(dens, t_i, plasma_rotation, main_impurity.atomic_weight * atomic_mass)
                plasma.composition.add(Species(main_impurity, charge, dist))

        return plasma

    def match_zeff(self, time, plasma: Plasma = None, interpolate=True, element = helium):
        """
        Fills plasma with absolutely stripped ions to match zeff simulated by metis. This can be useful when concrete
        impurity is not known which is not important for some kinds of simulations e.g. bremsstrahlung radiation.
        Returned plasma profile has zeff equal or higher than metis results.

        :param time: shot time
        :param plasma: Plasma object to be modified.
        :param interpolate: Specifies wether used metis profiles should be interpolated. If True Metis profiles are
        interpolated over time and poloidal flux. If False profiles for nearest simulated time slice are chosen.
        :param element: Element to fill the plasma with.
        :return: Plasma object with matched zeff
        """

        if plasma is None:#Create instance of Plasma if not passed as argument
            plasma = self.create_plasma(time=time, interpolate=interpolate)

        psin = self.profile1d("psin", time=time)
        zeff_metis = self.profile1d("zeff", time= time)
        zeff_plasma= np.ones_like(zeff_metis)

        for i, v in enumerate(psin):
            #it seems that psin=1 due to interpolation errors can do problems with finding no charged particles in cherab plasma object.
            try:
                zeff_plasma[i] = plasma.z_effective(self.equilibrium.psin_to_r(v), 0, 0)
            except ValueError:
                zeff_plasma[i] =plasma.z_effective(self.equilibrium.psin_to_r(v) - 0.001, 0, 0)
                print("Plasma does not contain any ionised species for psi_normalised = {0}, using value for spi_normalised = {1}".format(v, v-0.001))

        n_d = self.profile1d("n1p", time=time)
        n_e = self.profile1d("nep", time=time)
        t_i = self.equilibrium_map3d("tip", time=time, interpolate=False)
        n_fill = n_d * (zeff_metis - zeff_plasma)/(element.atomic_number ** 2 - element.atomic_number * zeff_metis)

        #remove negative densities
        tmp = np.where(n_fill < 0)[0]
        n_fill[tmp] = 0

        #create 3D interpolator
        n_fill = self.equilibrium.map3d((psin, n_fill))

        #create plasma species and add it to the plasma
        fill_distribution = Maxwellian(n_fill, t_i, ConstantVector3D(Vector3D(0, 0, 0)), element.atomic_weight * atomic_mass)
        fill_species = Species(element, element.atomic_number, fill_distribution)
        plasma.composition.add(fill_species)

        return plasma

