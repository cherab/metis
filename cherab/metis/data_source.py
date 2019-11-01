from cherab.metis import read_hdf5
import numpy as np
from cherab.core.utility import Notifier

class MetisDataSource_base:
    """
    Base class for METIS data source. This class obtans data from a data source and serves as a data getter for the
    METISModel class. The reason is ways of storing METIS simulations may vary for different institutes. This Class
    can be inherited within other cherab modules and the _get_data method is to be implemented to suit specific needs.
    """
    def __init__(self):

        self._notifier = Notifier()
        self._flush_variables()

    def get_data(self):
        """
        Reads data from the data source and calls notifier to notify listening objects that data have changed.
        :return:
        """

        self._get_data()
        self._notifier.notify()

    def _get_data(self):
        """
        Reads data from the data source.
        :return:
        """
        raise NotImplementedError("Virtual method must be implemented in a sub-class.")

    def flush_data(self):
        """
        Re-initializes class variables holding metis data to emtpy them out. Used when the data in the source is changed.
        :return:
        """

    def _flush_variables(self):
        """
        reset variables connected to data
        :return:
        """
        self._zerod_data = {}
        self._profile0d_data = {}

class MatlabMetisFile(MetisDataSource_base):
    """
    METIS data source, reading data from metis matlab hdf5 files.
    """
    def __init__(self, file_path: str = None):

        super().__init__()
        if file_path is not None:
            self._filepath = file_path
            self.get_data()



    def _get_data(self):
        """
        Reads data from metis file and uptades the object.
        :return:
        """
        self._flush_variables()

        if self._filepath is None:
            raise ValueError('Self.filepath has to be specified')

        zerod, profil0d = read_hdf5(self._filepath)

        self._zerod_data = zerod
        self._profile0d_data = profil0d
        self._time = zerod["temps"]

        # calculate normalized poloidal flux and add it to profiles
        profil0d["psin"] = np.divide((profil0d["psi"] - profil0d["psi"][0, :]),
                                     profil0d["psi"][-1, :] - profil0d["psi"][0, :])

        self._zerod_data = zerod
        self._profile0d_data = profil0d

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
        self.get_data()


