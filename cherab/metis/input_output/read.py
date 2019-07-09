import numpy as np
import h5py

folder = "/compass/Shared/Common/COMPASS-UPGRADE/RP1_Design/Scenarios/6.4/METIS_minorradius_28cm/"
file = "CU_6_4_dsol.mat"

def read_hdf5(file_path):
    """

    :param file_path: Path to the metis file saved in hdf5 file. To produce this file, matlab save function with
    the '-v7.3' or higher.
    :return: time, zerod, profile0d
        zerod holds 0D time-dependent profiles
        profile0D holds 1D profiles with coordinates [magnetic, temporal]
    """

    with h5py.File(file_path, "r") as metfile:

        #read the time traces of the 0D (zerod) datasets
        group_zerod = metfile["post/zerod"]

        zerod = {} #all the 0D profiles will be stored in a dictionary

        for key, item in group_zerod.items():
            zerod[key] = np.array(item).squeeze()

        #read 1d profiles from profil0d
        group_profile0d = metfile["post/profil0d"]

        profile0d = {}


        for key, item in group_profile0d.items():
            profile0d[key] = np.array(item)

    return zerod, profile0d