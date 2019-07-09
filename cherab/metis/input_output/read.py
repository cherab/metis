import h5py
import numpy as np


def read_hdf5(file_path):
    """
    Read outputs of metis simulation saved in hdf5 file. The file has to be created by using save function in matlab
    with the parameter '-v7.3' to be hdf5 file. Only output saved in post group within hdf5 file is read, both 0D
    profiles from zerod group and 1D profiles from profil0d group.
    :param file_path: Path to the metis file saved in hdf5 file.
    :return: time, zerod, profile0d
        zerod holds 0D time-dependent profiles
        profile0D holds 1D profiles with coordinates [magnetic, temporal]
    """
    # open hdf5 file and read contents
    with h5py.File(file_path, "r") as metfile:

        # read the time traces of the 0D (zerod) datasets
        group_zerod = metfile["post/zerod"]

        zerod = {}  # all the 0D profiles will be stored in a dictionary

        for key, item in group_zerod.items():
            zerod[key] = np.array(item).squeeze()  # transform to numpy and get rid of the 0th empty dimension

        # read 1d profiles from profil0d
        group_profile0d = metfile["post/profil0d"]

        profile0d = {}

        for key, item in group_profile0d.items():
            profile0d[key] = np.array(item)

    return zerod, profile0d
