from cherab.metis import read_hdf5
import matplotlib.pyplot as plt


#replace with path to a metis file saved in hdf5 file
file = "/path/to/metis/file.mat"

#read 0d and 1d profiles
zerod, profile0d = read_hdf5(file)


fig_timetrace, ax = plt.subplots()
ax.plot(zerod["temps"], zerod["ip"])
ax.set_xlabel("time [s]")
ax.set_ylabel("Iplasma [A]")
fig_timetrace.tight_layout()

timeslice = 100
fig_profile, ax = plt.subplots()
ax.plot(profile0d["psi"][:, timeslice], profile0d["prad"][:, timeslice])
ax.set_xlabel("psi [Wb]")
ax.set_ylabel("radiated power [Wm$^{-3}]$")
ax.set_title("Radiated power profile t = {0:2.3f} s".format(zerod["temps"][timeslice]))
fig_profile.tight_layout()