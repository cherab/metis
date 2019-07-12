from cherab.metis import METISModel
from cherab.metis.utils import generate_metis_example
import numpy as np

import matplotlib.pyplot as plt


#load metis example
metis = generate_metis_example()

#iterpolation of 0D signals in time
time_axis = np.linspace(metis.time.min(), metis.time.max(), 1000)

#interpolate_zerod returns array of interpolated values for specified quantity and times
ip_interpolated = metis.interpolate_zerod("ip", time_axis)
p_nbi_interpolated = metis.interpolate_zerod("pnbi", time_axis)

#interpolated plasma current
asdfx, ax = plt.subplots()
ax.plot(time_axis, ip_interpolated, "-")
ax.plot(metis.time, metis.zerod("ip"), "x")

#interpolation has limits, beware of high derivatives and induced oscillations in interpolated signal
asdfx, ax = plt.subplots()
ax.plot(time_axis, p_nbi_interpolated, "-")
ax.plot(metis.time, metis.zerod("pnbi"), "x")


#inerpolation of 1D profiles
time_slice1 = 0.76
time_slice2 = 0.77
time_slice = 0.765


#profile1d returns the closest time slice of a specified quantity
psi_slice1 = metis.profile1d_nearest("psi", time_slice1)
psi_slice2 = metis.profile1d_nearest("psi", time_slice1)

psin_slice1 = metis.profile1d_nearest("psin", time_slice1)
psin_slice2 = metis.profile1d_nearest("psin", time_slice2)

te_slice1 = metis.profile1d_nearest("tep", time_slice1)
te_slice2 = metis.profile1d_nearest("tep", time_slice2)

#profile1d interpolates 1D profile of requested quantity onto a specified "free variable" for a given time
psi_slice = metis.profile1d_interpolate("psi", time_slice)
psi_slice = np.linspace(psi_slice.min(), psi_slice.max(), 100)

psin_slice = np.linspace(0, 1, 30)

#free variable is determined by the parameter name and passed values
te_slice_psi = metis.profile1d_interpolate("tep", time_slice, psi=psi_slice)
te_slice_psin = metis.profile1d_interpolate("tep", time_slice, psin=psin_slice)

plot_tep_psi = plt.subplots()
ax = plot_tep_psi[1]
ax.plot(psi_slice1, te_slice1, "x", label="time 1, metis")
ax.plot(psi_slice2, te_slice2, "x", label="time 2, metis")
ax.plot(psi_slice, te_slice_psi, label="time, interpolated")
ax.legend()
ax.set_xlabel("psi [wb]")
ax.set_ylabel("te [ev]")

plot_tep_psin = plt.subplots()
ax = plot_tep_psin[1]
ax.plot(psin_slice1, te_slice1, "x", label="time 1, metis")
ax.plot(psin_slice2, te_slice2, "x", label="time 2, metis")
ax.plot(psin_slice, te_slice_psin, label="time, interpolated")
ax.legend()
ax.set_xlabel("psi")
ax.set_ylabel("te [ev]")

