from cherab.metis import METISModel

import numpy as np

import matplotlib.pyplot as plt


def doubleparabola(r, Centre, Edge, p, q):
    return (Centre - Edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) + Edge

def generate_metis_example():
    """
    Generates an example of a METISModel instance whith a few made up plasma properties.
    :return:
    """
    metis = METISModel()

    time_shape = 400
    time = np.linspace(0, 2, time_shape)
    metis._zerod_data["temps"] =time

    ipmax = 2e6
    rampup_len = 100
    rampdown_len = 100

    ip = np.ones_like(time) *  ipmax
    ip[0:rampup_len] = np.linspace(0, ipmax, rampup_len)
    ip[-1*rampdown_len::] = np.linspace( ipmax, 0, rampdown_len)

    paux_start = 150
    paux_stop = 250

    pohm_max = 1e6
    paux = 5e6

    pohm = ip / ip.max() * pohm_max #ohmic power

    pnbi= np.ones_like(ip)

    pnbi[paux_start:paux_stop] = paux



    metis._zerod_data["ip"] = ip
    metis._zerod_data["pohm"] = pohm
    metis._zerod_data["pnbi"] = pnbi

    #### profil0d
    profil0d_shape = 21

    xli = np.linspace(0, 1, profil0d_shape, endpoint=True)[:, np.newaxis]
    metis._profile0d_data["xli"] = xli

    psi_max = np.ones_like(ip)
    psi_max[0:120] = np.linspace(0, 0.4, 120, endpoint=False)
    psi_max[120:350] = np.linspace(0.4, 0.6, 230, endpoint=False)
    psi_max[350::] = np.linspace(0.6, 0.65, 50)

    psi_min = np.ones_like(ip)
    psi_min[0:150] = np.linspace(-0.01, 0.15, 150, endpoint=False)
    psi_min[150:350] = np.linspace(0.15, 0.3, 200, endpoint=False)
    psi_min[350::] = np.linspace(0.3, 0.6, 50)

    psi = np.linspace(psi_min, psi_max, profil0d_shape, endpoint=True)
    metis._profile0d_data["psi"] = psi
    psin = np.divide((psi - psi.min(axis=0)), psi.max(axis=0) - psi.min(axis=0))
    metis._profile0d_data["psin"] = psin

    te_max = np.zeros_like(psi_max)
    te_max[0:rampup_len] = np.logspace(2, np.log10(2e3), rampup_len)
    te_max[rampup_len:paux_start] = 2e3
    te_max[paux_start:paux_stop - 50] = np.logspace(np.log10(2e3), np.log10(3.5e3), paux_stop - 50 -  paux_start)
    te_max[paux_stop - 50: paux_stop] = 3.5e3
    te_max[paux_stop:-1*rampdown_len] = np.logspace(np.log10(3.5e3), np.log10(2e3), time_shape - paux_stop - rampdown_len)
    te_max[-1*rampdown_len::] = np.logspace(np.log10(2e3), np.log10(3e2), rampdown_len)

    te_min = np.zeros_like(psi_max)
    te_min[0:rampup_len] = np.logspace(np.log10(10), np.log10(250), rampup_len)
    te_min[rampup_len:paux_start] = 250
    te_min[paux_start:paux_stop - 50] = np.logspace(np.log10(250),np.log10(300), paux_stop - 50 -  paux_start)
    te_min[paux_stop - 50: paux_stop] = 350
    te_min[paux_stop:-1*rampdown_len] = np.logspace(np.log10(350), np.log10(250), time_shape - paux_stop - rampdown_len)
    te_min[-1*rampdown_len::] = np.logspace(np.log10(250), np.log10(30), rampdown_len)

    te_p = np.zeros_like(psi_max)
    te_p[0: rampup_len] = np.linspace(1, 2, rampup_len, endpoint=False)
    te_p[rampup_len:paux_start] = 2
    te_p[paux_start: paux_stop - 50] = np.logspace(np.log10(2),np.log10(4), paux_stop - 50 -  paux_start)

    te_p = np.zeros_like(psi_max)
    te_p[0:rampup_len] = np.logspace(np.log10(1), np.log10(2), rampup_len)
    te_p[rampup_len:paux_start] = 2
    te_p[paux_start:paux_stop - 50] = np.logspace(np.log10(2),np.log10(4), paux_stop - 50 -  paux_start)
    te_p[paux_stop - 50: paux_stop] = 4
    te_p[paux_stop:-1*rampdown_len] = np.logspace(np.log10(4), np.log10(2), time_shape - paux_stop - rampdown_len)
    te_p[-1*rampdown_len::] = np.logspace(np.log10(2), np.log10(1), rampdown_len)

    te_q = np.ones_like(psi_max) * 2
    te_q[0:rampup_len] = np.logspace(np.log10(2), np.log10(2), rampup_len)
    te_q[rampup_len:paux_start] = 2
    te_q[paux_start:paux_stop - 50] = np.logspace(np.log10(2),np.log10(3), paux_stop - 50 -  paux_start)
    te_q[paux_stop - 50: paux_stop] = 3
    te_q[paux_stop:-1*rampdown_len] = np.logspace(np.log10(3), np.log10(2), time_shape - paux_stop - rampdown_len)
    te_q[-1*rampdown_len::] = np.logspace(np.log10(2), np.log10(2), rampdown_len)

    tep = np.zeros_like(psi)
    for index, args in enumerate(zip(te_max, te_min, te_p, te_q)):
        tep[:, index] = doubleparabola(psin[:,index], *args)

    metis._profile0d_data["tep"] = tep

    return metis
