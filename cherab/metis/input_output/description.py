# data description obtained from the METIS  documentation. It will be neccessary to keep this up with the code.

profil0d = {"xli": 'radial normalized coordinate (Lao coordinate ~ r/a, normalized to 1 at the edge).',
         "Rsepa": 'radial coordinate of the LCMS points (m)',
         "Zsepa": 'vertical coordinate of the LCMS points (m)',
         "nbishape_el": 'NBI power deposition on electrons shape; reserved to an internal used only',
         "nbishape_ion": 'NBI power deposition on ions shape; reserved to an internal used only',
         "jnbishape": 'NBICD current density shape; reserved to an internal used only',
         "pitch": 'NBI pitch angle profile = cos(beam,B)',
         "tep": 'electron temperature  (eV)',
         "tip": 'ion temperature  (eV)',
         "jli": 'averaged current density, equivalent at jmoy in CRONOS (A/m^2)',
         "jeff": '<J.B>/B0 (A/m^2)',
         "qjli": 'safety factor',
         "jboot": ' bootstrap current density (A/m^2)',
         "eta": 'neoclassical resistivity (ohm * m)',
         "jnbicd": 'parallel current density source  due to NBICD (j = <J.B>/Bo, A/m^2)',
         "jlh": 'parallel current density source  due to LHCD (j = <J.B>/Bo, A/m^2)',
         "jeccd": 'parallel current density source  due to ECCD (j = <J.B>/Bo, A/m^2)',
         "jfwcd": 'parallel current density source  due to FWCD (j = <J.B>/Bo, A/m^2)',
         "jfus": 'fast alpha bootstrap current density (A/m^2)',
         "jrun": 'average runaway  current density (A/m^2)',
         "plh": 'power density  due to LH (W/m^3)',
         "pnbi": 'power density  due to NBI (W/m^3)',
         "pnbi_ion": 'power density  due to NBI, coupled to ions (W/m^3)',
         "pecrh": 'power density  due to ECRH (W/m^3)',
         "pfweh": 'power density  due to FWEH (W/m^3)',
         "picrh": 'power density  due to ICRH minority scheme (W/m^3)',
         "picrh_ion": 'power density  due to ICRH minority scheme, coupled to ions (W/m^3)',
         "pfus": 'power density  due to fusion reactions (W/m^3)',
         "pfus_ion": 'power density  due to fusion reactions, coupled to ions (W/m^3)',
         "pbrem": 'bremsstrahlung power  sink (W/m^3)',
         "prad": 'radiated power  sink  (W/m^3)',
         "pcyclo": 'cyclotron radiation power  source (W/m^3)',
         "pohm": 'ohmic power deposition  (W/m^3)',
         "nep": 'electron density  (m^-3)',
         "nip": 'ions density  (m^-3)',
         "vpr": 'volume element (m^3, int(vpr,x= 0..1) = plasma volume)',
         "vpr_tor": 'volume element, reserved for internal used (m^2, rhomax * int(vpr,x= 0..1) = plasma volume) ',
         "spr": 'surface element (m^2, int(spr,x= 0..1) = plasma poloidal section surface) ',
         "grho2r2": '<|gradient(rho)|^2/R^2> (see CRONOS technical document)',
         "r2i": '<1/R^2> (see CRONOS technical document)',
         "ri": '<1/R> (see CRONOS technical document)',
         "C2": 'C2 geometrical coefficient (see CRONOS technical document)',
         "C3": 'C3 geometrical coefficient (see CRONOS technical document)',
         "grho": '<|gradient(rho)|> (see CRONOS technical document)',
         "grho2": '<|gradient(rho)|^2> (see CRONOS technical document)',
         "ej": 'ohmic power density (W/m^3)',
         "kx": 'flux surface elongation ',
         "dx": 'flux surface geometrical triangularity ',
         "Raxe": ' of major radius of the centre of each flux surface (m)',
         "epsi": ' of aspect ratio (m)',
         "rmx": ' of average radius of each flux surface (m) ',
         "bpol": 'average poloidal magnetic field  (T)',
         "fdia": 'diamagnetic function  (T*m)',
         "psi": 'poloidal flux  (Wb)',
         "dpsidt": 'time derivative of poloidal flux  (V)',
         "epar": 'parallel electric field  (V/m) ',
         "zeff": 'effective charge',
         "n1p": 'density of HDT ions (m^-3)',
         "nhep": 'density of helium (m^-3)',
         "nzp": 'density of main impurity (m^-3)',
         "xieshape": 'heat transport coefficient (Ke) shape without ITB effect ',
         "xieshape_itb": 'heat transport coefficient (Ke) shape with ITB effect',
         "source_ion": 'total heat power density coupled to ions (W/m^3)',
         "source_el": 'total heat power density coupled to electrons (W/m^3)',
         "jni": 'total current density source (A/m^2)',
         "ftrap": 'effective trap fraction profile ',
         "ptot": 'total pressure profile (Pa)',
         "jfusshape": 'alpha particles bootstrap current density shape; reserved to an internal used only ',
         "salf": 'alpha particles source (m^-3)',
         "palf": 'alpha power source (W/m^-3)',
         "fprad": 'line radiative power shape; reserved to an internal used only',
         "temps": 'vectors of time associate to the s  (only time at which the s are computed are stored)',
         "xie": 'electron diffusivity estimation (m^2/s)',
         "xii": 'ion diffusivity estimation (m^2/s)',
         "n0": 'neutral density coming from edge, hot neutral (m^-3)',
         "s0": 'ionisation sources coming from edge, hot neutral (m^-3/s)',
         "n0m": 'neutral density coming from edge, cold neutral (m^-3)',
         "s0m": 'ionisation sources coming from edge, cold neutral (m^-3/s)',
         "ware": 'Ware pinch estimation (m/s)',
         "dn": 'density diffusivity estimation (m^2/s)',
         "vn": 'anormal density convection velocity estimation (m^2/s)',
         "omega": 'plasma solid rotation frequency in toroidal direction (rad/s) { sum(nk*mk*<Vk,phi * R>) /  sum(nk*mk)}',
         "vtheta": 'fluid velocity, theta component, at R = Rmax of each flux surface, for main impurity (m/s)',
         "utheta": 'neoclassical poloidal rotation speed ,for main impurity  (<V_k . theta> / <B . theta> , m/s/T)',
         "vtor": 'toroidal rotation speed (m/s), at R = Rmax of each flux surface, for  main impurity ',
         "er": 'neoclassical radial electric field (V/m) = Er / gradient(rho)',
         "spellet": 'equivalent continue source due to pellet injection (m^-3/s)',
         "ge": 'electrons flux  (m^-2/s)',
         "pioniz": 'loss power due to cold neutral ionization (W m^-3)',
         "phi": 'toroidal flux (Wb)',
         "dphidx": 'toroidal flux space derivative (Wb)',
         "nbinesource": 'electron source due to fast neutral ionisation (Number s^-1 m^-3)',
         "web": 'rotation shear (NClass definition)',
         "qe": 'electron heat flux (W)',
         "qi": 'ion heat flux (W)',
         "qei": 'electron to ion heat flux (W)',
         "rot_nbi": 'toroidal torque source from NBI (N m^-2)',
         "rot_lh": 'toroidal torque source from LHCD and ECCD (N m^-2)',
         "rot_n0": 'toroidal torque sink from edge neutral friction (N m^-2)',
         "frot": 'toroidal rotation moment flux (N m^-1)',
         "drot": 'toroidal rotation diffusion coefficient (m^2/s)',
         "vrot": 'toroidal rotation pinch coefficient (m/s)',
         "rtor": 'toroidal rotation moment density (kg m^-1 s^-1)',
         "nwp": 'density of W when tungsten effects are taking into account (m^-3)'}

zerod = {"nbar_nat": 'natural density (m^{-3})',
            "eddy_current": 'Eddy current in passive structure for breakdown description (A)',
            "flux_edge_cor": 'Poloidal flux modification of reference cons.flux due to eddy current (Wb/ (2*pi))',
            "firstorb_nbi": 'fraction of NBI power lost due to fast ion first orbit loss',
            "pin": 'total heat power (W)',
            "ploss": 'plasma losses power, as defined in ITER basis (W)',
            "zeff": 'plasma effective charge with alpha particles',
            "vp": 'plasma volume (m^3)',
            "sp": 'poloidal plasma surface (m^ 2)',
            "sext": 'external plasma surface (m^ 2)',
            "peri": 'length of the LCMS (m)',
            "ane": 'exponent of electron density profile',
            "ape": 'exponent of electron pressure profile',
            "ate": 'exponent of electron temperature profile',
            "nsat": 'saturation electron density, use for the calculation of density  (m^-3)',
            "nem": 'volume averaged electron density (m^-3)',
            "ne0": 'estimation central electron density (m^-3)',
            "nebord": 'estimation of plasma edge electron density (m^-3)',
            "negr": 'Greenwald limit for electron density (m^-3)',
            "nhem": 'volume averaged density of helium (m^-3)',
            "nimpm": 'volume averaged density of impurity (other than helium) (m^-3)',
            "n1m": 'volume averaged density of H + D + T ions (m^-3)',
            "nDm": 'volume averaged density of deuterium ions (m^-3)',
            "nTm": 'volume averaged density of tritium ions (m^-3)',
            "nim": 'sum of volume averaged  ions density (m^-3)',
            "ni0": 'estimation of central ionic density (m^-3)',
            "meff": 'effective mass (number of atomic mass, hydrogenoid ions)',
            "dwdt": 'time derivative of total plasma energy (W)',
            "esup_fus": 'D-T fusion fast alpha suprathermal energy (J)',
            "pfus_th": 'fusion thermal power depostion of alpha particles (W)',
            "taus_he": 'caracteristic slowing down time for alpha particules (s)',
            "esup_nbi": 'NBI fast ions suprathermal energy (J)',
            "pnbi_th": 'NBI thermal power depostion (W)',
            "taus_nbi": 'caracteristic slowing down time for NBI fast ions (s)',
            "esup_icrh": 'ICRH fast ions suprathermal energy (J)',
            "picrh_th": 'ICRH thermal power depostion (W)',
            "einj_icrh": 'mean energy of fast ion produce by ICRH in minoritary scheme (eV)',
            "einj_lh": 'mean LH electron energy, only define if rip = 1 (eV)',
            "taus_icrh": 'caracteristic slowing down time for ICRH fast ions (s)',
            "esup_lh": 'LH fast ions suprathermal energy (J)',
            "plh_th": 'LH thermal power depostion (W)',
            "etalh0": 'LH current drive efficiency @ vloop = 0 (A W^-1 m^-2)',
            "etalh1": 'LH current drive efficiency correction due to hot conductivity (A W^-1 m^-2)',
            "etalh": 'LH current drive efficiency (A W^-1 m^-2)',
            "pth": 'thermal power input, define as tau_E * Pth = Wth (W)',
            "pion": 'total thermal power deposition on ions, use in the calculation of  Ti/Te (W)',
            "pel": 'total thermal power deposition on electron, use in the calculation of  Ti/Te (W)',
            "wrlw": 'plasma energy contents for electron  computed with RLW sacling law (J)',
            "plossl2h": 'threshold  power for transition from  L mode to  H mode  for the selected scaling law (W)',
            "tauthl": 'confinement time of energy in L mode for the selected scaling law (s)',
            "tauh": 'confinement time of energy in H mode for the selected scaling law (s)',
            "tauhe_l": 'confinement time of helium impurity/ashes in L mode (s)',
            "tauhe_h": 'confinement time of helium impurity/ashes in H mode (s)',
            "modeh": 'confinement mode versu time:  0 = L et 1 = H',
            "taue": 'energy confinement time (s)',
            "taue_alt": 'energy confinement time, Helander & Sigmar definition (s)',
            "tauhe": 'confinement time of helium impurity/ashes (s)',
            "pw": 'effective power define as taue  * pw = W (W)',
            "w": 'total plasma energy (J)',
            "wdia": '3/2 perpendicular plasma energy (J)',
            "wth": 'thermal plasma energy (J)',
            "dwdt": 'time derivative of total plasma energy  (W)',
            "dwthdt": 'time derivative of thermal plasma energy (W)',
            "tite": 'volume averaged ratio Ti / Te',
            "te0": 'estimation of central electron temperature (eV)',
            "tebord": 'estimation of plasma  edge electron temperature (eV)',
            "tem": 'volume averaged electron temperature (eV)',
            "ilh": 'LH current drive(A)',
            "ifwcd": 'FWCD current drive(A)',
            "ifus": 'fast alpha (fusion) "bootstrap" current (A) ',
            "ieccd": 'ECCD current drive(A)',
            "inbicd": 'NBI current drive (A)',
            "ecrit_nbi": 'critical energy of NBI beam (eV)',
            "ecrit_icrh": 'critical energy of ICRH accelerated fast ions (eV)',
            "frloss_icrh": 'part of icrh power lost due to fast ions losses',
            "icd": 'total current drive (A)',
            "iboot": 'bootstrap current (A)',
            "iohm": 'ohmic current (A)',
            "pohm": 'ohmic power (W)',
            "RR": 'plasma resistor (ohm)',
            "vloop": 'loop voltage, as vloop .* iohm = pohm (V)',
            "qa": 'edge safety factor',
            "q95": 'safety factor @ 95% of poloidal flux',
            "qmin": 'estimation of minimal value of safety factor',
            "q0": 'estimation of central value of safety factor',
            "betap": 'poloidal normalized pressure of the plasma (thermal, ITER definition : betap_th(1))',
            "betaptot": 'poloidal normalized pressure of the plasma (total, ITER definition : betap(1))',
            "ini": 'total non inductive current (A)',
            "ip": 'plasma current (A)',
            "wbp": 'poloidal field energy of the plasma (J)',
            "dwbpdt": 'time derivative of plasma poloidal field energy (W)',
            "pfus": 'total fusion power of alpha (W)',
            "pfus_nbi": 'NBI induce D-T fusion piower of alpha (W)',
            "salpha": 'total number of alpha fusion particules from D-T ractions  per second (s^-1)',
            "ecrit_he": 'alpha critical energy (eV)',
            "prad": 'impurity radition losses in core plasma, without Bremsstrahlung (W)',
            "pradsol": 'radiation losses in the SOL (W)',
            "pbrem": 'Bremsstrahlung radition losses (W)',
            "pcyclo": 'cyclotron radiation losses (W)',
            "nb": 'number of convergence loops made',
            "dw": 'relative error on w, at the end of convergence',
            "dpfus": 'relative error on pfus, at the end of convergence',
            "dini": 'relative error on Ini, at the end of convergence',
            "diboot": 'relative error on Iboot, at the end of convergence',
            "temps": 'time slices vector',
            "nbar": 'line averaged electron density (m^-3)',
            "picrh": 'ICRH power, decrease of ripple losses in TS (W)',
            "plh": 'LH power , decrease of ripple losses in TS (W)',
            "pnbi": 'NBI power (W)',
            "pecrh": 'ECRH power(W)',
            "priptherm": 'TS ripple losses, thermal part (W)',
            "tauee": 'scale of electron heat confinement time (s)',
            "tauii": 'scale of ion heat confinement time (s)',
            "tauei": 'exchange electon/ion heat time (s)',
            "hitb": 'H factor of ITB',
            "xitb": 'radius of itb due to a revserse shear (estimation,su)',
            "aitb": 'effect of itb on temperature peaking factor',
            "hmhd": 'H factor limitation due to MHD (BetaN limit)',
            "betan": 'normalized total beta of the plasma',
            "disrup": 'if = 1, ratiative disruption flag',
            "pei": 'equipartition power (W)',
            "asser": 'if 1, feedback on vloop is on',
            "tauj": 'diffusion time of current (s)',
            "tauip": 'carateristic time of  R-L equivalent plasma circuit (s)',
            "li": 'internal self inductance (formule 3 of ITER FDR)',
            "zeffsc": 'zeff scaling (private data, do not use)',
            "pfus_loss": 'fusion power loss due to first orbit losses of fast alpha (estimation, W)',
            "d0": 'Shafranov shift (m)',
            "pped": 'pressure at the top of piedestal (Pa)',
            "xnbi": 'position of the maximum power deposition of NBI(su)',
            "piqnbi": 'peaking factor of the NBI power deposition profile  (su)',
            "xeccd": 'position of the maximum power deposition of ECCD(su)',
            "xlh": 'position of the maximum power deposition of LH  deposition maximum position (su)',
            "dlh": 'width of the power deposition of LH (su)',
            "wrad": 'estimation of bulk volume averaged toroidal rotation velocity (rad/s)',
            "xfus": 'Jalpha  maximum position (su)',
            "jxfus": 'Jalpha at xfus (su)',
            "j0fus": 'Jalpha at center (usage interne,su)',
            "pel_icrh": 'ICRH power going on electron (W)',
            "pion_icrh": 'ICRH power going on ions (W)',
            "pel_nbi": 'NBI power going on electron (W)',
            "pion_nbi": 'NBI power going on ions (W)',
            "pel_fus": 'Alpha power going on electron (W)',
            "pion_fus": 'Alpha power going on ions (W)',
            "frnbi": 'fraction of NBI power absorded in the plasma',
            "telim": 'plasma eletron temparature estimation near the divertor plate (eV)',
            "nelim": 'plasma eletron density temparature estimation near the divertor plate (m^-3)',
            "plim": 'total power estimation on  divertor plate (W)',
            "peakdiv": 'divertor peak power surface density estimation  (W/m^2)',
            "stf": 'number of wrong data in zs data structure containing NaN or Imag',
            "fwcorr": 'internal data to compute tem',
            "plhthr": 'additionnal power crossing the LCMS; must be compare to  L->H threshold power (Ryter PPCF 2002)',
            "qeff": 'effective safety factor at the edge (computed with ITER formula)',
            "zmszl": 'ratio between volume averaged zeff and line averaged zeff ',
            "plhrip": 'LH power loss in ripple for Tore Supra (W)',
            "n0a": 'number of cold neutral that input in plasma at the edge every second coming from recycling and gaz puff (s^-1)',
            "rres": 'ICRF resonance layer radial position (m)',
            "xres": 'ICRF resonance layer normalized radius position',
            "rm": 'sqrt(PHI/pi/B0) of LCMS  (m) ',
            "drmdt": 'd  sqrt(PHI/pi/B0) / dt  of LCMS(m/s) ',
            "ndd": 'total DD neutrons per second (s^-1)',
            "ndd_th": 'plasma/plasma DD neutrons per second (s^-1)',
            "ndd_nbi_th": 'beam/plasma DD neutrons per second (s^-1)',
            "ndd_nbi_nbi": 'beam/beam DD neutrons  per second (s^-1)',
            "pddfus": 'fusion power from DD reactions (W)',
            "pttfus": 'fusion power from TT reactions (W)',
            "mu0_nbi": 'initial value of pitch angle for NBI (V// / V'')',
            "efficiency": 'Fisch like LH current drive efficiency for LHCD(A/W/m^2)',
            "fracmino": 'minority ions fraction accelerated by ICRH',
            "pelec": 'reactor electric power provide to the network (W)',
            "pped": 'total pressure @ pedestal top (Pa)',
            "ppedmax": 'maximal allowed total pressure @ pedestal top (Pa)',
            "piqj": 'peaking factor of current profile',
            "vmes": 'loop voltage as measured on a fixed magnetic loop (V)',
            "pioniz": 'power losses due to cold neutral ionization (W)',
            "irun": 'runaway current (A)',
            "difcurconv": 'current diffusion solver state',
            "ipar": 'plasma current // B (A)',
            "xpoint": 'flag for diverted plasma;for plasma in limiter mode = 0; for plamsa in divertor mode = 1',
            "harm": 'ICRH minority scheme harmonic',
            "nmino": 'volume averaged ICRH minority density',
            "einj_nbi_icrh": 'beam energy including ICRH effects (eV; reserved for internal used only)',
            "pnbi_icrh": 'equivalent beam power increase needs for neutron rate enhancement due to  ICRH synergy effects (W)',
            "ialign": 'current drive alignment quality parameter (1 = good , 0 = bad)',
            "frac_pellet": 'fraction of fuelling due to pellet',
            "dwmagtordt": 'time derivative of toroidal magnetic plasma energy  (W)',
            "wmagtor": 'toroidal magnetic plasma energy  (J)',
            "phiplasma": 'toroidal magnetic flux due to  the plasam (Wb)',
            "tibord": 'edge ion temperature (eV)',
            "nibord": 'edge ion density (m^-3)',
            "xped": 'pedestal normalized position',
            "teped": 'top electron pedestal temperature (eV)',
            "tiped": 'top ion pedestal temperature (eV)',
            "neped": 'top electron pedestal density (m^-3)',
            "niped": 'top ion pedestal density (m^-3)',
            "edgeflux": 'poloidal flux at the edge of plasma (V s)',
            "indice_inv": 'sawtooth invert radius indice',
            "sn0fr": 'friction source on edge neutral that damps the toroidal rotation (N m)',
            "snbi": 'rotation torque due to NBI (N m)',
            "wrot": 'toroidal plasma rotation stored energy (J)',
            "slh": 'rotation torque due to LHCD and ECCD (N m)',
            "taup": 'confinement time of matter (s)',
            "poynting": 'flux of Poynting vector through the LCMS (W)',
            "nbar_nat": 'natural density (m^{-3})',
            "ane_actual": 'actual exponent of electron density profile obtained taking into account the reference and constraints from transport',
            "nwm": 'volume averaged tungsten density (m^-3)',
            "dsol": 'caracteristic SOL width (lambda_q,  m)',
            "eddy_current": 'Eddy current in passive structure for breakdown description (A)',
            "flux_edge_cor": 'Poloidal flux modification of reference cons.flux due to eddy current (Wb/ (2*pi))',
            "firstorb_nbi": 'fraction of NBI power lost due to fast ion first orbit loss'}
