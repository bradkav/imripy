import numpy as np
from imripy import merger_system as ms
from imripy import halo
from imripy import inspiral
from imripy import waveform

from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import argparse
import os

#----------------------------------------------

#Parse the arguments!                                                       
parser = argparse.ArgumentParser(description='...')
#parser.add_argument('-IDstr','--IDstr', help='ID string for saving files', type=str, required=True)
parser.add_argument('-M1', '--M1', help='IMBH mass in M_sun', type=float, default=1000.)
parser.add_argument('-M2', '--M2', help="Smaller BH mass in M_sun", type=float, default = 1.0)
parser.add_argument('-rho_sp', '--rho_sp', help='Spike density [M_sun/pc^3]', type=float, default=226.)
parser.add_argument('-gamma', '--gamma', help='slope of DM spike', type=float, default=2.333)

#parser.add_argument('-IDtag', '--IDtag', help='Optional IDtag to add on the end of the file names', type=str, default="NONE")

args = parser.parse_args()

#print (IDstr)

#---------------------------------------------

# Basic system and spike properties
g_spike = args.gamma
m1 = args.M1*ms.solar_mass_to_pc
m2 = args.M2*ms.solar_mass_to_pc

D = 5e8
sp_0 = ms.SystemProp(m1, m2, halo.ConstHalo(0.), D)

#Convert from rho_6 -> rho_spike
#k_ = ((3 - g_spike)*(0.2)**(3-g_spike))/(2 * np.pi)
#rho_6 = args.rho_6*ms.solar_mass_to_pc
#r_6 = 1e-6 #pc
#rho_spike = (rho_6*r_6**g_spike/(k_*m1)**(g_spike/3))**(3/(3-g_spike))

rho_spike = args.rho_sp*ms.solar_mass_to_pc
r_spike = ( (3 - g_spike) * m1 / (2 * np.pi * rho_spike) * 0.2**(3.-g_spike) )**(1./3.)

modelName = "Form.m3.m1.alpha2.3"


# Model the system with spike
potential = lambda r: m1/r
Eps_grid = np.geomspace(1e-13, 1e1, 1000)

spike = halo.Spike(rho_spike, r_spike, g_spike)
dh = halo.DynamicSS.FromSpike(Eps_grid, sp_0, spike)

# This is so that we can extend the grid easily for the evolution
f_grid_interp = interp1d(dh.Eps_grid, dh.f_grid, kind='cubic', fill_value=(0.,0.), bounds_error=False, copy=True)

sp = ms.SystemProp(m1, m2, dh, D)

# Model the inspiral
R0 = 50.* sp.r_isco()
R_fin = 40. * sp.r_isco()
r_grid = np.geomspace(sp.r_isco(), 50*R0, 1000)

Eps_grid = np.geomspace(1e-13, 1e1, 1000)
Eps_grid = np.sort(np.append(Eps_grid, np.geomspace(1e-1 * (sp.m1/R0 - (sp.omega_s(R0)*R0)**2 / 2.), 1e1 * sp.m1/R0, 2000)))

sp.halo.Eps_grid = Eps_grid; sp.halo.update_Eps()
sp.halo.f_grid = f_grid_interp(Eps_grid)
haloModel = inspiral.HaloFeedback(sp)
haloModel.options.verbose = 1

#------ Checks:
plot_checks = False

if (plot_checks):
    # Plot initial configuration
    fig, (ax_rho, ax_f) = plt.subplots(2, 1, figsize=(20,20))
    ax_rho.loglog(r_grid, spike.density(r_grid), label='analytic')
    ax_rho.loglog(r_grid, dh.density(r_grid), linestyle='--', label=r'recovered')
    ax_rho.axvline(R0, linestyle = '--', color='black', label = r'$R_0$')
    ax_rho.set_xlabel('r / pc')
    ax_rho.legend();
    ax_f.loglog(dh.Eps_grid, dh.f_grid, label="$f$")
    ax_f.axvline(sp.m1/R0, linestyle='-.', label='$m1/r$')
    ax_rho.grid(); ax_f.grid()
    plt.legend(); 
    plt.show()
#------------


# Evolve the system
ev = haloModel.Evolve( R0, R_fin = R_fin)

rhoeff = np.zeros(len(ev.t))
for i in tqdm(range(len(ev.t))):
    dh.f_grid = ev.f[i,:]
    j0 = np.digitize(ev.R[i], r_grid)
    jvals = np.arange(j0-5, j0+5)
    r_tmp = r_grid[jvals]
    rhovals = dh.density(r_tmp, v_max=[sp.omega_s(r)*r for r in r_tmp])
    rhoeff[i] = np.interp(ev.R[i], r_tmp, rhovals)
    
plt.figure()
plt.loglog(ev.t, ev.R)
    
plt.figure()
plt.loglog(ev.R, rhoeff)
plt.show()