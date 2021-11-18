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

parser.add_argument('-ID', '--ID', help='Optional ID to add on the end of the file names', type=str, default="NONE")
parser.add_argument('-outdir', '--outdir', type=str, default="")
parser.add_argument('-verbose', '--verbose', type=int, default=1)

parser.add_argument('-ri', '--ri', help ='Initial separation of the binary, in units of r_isco', type=float, default=50.0)
parser.add_argument('-rf', '--rf', help ='Final separation of the binary, in units of r_isco', type=float, default=1.0)

args = parser.parse_args()

IDstr = args.ID

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

modelName = IDstr




# Model the system with spike
potential = lambda r: m1/r
Eps_grid = np.geomspace(1e-13, 1e1, 1000)

spike = halo.Spike(rho_spike, r_spike, g_spike)
dh = halo.DynamicSS.FromSpike(Eps_grid, sp_0, spike)

# This is so that we can extend the grid easily for the evolution
f_grid_interp = interp1d(dh.Eps_grid, dh.f_grid, kind='cubic', fill_value=(0.,0.), bounds_error=False, copy=True)

sp = ms.SystemProp(m1, m2, dh, D)

# Model the inspiral
R0 = args.ri* sp.r_isco()
R_fin = args.rf * sp.r_isco()
r_grid = np.geomspace(sp.r_isco(), 50*R0, 10000)

Eps_grid = np.geomspace(1e-13, 1e1, 1000)
Eps_grid = np.sort(np.append(Eps_grid, np.geomspace(1e-1 * (sp.m1/R0 - (sp.omega_s(R0)*R0)**2 / 2.), 1e1 * sp.m1/R0, 2000)))

sp.halo.Eps_grid = Eps_grid; sp.halo.update_Eps()
sp.halo.f_grid = f_grid_interp(Eps_grid)
haloModel = inspiral.HaloFeedback(sp, options=inspiral.HaloFeedback.EvolutionOptions(accuracy=1e-6))
haloModel.options.verbose = args.verbose

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

print("r_i [pc]:", R0)
print("r_f [pc]:", R_fin)
print("f_i [Hz]:", (sp.omega_s(R0)/np.pi)*ms.s_to_pc)


# Evolve the system
#ev = haloModel.Evolve( R0, R_fin = R_fin)
ev = haloModel.Evolve_HFK( R0, R_fin = R_fin, dt_Torb=1e3, adjust_stepsize=True)

rhoeff = np.zeros(len(ev.t))
for i in range(len(ev.t)):
    dh.f_grid = ev.f[i,:]
    j0 = np.digitize(ev.R[i], r_grid)
    jvals = np.arange(max(0,j0-5), min(j0+5,len(r_grid)))
    r_tmp = r_grid[jvals]
    rhovals = dh.density(r_tmp, v_max=[sp.omega_s(r)*r for r in r_tmp])
    rhoeff[i] = np.interp(ev.R[i], r_tmp, rhovals)/ms.solar_mass_to_pc
    
f_GW = (sp.omega_s(ev.R)/np.pi)*ms.s_to_pc
    
htxt = f'M1 = {args.M1}; M2 = {args.M2}; rho_sp = {args.rho_sp}; gamma = {args.gamma}'
htxt += '\nColumns: t [s], r [pc], f_GW [Hz], rho_eff (< v_orb) [Msun/pc^3]'
np.savetxt(args.outdir + "Trajectory_" + IDstr + ".txt", list(zip(ev.t/ms.s_to_pc, ev.R, f_GW, rhoeff)), header=htxt)

fig, ax = plt.subplots(ncols=2, nrows=1,figsize=(10, 5))
ax[0].semilogy(ev.t/ms.s_to_pc, ev.R)
ax[0].set_xlabel(r"$t$ [s]")
ax[0].set_ylabel(r"$R$ [pc]")
    
ax[1].loglog(ev.R, rhoeff)
ax[1].set_xlabel(r"$R$ [pc]")
ax[1].set_ylabel(r"$\rho_{\mathrm{eff}, v < v_\mathrm{orb}}(R)$ [$M_\odot\,\mathrm{pc}^{-3}$]")

plt.tight_layout()

plt.savefig(args.outdir + "Evolution_" + IDstr + ".pdf", bbox_inches='tight')
#plt.show()
