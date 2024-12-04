from pathlib import Path
import gc
import sys

import numpy as np
import asdf
from scipy.fft import rfftn, irfftn
from classy import Class
from abacusnbody.metadata import get_meta
from abacusnbody.analysis.power_spectrum import (
    calc_pk_from_deltak,
    get_field_fft,
    get_k_mu_edges,
    get_delta_mu2,
    get_W_compensated,
)

from tools import bin_kmu, expand_poles_to_3d

"""
python get_power_ell.py AbacusSummit_base_c000_ph000 z 1
python get_power_ell.py AbacusSummit_base_c000_ph001 z 1
python get_power_ell.py AbacusSummit_base_c000_ph002 z 1
python get_power_ell.py AbacusSummit_base_c000_ph003 z 1
python get_power_ell.py AbacusSummit_base_c000_ph004 z 1
python get_power_ell.py AbacusSummit_base_c000_ph005 z 1

python get_power_ell.py AbacusSummit_base_c000_ph000 z 2
python get_power_ell.py AbacusSummit_base_c000_ph001 z 2
python get_power_ell.py AbacusSummit_base_c000_ph002 z 2
python get_power_ell.py AbacusSummit_base_c000_ph003 z 2
python get_power_ell.py AbacusSummit_base_c000_ph004 z 2
python get_power_ell.py AbacusSummit_base_c000_ph005 z 2

python get_power_ell.py AbacusSummit_base_c000_ph000 z 3
python get_power_ell.py AbacusSummit_base_c000_ph001 z 3
python get_power_ell.py AbacusSummit_base_c000_ph002 z 3
python get_power_ell.py AbacusSummit_base_c000_ph003 z 3
python get_power_ell.py AbacusSummit_base_c000_ph004 z 3
python get_power_ell.py AbacusSummit_base_c000_ph005 z 3

python get_power_ell.py AbacusSummit_base_c000_ph000 z 4
python get_power_ell.py AbacusSummit_base_c000_ph001 z 4
python get_power_ell.py AbacusSummit_base_c000_ph002 z 4
python get_power_ell.py AbacusSummit_base_c000_ph003 z 4
python get_power_ell.py AbacusSummit_base_c000_ph004 z 4
python get_power_ell.py AbacusSummit_base_c000_ph005 z 4


python get_power_ell.py AbacusSummit_base_c000_ph000 y 1
python get_power_ell.py AbacusSummit_base_c000_ph001 y 1
python get_power_ell.py AbacusSummit_base_c000_ph002 y 1
python get_power_ell.py AbacusSummit_base_c000_ph003 y 1
python get_power_ell.py AbacusSummit_base_c000_ph004 y 1
python get_power_ell.py AbacusSummit_base_c000_ph005 y 1

python get_power_ell.py AbacusSummit_base_c000_ph000 y 2
python get_power_ell.py AbacusSummit_base_c000_ph001 y 2
python get_power_ell.py AbacusSummit_base_c000_ph002 y 2
python get_power_ell.py AbacusSummit_base_c000_ph003 y 2
python get_power_ell.py AbacusSummit_base_c000_ph004 y 2
python get_power_ell.py AbacusSummit_base_c000_ph005 y 2

python get_power_ell.py AbacusSummit_base_c000_ph000 y 3
python get_power_ell.py AbacusSummit_base_c000_ph001 y 3
python get_power_ell.py AbacusSummit_base_c000_ph002 y 3
python get_power_ell.py AbacusSummit_base_c000_ph003 y 3
python get_power_ell.py AbacusSummit_base_c000_ph004 y 3
python get_power_ell.py AbacusSummit_base_c000_ph005 y 3

python get_power_ell.py AbacusSummit_base_c000_ph000 y 4
python get_power_ell.py AbacusSummit_base_c000_ph001 y 4
python get_power_ell.py AbacusSummit_base_c000_ph002 y 4
python get_power_ell.py AbacusSummit_base_c000_ph003 y 4
python get_power_ell.py AbacusSummit_base_c000_ph004 y 4
python get_power_ell.py AbacusSummit_base_c000_ph005 y 4
"""


def get_k_hMpc(Ndim, L_hMpc):
    """ get frequencies in x and y direction"""
    k_hMpc = np.fft.fftfreq(Ndim)
    # normalize using box size
    k_hMpc *= (2.0*np.pi) * Ndim / L_hMpc
    return k_hMpc

def get_k_hMpc_real(Ndim, L_hMpc):
    """ get frequencies in z direction"""
    k_hMpc = np.fft.rfftfreq(Ndim)
    # normalize using box size
    k_hMpc *= (2.0*np.pi) * Ndim / L_hMpc
    return k_hMpc


def get_poles(k, pk, bias, beta, poles=[0, 2, 4]):
    """
    Compute the len(poles) multipoles given the linear power spectrum, pk, the growth function,
    the growth factor and the bias.
    """
    p_ell = np.zeros((len(poles), len(k)))
    for i, pole in enumerate(poles):
        if pole == 0:
            p_ell[i] = (1.0 + 2.0 / 3.0 * beta + 1.0 / 5 * beta**2) * pk
        elif pole == 2:
            p_ell[i] = (4.0 / 3.0 * beta + 4.0 / 7 * beta**2) * pk
        elif pole == 4:
            p_ell[i] = (8.0 / 35 * beta**2) * pk
    p_ell *= bias**2
    return k, p_ell


# sim params
sim_name = sys.argv[1]
los_dir = sys.argv[2]
nmesh = 576*2
z_this = 2.5
Lbox = 2000.
model_no = int(sys.argv[3])
ph = int(sim_name.split("_ph")[-1])

# power params
k_Ny = np.pi*nmesh/Lbox
nbins_k = nmesh//2
nbins_mu = 1
poles = np.array([0, 2, 4])
k_hMpc_max = k_Ny
kcut = k_Ny/2.
logk = False

# dimensions of mocks
Ndim_x = Ndim_y = Ndim_z = 6912

# fourier space params
klos_max = 4. # h/Mpc
kperp_max = 2. # h/Mpc

# get the wavenumbers at each grid point
kperp_hMpc = get_k_hMpc(Ndim_x, Lbox)
klos_hMpc = get_k_hMpc_real(Ndim_x, Lbox)

# get mask in transverse and los direction in Fourier space
mask_perp_1d = (np.abs(kperp_hMpc) < kperp_max)
mask_los_1d = (np.abs(klos_hMpc) < klos_max)

# wavenumbers of mocks
kperp_hMpc = kperp_hMpc[mask_perp_1d]
klos_hMpc = klos_hMpc[mask_los_1d]

# define k, mu bins
k_bin_edges, mu_bin_edges = get_k_mu_edges(Lbox, k_hMpc_max, nbins_k, nbins_mu, logk)
k_binc = (k_bin_edges[1:] + k_bin_edges[:-1]) * 0.5
mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1]) * 0.5


# "z" and then "y"
rmin = 30. # Mpc/h
data = np.load(f"../data_fits/stacked_stats_bb_Model_{model_no:d}_LOSzy_rpmin0.0_rpmax200.0_rtmin0.0_rtmax200.0_rmin{rmin:.1f}_rmax180.0_njk12_fft.npz")
bias = data['bias']
beta = data['beta']
if los_dir == "z":
    bias_LYA = bias[ph]
    beta_LYA = beta[ph]
elif los_dir == "y":
    bias_LYA = bias[ph+6]
    beta_LYA = beta[ph+6]

# load the linear density field
data = np.load(f"/pscratch/sd/b/boryanah/AbacusLymanAlpha/control_variates/lin_dens_z{z_this:.3f}_N{nmesh:d}_{sim_name}_los{los_dir}.npz")
p_m_lin = data['p_m_lin']
kth = data['kth']
_, pk_th_ell = get_poles(kth, p_m_lin, bias_LYA, beta_LYA, poles=poles)

# [b (1 + beta mu^2) D delta]* [b (1 + beta mu^2) D delta] = b^2 D^2 [<delta^2> + beta <mu^2 delta^2> + beta^2 <mu^4 delta^2>]
fields_fft = {'delta': bias_LYA*data['delta_fft_pad'], 'deltamu2': bias_LYA*beta_LYA*data['deltamu2_fft_pad']}
keynames = list(fields_fft.keys())
del data; gc.collect()

# auto power spectrum of the linear field
count = 0
for i in range(len(keynames)):
    for j in range(len(keynames)):
        print('Computing cross-correlation of', keynames[i], keynames[j])

        # compute
        if count == 0:
            pk3d = np.array(
                (fields_fft[keynames[i]] * np.conj(fields_fft[keynames[j]])).real,
                dtype=np.float32,
            )
        else:
            pk3d += np.array(
                (fields_fft[keynames[i]] * np.conj(fields_fft[keynames[j]])).real,
                dtype=np.float32,
            )
        count += 1
pk3d *= Lbox**3 # [cMpc/h]^3
pk3d[0, 0, 0] = 0.
_, _, pk_ll_ell, ct_ell, k_avg = bin_kmu(pk3d, kperp_hMpc, klos_hMpc, k_bin_edges, mu_bin_edges, poles=poles)
del pk3d
gc.collect()

# mock directory
save_dir = "/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/"

# load low-pass-filtered delta F in Fourier space
f = asdf.open(save_dir + sim_name + f"/z{z_this:.3f}/Model_{model_no:d}/complex_dF_Model_{model_no:d}_LOS{los_dir[-1]}.asdf")
field_fft = f['data']['ComplexDeltaFluxReal'][:] + 1j*f['data']['ComplexDeltaFluxImag'][:]
field_fft /= np.float32(Ndim_x*Ndim_y*Ndim_z) 
print(f['data'].keys())
del f
gc.collect()

if los_dir == "y":
    field_fft = np.transpose(field_fft, (0, 2, 1))

count = 0
for i in range(len(keynames)):
    print('Computing cross-correlation of', keynames[i], "Lyaf")

    # compute
    if count == 0:
        pk3d = np.array(
            (fields_fft[keynames[i]] * np.conj(field_fft)).real,
            dtype=np.float32,
        )
    else:
        pk3d += np.array(
            (fields_fft[keynames[i]] * np.conj(field_fft)).real,
            dtype=np.float32,
        )
    count += 1
pk3d *= Lbox**3 # [cMpc/h]^3
pk3d[0, 0, 0] = 0.
del fields_fft
gc.collect()
_, _, pk_tl_ell, ct_ell, k_avg = bin_kmu(pk3d, kperp_hMpc, klos_hMpc, k_bin_edges, mu_bin_edges, poles=poles)
del pk3d
gc.collect()

pk3d = np.array((field_fft * np.conj(field_fft)).real, dtype=np.float32)
pk3d *= Lbox**3 # [cMpc/h]^3
pk3d[0, 0, 0] = 0.
_, _, pk_tt_ell, ct_ell, k_avg = bin_kmu(pk3d, kperp_hMpc, klos_hMpc, k_bin_edges, mu_bin_edges, poles=poles)
del pk3d
gc.collect()

np.savez(f"/pscratch/sd/b/boryanah/AbacusLymanAlpha/control_variates/power_ell_z{z_this:.3f}_N{nmesh:d}_{sim_name}_los{los_dir}_Model_{model_no:d}.npz", pk_tt_ell=pk_tt_ell, pk_tl_ell=pk_tl_ell, pk_ll_ell=pk_ll_ell, ct_ell=ct_ell, k_avg=k_avg, kth=kth, pk_th_ell=pk_th_ell, bias_LYA=bias_LYA, beta_LYA=beta_LYA)
