from pathlib import Path
import gc
import sys
sys.path.append("/global/homes/b/boryanah/repos/abacus_tng_lyalpha/")

import numpy as np
import asdf
from scipy.fft import rfftn, irfftn
from classy import Class
from abacusnbody.metadata import get_meta
#from compute_power import get_rp_pi_box_edges, compute_xirppi_from_xi3d, get_s_mu_box_edges, compute_xismu_from_xi3d
from abacusnbody.analysis.power_spectrum import get_delta_mu2

"""
python save_linear_delta.py AbacusSummit_base_c000_ph000 z
python save_linear_delta.py AbacusSummit_base_c000_ph001 z
python save_linear_delta.py AbacusSummit_base_c000_ph002 z
python save_linear_delta.py AbacusSummit_base_c000_ph003 z
python save_linear_delta.py AbacusSummit_base_c000_ph004 z
python save_linear_delta.py AbacusSummit_base_c000_ph005 z

python save_linear_delta.py AbacusSummit_base_c000_ph000 y
python save_linear_delta.py AbacusSummit_base_c000_ph001 y
python save_linear_delta.py AbacusSummit_base_c000_ph002 y
python save_linear_delta.py AbacusSummit_base_c000_ph003 y
python save_linear_delta.py AbacusSummit_base_c000_ph004 y
python save_linear_delta.py AbacusSummit_base_c000_ph005 y
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

# load density field
nmesh = 576*2
ic_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/ic/"
sim_name = sys.argv[1]#"AbacusSummit_base_c000_ph000"
ic_fn = Path(ic_dir) / sim_name / f'ic_dens_N{nmesh:d}.asdf'
f = asdf.open(ic_fn)
print(f['data'].keys())
delta = f['data']['density'][:, :, :]
print('mean delta', np.mean(delta))
los_dir = sys.argv[2]
if los_dir == "y":
    delta = np.transpose(delta, (0, 2, 1))
elif los_dir == "x":
    delta = np.transpose(delta, (2, 1, 0))

# do fourier transform
delta_fft = rfftn(delta, workers=-1) / np.float32(nmesh**3)
del delta
gc.collect()
z_this = 2.5
meta = get_meta(sim_name, redshift=z_this)
Lbox = meta['BoxSize']
z_ic = meta['InitialRedshift']
Ndim = int(meta['ppd'])

# set up cosmology
boltz = Class()
cosmo = {}
cosmo['output'] = 'mPk mTk'
cosmo['P_k_max_h/Mpc'] = 20.0
int(sim_name.split('ph')[-1])
for k in (
        'H0',
        'omega_b',
        'omega_cdm',
        'omega_ncdm',
        'N_ncdm',
        'N_ur',
        'n_s',
        'A_s',
        'alpha_s',
        #'wa', 'w0',
):
    cosmo[k] = meta[k]
boltz.set(cosmo)
boltz.compute()

D = boltz.scale_independent_growth_factor(z_this)
D /= boltz.scale_independent_growth_factor(z_ic)
f_growth = boltz.scale_independent_growth_factor_f(z_this)

# load input linear power
kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']
D_ratio = meta['GrowthTable'][z_ic] / meta['GrowthTable'][1.0]
p_m_lin = D_ratio**2 * pk_z1 # now it's at z_IC
p_m_lin *= D**2  # now at z_this

# apply gaussian cutoff to linear power
#p_m_lin *= np.exp(-((kth / kcut) ** 2))

# get the mu2 field
deltamu2_fft = get_delta_mu2(delta_fft, nmesh)

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
print("sum mask", np.sum(mask_perp_1d))
#[mask_perp_1d[:, np.newaxis] & mask_perp_1d[np.newaxis, :]]

# wavenumbers of mocks
kperp_hMpc = kperp_hMpc[mask_perp_1d]
klos_hMpc = klos_hMpc[mask_los_1d]

# initialize the final size of arrays which will be padded
delta_fft_pad = np.zeros((np.sum(mask_perp_1d), np.sum(mask_perp_1d), np.sum(mask_los_1d)), dtype=np.complex64)
deltamu2_fft_pad = np.zeros((np.sum(mask_perp_1d), np.sum(mask_perp_1d), np.sum(mask_los_1d)), dtype=np.complex64)

# mask to go from mocks to linear density
k_Ny_ic = np.pi*nmesh/Lbox
mask_perp_1d = ((kperp_hMpc < k_Ny_ic - 1.e-6) & (kperp_hMpc > - k_Ny_ic - 1.e-6))
mask_los_1d = (klos_hMpc < k_Ny_ic + 1.e-6)
print(np.sum(mask_perp_1d), np.sum(mask_los_1d), delta_fft.shape, delta_fft_pad.shape)

# apply cuts
delta_fft_pad[mask_perp_1d[:, np.newaxis, np.newaxis] & mask_perp_1d[np.newaxis, :, np.newaxis] & mask_los_1d[np.newaxis, np.newaxis, :]] = delta_fft.flatten()
deltamu2_fft_pad[mask_perp_1d[:, np.newaxis, np.newaxis] & mask_perp_1d[np.newaxis, :, np.newaxis] & mask_los_1d[np.newaxis, np.newaxis, :]] = deltamu2_fft.flatten()

# [b (1 + beta mu^2) D delta]* [b (1 + beta mu^2) D delta] = b^2 D^2 [<delta^2> + beta <mu^2 delta^2> + beta^2 <mu^4 delta^2>]
#fields_fft = {'delta': D*delta_fft_pad, 'deltamu2': D*deltamu2_fft_pad}
np.savez(f"/pscratch/sd/b/boryanah/AbacusLymanAlpha/control_variates/lin_dens_z{z_this:.3f}_N{nmesh:d}_{sim_name}_los{los_dir}.npz", delta_fft_pad=D*delta_fft_pad, deltamu2_fft_pad=D*deltamu2_fft_pad, p_m_lin=p_m_lin, kth=kth, D_growth=D, f_growth=f_growth)
