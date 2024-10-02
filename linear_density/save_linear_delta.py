from pathlib import Path
import gc
import sys
sys.path.append("/global/homes/b/boryanah/repos/abacus_tng_lyalpha/")

import numpy as np
import asdf
from scipy.fft import rfftn, irfftn
from classy import Class
from abacusnbody.metadata import get_meta
from compute_power import get_rp_pi_box_edges, compute_xirppi_from_xi3d, get_s_mu_box_edges, compute_xismu_from_xi3d
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

python save_linear_delta.py AbacusSummit_base_c000_ph000 x
python save_linear_delta.py AbacusSummit_base_c000_ph001 x
python save_linear_delta.py AbacusSummit_base_c000_ph002 x
python save_linear_delta.py AbacusSummit_base_c000_ph003 x
python save_linear_delta.py AbacusSummit_base_c000_ph004 x
python save_linear_delta.py AbacusSummit_base_c000_ph005 x
"""

def get_r_hMpc(Ndim, L_hMpc):
    """ get r bins"""
    r_hMpc = np.fft.fftfreq(Ndim)
    r_hMpc *= L_hMpc
    return r_hMpc

# bias and beta
bias_LYA = -0.11629442782749021
beta_LYA = 1.67

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
delta_fft = rfftn(delta, workers=-1) / np.float32(nmesh**3) #?
del delta
gc.collect()
z_this = 2.5
meta = get_meta(sim_name, redshift=z_this)
Lbox = meta['BoxSize']
z_ic = meta['InitialRedshift']
Ndim = int(meta['ppd'])
print(Ndim)
# k_Ny = np.pi*nmesh/Lbox

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


# [b (1 + beta mu^2) D delta]* [b (1 + beta mu^2) D delta] = b^2 D^2 [<delta^2> + beta <mu^2 delta^2> + beta^2 <mu^4 delta^2>]
fields_fft = {'delta': D*bias_LYA*delta_fft, 'deltamu2': D*bias_LYA*beta_LYA*get_delta_mu2(delta_fft, nmesh)}
keynames = list(fields_fft.keys())

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
#pk3d *= Lbox**3 # [cMpc/h]^3
xi3d = irfftn(pk3d, workers=-1)
        
# transform into Xi(rp,pi)
rlos_max = 200.#data['rlos_max'] # 200 Mpc/h
rperp_max = 200.#data['rperp_max'] # 200 Mpc/h
rperp_hMpc = get_r_hMpc(nmesh, Lbox)#data['rperp_hMpc']
rlos_hMpc = get_r_hMpc(nmesh, Lbox)#data['rlos_hMpc']

# define bins in los and perp direction
#d = 2. # Mpc/h
d = 4. # Mpc/h
n_rperp_bins = int(rperp_max/d)
n_rlos_bins = int(rlos_max/d)

# get 3D separation grid
rperp_box, rlos_box, rperp_bin_edges, rlos_bin_edges = get_rp_pi_box_edges(rperp_hMpc, rlos_hMpc, rperp_max, rlos_max, n_rperp_bins, n_rlos_bins)

# reduce memory by recasting
rperp_box = rperp_box.astype(np.float32)
rlos_box = rlos_box.astype(np.float32)
rperp_bin_edges = rperp_bin_edges.astype(np.float32)
rlos_bin_edges = rlos_bin_edges.astype(np.float32)

# do binning in real space
xirppi = compute_xirppi_from_xi3d(xi3d, Lbox, nmesh, nmesh, rperp_box, rlos_box, rperp_bin_edges, rlos_bin_edges)

# record
np.savez(f"data/Xi_rppi_LyAxLyA_{sim_name}_linear_density_LOS{los_dir[-1]}_d{d:.1f}.npz", xirppi=xirppi, rp_bins=rperp_bin_edges, pi_bins=rlos_bin_edges, rlos_max=rlos_max, rperp_max=rperp_max)
