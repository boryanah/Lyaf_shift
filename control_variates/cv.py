from pathlib import Path
import gc
import sys

import numpy as np
import asdf
from scipy.fft import rfftn, irfftn, ifftn
from scipy.signal import savgol_filter
from classy import Class
from abacusnbody.metadata import get_meta
from abacusnbody.analysis.power_spectrum import (
    calc_pk_from_deltak,
    get_field_fft,
    get_k_mu_edges,
    get_delta_mu2,
    get_W_compensated,
)

from tools import bin_kmu, expand_poles_to_3d, bin_rppi, get_rp_pi_edges

#sys.path.append("/global/homes/b/boryanah/repos/abacus_tng_lyalpha/")
from tools import get_rp_pi_box_edges, compute_xirppi_from_xi3d#, get_s_mu_box_edges, compute_xismu_from_xi3d

"""
DO WE NEED TO SMOOTH THEORY AND DATA WITH KCUT?
python cv.py AbacusSummit_base_c000_ph000 z 1
python cv.py AbacusSummit_base_c000_ph001 z 1
python cv.py AbacusSummit_base_c000_ph002 z 1
python cv.py AbacusSummit_base_c000_ph003 z 1
python cv.py AbacusSummit_base_c000_ph004 z 1
python cv.py AbacusSummit_base_c000_ph005 z 1

python cv.py AbacusSummit_base_c000_ph000 z 2
python cv.py AbacusSummit_base_c000_ph001 z 2
python cv.py AbacusSummit_base_c000_ph002 z 2
python cv.py AbacusSummit_base_c000_ph003 z 2
python cv.py AbacusSummit_base_c000_ph004 z 2
python cv.py AbacusSummit_base_c000_ph005 z 2

python cv.py AbacusSummit_base_c000_ph000 z 3
python cv.py AbacusSummit_base_c000_ph001 z 3
python cv.py AbacusSummit_base_c000_ph002 z 3
python cv.py AbacusSummit_base_c000_ph003 z 3
python cv.py AbacusSummit_base_c000_ph004 z 3
python cv.py AbacusSummit_base_c000_ph005 z 3

python cv.py AbacusSummit_base_c000_ph000 z 4
python cv.py AbacusSummit_base_c000_ph001 z 4
python cv.py AbacusSummit_base_c000_ph002 z 4
python cv.py AbacusSummit_base_c000_ph003 z 4
python cv.py AbacusSummit_base_c000_ph004 z 4
python cv.py AbacusSummit_base_c000_ph005 z 4


python cv.py AbacusSummit_base_c000_ph000 y 1
python cv.py AbacusSummit_base_c000_ph001 y 1
python cv.py AbacusSummit_base_c000_ph002 y 1
python cv.py AbacusSummit_base_c000_ph003 y 1
python cv.py AbacusSummit_base_c000_ph004 y 1
python cv.py AbacusSummit_base_c000_ph005 y 1

python cv.py AbacusSummit_base_c000_ph000 y 2
python cv.py AbacusSummit_base_c000_ph001 y 2
python cv.py AbacusSummit_base_c000_ph002 y 2
python cv.py AbacusSummit_base_c000_ph003 y 2
python cv.py AbacusSummit_base_c000_ph004 y 2
python cv.py AbacusSummit_base_c000_ph005 y 2

python cv.py AbacusSummit_base_c000_ph000 y 3
python cv.py AbacusSummit_base_c000_ph001 y 3
python cv.py AbacusSummit_base_c000_ph002 y 3
python cv.py AbacusSummit_base_c000_ph003 y 3
python cv.py AbacusSummit_base_c000_ph004 y 3
python cv.py AbacusSummit_base_c000_ph005 y 3

python cv.py AbacusSummit_base_c000_ph000 y 4
python cv.py AbacusSummit_base_c000_ph001 y 4
python cv.py AbacusSummit_base_c000_ph002 y 4
python cv.py AbacusSummit_base_c000_ph003 y 4
python cv.py AbacusSummit_base_c000_ph004 y 4
python cv.py AbacusSummit_base_c000_ph005 y 4
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

def get_r_hMpc(Ndim, L_hMpc):
    """ get r bins"""
    r_hMpc = np.fft.fftfreq(Ndim)
    r_hMpc *= L_hMpc
    return r_hMpc

# sim params
z_this = 2.5
sim_name = sys.argv[1]
los_dir = sys.argv[2]
nmesh = 576*2
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

# smoothing parameters
sg_window = 21
k0 = np.min([kcut, 0.618])
dk_cv = 0.167
beta1_k = 0.05

# get bin edges
k_bin_edges, mu_bin_edges = get_k_mu_edges(Lbox, k_hMpc_max, nbins_k, nbins_mu, logk)
k_binc = (k_bin_edges[1:] + k_bin_edges[:-1]) * 0.5

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

# load saved power spectrum
data = np.load(f"/pscratch/sd/b/boryanah/AbacusLymanAlpha/control_variates/power_ell_z{z_this:.3f}_N{nmesh:d}_{sim_name}_los{los_dir}_Model_{model_no:d}.npz")
pk_tt_ell = data['pk_tt_ell']
pk_tl_ell = data['pk_tl_ell']
pk_ll_ell = data['pk_ll_ell']
pk_th_ell = data['pk_th_ell']
ct_ell = data['ct_ell']
k_avg = data['k_avg']
kth = data['kth']
bias_LYA = data['bias_LYA']
beta_LYA = data['beta_LYA']
tmp = np.zeros((3, len(k_binc)), dtype=np.float32)
tmp[0] = np.interp(k_binc, kth, pk_th_ell[0])
tmp[1] = np.interp(k_binc, kth, pk_th_ell[1])
tmp[2] = np.interp(k_binc, kth, pk_th_ell[2])
pk_th_ell = tmp

# beta parameter
beta_proj = pk_tl_ell**2/pk_ll_ell**2
beta_damp = 1 / 2 * (1 - np.tanh((k_binc - k0) / dk_cv)) * beta_proj
beta_damp = np.atleast_2d(beta_damp)
beta_damp[:, : k_binc.searchsorted(beta1_k)] = 1.0
beta_smooth = np.zeros_like(beta_damp)
for i in range(beta_smooth.shape[0]):
    beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)

# load fields
data = np.load(f"/pscratch/sd/b/boryanah/AbacusLymanAlpha/control_variates/lin_dens_z{z_this:.3f}_N{nmesh:d}_{sim_name}_los{los_dir}.npz")
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
pk3d[0, 0, 0] = 0.
del fields_fft
gc.collect()

# apply CV equation -beta*(DL - TL) # note we ain't multiplying by Lbox^3 TESTING
pk3d -= expand_poles_to_3d(k_binc, pk_th_ell/Lbox**3, kperp_hMpc, klos_hMpc, poles)
pk3d *= -expand_poles_to_3d(k_binc, beta_smooth, kperp_hMpc, klos_hMpc, np.array([0]))
"""
beta = expand_poles_to_3d(k_binc, beta_ell, kperp_hMpc, klos_hMpc, poles)
pk_th = expand_poles_to_3d(k_binc, pk_th_ell, kperp_hMpc, klos_hMpc, poles)
"""

# mock directory
save_dir = "/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/"

# load low-pass-filtered delta F in Fourier space
f = asdf.open(save_dir + sim_name + f"/z{z_this:.3f}/Model_{model_no:d}/complex_dF_Model_{model_no:d}_LOS{los_dir[-1]}.asdf")
field_fft = f['data']['ComplexDeltaFluxReal'][:] + 1j*f['data']['ComplexDeltaFluxImag'][:]
field_fft /= np.float32(Ndim_x*Ndim_y*Ndim_z)
print(f['data'].keys())
del f; gc.collect()

if los_dir == "y":
    field_fft = np.transpose(field_fft, (0, 2, 1))

# 3D power spectrum
pk3d += np.array((field_fft * np.conj(field_fft)).real, dtype=np.float32)
pk3d[0, 0, 0] = 0.
del field_fft
gc.collect()
"""
xi3d = irfftn(pk3d, workers=-1)
del pk3d
gc.collect()
print("float32", xi3d.dtype)
"""

# transform into Xi(rp,pi)
rlos_max = 200. # 200 Mpc/h
rperp_max = 200. # 200 Mpc/h
rperp_hMpc = get_r_hMpc(Ndim_x, Lbox)
rlos_hMpc = get_r_hMpc(Ndim_z, Lbox)

# get mask in transverse and los direction in real space
mask_rperp_1d = (np.abs(rperp_hMpc) < rperp_max)
mask_rlos_1d = (np.abs(rlos_hMpc) < rlos_max)

# initialize array
Xi = np.zeros((np.sum(mask_rperp_1d), np.sum(mask_rperp_1d), pk3d.shape[2]), dtype=np.float32)
    
# loop over kz direction (LOS) and perform IFFT of Pk in the kx and ky direction to get Xi(x, y, kz)
for k in range(Xi.shape[2]):
    if k%10 == 0: print(k)
    # create empty 2D array anticipating correct shape
    raw_p3d_xy = np.zeros((Ndim_x, Ndim_y), dtype=np.float32)

    # populate array such that you have zero for all the modes that have been filtered due to the low-pass filter in kx and ky direction (i.e. zero-padding)
    raw_p3d_xy[mask_perp_1d[:, np.newaxis] & mask_perp_1d[np.newaxis, :]] = pk3d[:, :, k].flatten()

    # perform IFFT in kx and ky direction and immediately cut out large real-space separations according to mask in transverse direction
    Xi[:, :, k] = (ifftn(raw_p3d_xy, workers=-1)[mask_rperp_1d[:, np.newaxis] & mask_rperp_1d[np.newaxis, :]]).reshape(np.sum(mask_rperp_1d), np.sum(mask_rperp_1d))
    del raw_p3d_xy; gc.collect()
del pk3d; gc.collect()

# create empty array anticipating Xi(x, y, kz); note that kz here does not have the filtered dimensions tuks
Xi_new = np.zeros((np.sum(mask_rperp_1d), np.sum(mask_rperp_1d), len(mask_los_1d)), dtype=np.float32)

# populate array such that you have zero for all the modes that have been filtered due to the low-pass filter in kz direction (i.e. zero-padding)
Xi_new[mask_los_1d[np.newaxis, np.newaxis, :] & np.ones_like(Xi_new).astype(bool)] = Xi.flatten()
del Xi
gc.collect()

# finally, IFFT in the kz direction and immediately cut out large real-space separations according to mask in LOS direction
Xi_new = irfftn(Xi_new, axes=(2,), workers=-1)[mask_rlos_1d[np.newaxis, np.newaxis, :] & np.ones((Xi_new.shape[0], Xi_new.shape[1], len(mask_rlos_1d)), dtype=bool)]

# note that when you apply mask even in 3D the array gets automatically flattened, so restore correct dimension
Xi_new = Xi_new.reshape(np.sum(mask_rperp_1d), np.sum(mask_rperp_1d), np.sum(mask_rlos_1d))


# define bins in los and perp direction
d = 4. # Mpc/h
n_rperp_bins = int(rperp_max/d)
n_rlos_bins = int(rlos_max/d)

"""
# attempt at faster
rperp_bin_edges, rlos_bin_edges = get_rp_pi_edges(rperp_max, rlos_max, n_rperp_bins, n_rlos_bins)
xirppi, _ = bin_rppi(Xi_new, rperp_hMpc[mask_rperp_1d], rlos_hMpc[mask_rlos_1d], rperp_bin_edges, rlos_bin_edges, dtype=np.float32)
xirppi *= Ndim_x**2*Ndim_z
"""

# get 3D separation grid
rperp_box, rlos_box, rperp_bin_edges, rlos_bin_edges = get_rp_pi_box_edges(rperp_hMpc[mask_rperp_1d], rlos_hMpc[mask_rlos_1d], rperp_max, rlos_max, n_rperp_bins, n_rlos_bins)
print(rperp_box.shape, rlos_box.shape)

# reduce memory by recasting
rperp_box = rperp_box.astype(np.float32)
rlos_box = rlos_box.astype(np.float32)
rperp_bin_edges = rperp_bin_edges.astype(np.float32)
rlos_bin_edges = rlos_bin_edges.astype(np.float32)

# do binning in real space
xirppi = compute_xirppi_from_xi3d(Xi_new, Lbox, Ndim_x, Ndim_z, rperp_box, rlos_box, rperp_bin_edges, rlos_bin_edges)

# record
if nmesh != 576:
    np.savez(f"../data_fft/Xi_rppi_LyAxLyA_LCV_{sim_name}_Model_{model_no:d}_LOS{los_dir[-1]}_d{d:.1f}_nmesh{nmesh:d}.npz", xirppi=xirppi, rp_bins=rperp_bin_edges, pi_bins=rlos_bin_edges, rlos_max=rlos_max, rperp_max=rperp_max)
else:
    np.savez(f"../data_fft/Xi_rppi_LyAxLyA_LCV_{sim_name}_Model_{model_no:d}_LOS{los_dir[-1]}_d{d:.1f}.npz", xirppi=xirppi, rp_bins=rperp_bin_edges, pi_bins=rlos_bin_edges, rlos_max=rlos_max, rperp_max=rperp_max)
