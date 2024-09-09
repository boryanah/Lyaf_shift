import time
import gc
import sys

import numpy as np
import asdf
from scipy.fft import ifftn, irfftn, fftn

"""
Usage:
python get_corr_fft.py "AbacusSummit_base_c000_ph000" 1 losz #0 # auto for Model 1
python get_corr_fft.py "AbacusSummit_base_c000_ph000" 1 losz #1 # cross for Model 1
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

# fixed parameters
z = 2.5
Lbox = 2000. # Mpc/h
Ndim_x = Ndim_y = Ndim_z = 6912

# user choices
sim_name = sys.argv[1] # "AbacusSummit_base_c000_ph000"
model = int(sys.argv[2]) # 1, 2, 3, 4
want_cross = 0 # int(sys.argv[3]) # 0 or 1 (F or T)
if want_cross:
    assert not want_qso
los_dir = sys.argv[3]#"losy" # "losy"

# mock directory
save_dir = "/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/"
save_xi_dir = "/pscratch/sd/b/boryanah/AbacusLymanAlpha/"

# load low-pass-filtered delta F in Fourier space
want_qso = 0#True
if want_qso:
    f = asdf.open(save_dir + f"/mocks/{sim_name}/z{z:.3f}/complex_QSO_LOS{los_dir[-1]}.asdf")
    field_fft = f['data']['ComplexDeltaGalaxyReal'][:] + 1j*f['data']['ComplexDeltaGalaxyImag'][:]
else:
    f = asdf.open(save_dir + sim_name + f"/z{z:.3f}/Model_{model:d}/complex_dF_Model_{model:d}_LOS{los_dir[-1]}.asdf")
    field_fft = f['data']['ComplexDeltaFluxReal'][:] + 1j*f['data']['ComplexDeltaFluxImag'][:]
print(f['data'].keys())

# tuks TESTING
if los_dir == "losy":
    field_fft = np.transpose(field_fft, (0, 2, 1)) 

# fourier space params
klos_min = f['header']['klos_min'] # 0 h/Mpc
kperp_min = f['header']['kperp_min'] # 0 h/Mpc
klos_max = f['header']['klos_max'] # 4 h/Mpc
kperp_max = f['header']['kperp_max'] # 2 h/Mpc
del f; gc.collect()

# real space params
rperp_max = 200.
rlos_max = 200.

# get the wavenumbers at each grid point
kperp_hMpc = get_k_hMpc(Ndim_x, Lbox)
klos_hMpc = get_k_hMpc_real(Ndim_x, Lbox)

# get the distances at each grid point
rperp_hMpc = get_r_hMpc(Ndim_x, Lbox)
rlos_hMpc = get_r_hMpc(Ndim_z, Lbox)

# get mask in transverse and los direction in Fourier space
mask_perp_1d = (np.abs(kperp_hMpc) < kperp_max)
mask_los_1d = (np.abs(klos_hMpc) < klos_max)
print("sum mask", np.sum(mask_perp_1d))

# get mask in transverse and los direction in real space
mask_rperp_1d = (np.abs(rperp_hMpc) < rperp_max)
mask_rlos_1d = (np.abs(rlos_hMpc) < rlos_max)

# TESTING!!!!!!!!!!
def get_kernel(k, kp, kw):
    return 0.5*(1. - np.tanh((k-kp)/kw))
w_kperp = get_kernel(kperp_hMpc[mask_perp_1d], 0.9*kperp_max, 0.1*kperp_max/2.)
w_klos = get_kernel(klos_hMpc[mask_los_1d], 0.9*klos_max, 0.1*klos_max/2.)

# if want cross-correlation with quasars, load delta QSO
if want_cross:
    f = asdf.open(save_dir + f"/mocks/{sim_name}/z{z:.3f}/complex_QSO_LOS{los_dir[-1]}.asdf")
    field2_fft = f['data']['ComplexDeltaGalaxyReal'][:] + 1j*f['data']['ComplexDeltaGalaxyImag'][:]
    print(f['data'].keys())
    del f; gc.collect()
else:
    field2_fft = None
    
# compute power spectrum (Julien can ignore this)
want_pk = False
#want_pk = True
if want_pk:
    # load old module
    sys.path.append("..")
    from compute_power import get_k_mu_box_edges, compute_pk3d_fourier_light

    # time
    t = time.time()
    
    # define bins and compute 3d power (less memory intense)
    n_k_bins = 320 # dk = 0.005
    n_mu_bins = 60
    k_box, mu_box, k_bin_edges, mu_bin_edges = get_k_mu_box_edges(kperp_hMpc[mask_perp_1d], klos_hMpc[mask_los_1d], n_k_bins, n_mu_bins, logk=False)
    p3d_hMpc = compute_pk3d_fourier_light(field_fft, Lbox, Ndim_x, Ndim_z, k_box, mu_box, k_bin_edges, mu_bin_edges, logk=False, dF2_fft=field2_fft)
    counts = np.ones_like(p3d_hMpc)
    mu = np.zeros_like(p3d_hMpc)
    k_hMpc = np.zeros_like(p3d_hMpc)

    # compute power
    mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5
    k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
    print(mu_binc.shape, k_binc.shape, mu.shape)
    for i in range(mu.shape[0]):
        mu[i, :] = mu_binc
    for i in range(k_hMpc.shape[1]):
        k_hMpc[:, i] = k_binc

    # save NPZ file
    if want_cross:
        np.savez(save_dir + f"/correlations/{sim_name}/z{z:.3f}/power3d_LyAxQSO_Model_{model:d}_LOS{los_dir[-1]}.npz", p3d_hMpc=p3d_hMpc, k_hMpc=k_hMpc, mu=mu, counts=counts)
    else:
        np.savez(save_dir + f"/correlations/{sim_name}/z{z:.3f}/power3d_LyAxLyA_Model_{model:d}_LOS{los_dir[-1]}.npz", p3d_hMpc=p3d_hMpc, k_hMpc=k_hMpc, mu=mu, counts=counts)

    # report time (super slow compared with our new package)
    print("time = ", time.time()-t)
    quit()

# normalize flux
field_fft /= (Ndim_x*Ndim_y*Ndim_z)

# normalize QSO
if want_cross:
    field2_fft /= (Ndim_x*Ndim_y*Ndim_z)

# compute 3d power
if want_cross:
    # cross-correlation
    raw_p3d = (np.conj(field_fft)*field2_fft).real
    del field2_fft; gc.collect()
else:
    # auto-correlation
    raw_p3d = np.abs(field_fft)**2
del field_fft; gc.collect()

# TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
raw_p3d[0, 0, 0] = 0.

# create empty array anticipating Xi(x, y, kz); note that x and y are expected to be masked for large separations (hence, file is smaller)
#if los_dir == "losz":
Xi = np.zeros((np.sum(mask_rperp_1d), np.sum(mask_rperp_1d), raw_p3d.shape[2]), dtype=np.float32)
assert Xi.shape[2] == len(w_klos)
    
# loop over kz direction (LOS) and perform IFFT of Pk in the kx and ky direction to get Xi(x, y, kz)
for k in range(Xi.shape[2]):
    if k%10 == 0: print(k)
    # create empty 2D array anticipating correct shape
    raw_p3d_xy = np.zeros((Ndim_x, Ndim_y), dtype=np.float32)

    # populate array such that you have zero for all the modes that have been filtered due to the low-pass filter in kx and ky direction (i.e. zero-padding)
    #raw_p3d_xy[mask_perp_1d[:, np.newaxis] & mask_perp_1d[np.newaxis, :]] = raw_p3d[:, :, k].flatten()
    # TESTING!!!!!!!!!!!!!! 
    raw_p3d_xy[mask_perp_1d[:, np.newaxis] & mask_perp_1d[np.newaxis, :]] = (raw_p3d[:, :, k]*w_kperp[:, np.newaxis]*w_kperp[np.newaxis, :]).flatten()

    # perform IFFT in kx and ky direction and immediately cut out large real-space separations according to mask in transverse direction
    #Xi[:, :, k] = (ifftn(raw_p3d_xy, workers=-1)[mask_rperp_1d[:, np.newaxis] & mask_rperp_1d[np.newaxis, :]]).reshape(np.sum(mask_rperp_1d), np.sum(mask_rperp_1d))
    # TESTING!!!!!!!!!!!!!!!
    Xi[:, :, k] = (ifftn(raw_p3d_xy, workers=-1)[mask_rperp_1d[:, np.newaxis] & mask_rperp_1d[np.newaxis, :]]).reshape(np.sum(mask_rperp_1d), np.sum(mask_rperp_1d))*w_klos[k]
    del raw_p3d_xy; gc.collect()
del raw_p3d; gc.collect()

# create empty array anticipating Xi(x, y, kz); note that kz here does not have the filtered dimensions
Xi_new = np.zeros((np.sum(mask_rperp_1d), np.sum(mask_rperp_1d), len(klos_hMpc)), dtype=np.float32)

# populate array such that you have zero for all the modes that have been filtered due to the low-pass filter in kz direction (i.e. zero-padding)
Xi_new[mask_los_1d[np.newaxis, np.newaxis, :] & np.ones_like(Xi_new).astype(bool)] = Xi.flatten()
del Xi
gc.collect()

# finally, IFFT in the kz direction and immediately cut out large real-space separations according to mask in LOS direction
Xi_new = irfftn(Xi_new, axes=(2,), workers=-1)[mask_rlos_1d[np.newaxis, np.newaxis, :] & np.ones((Xi_new.shape[0], Xi_new.shape[1], len(mask_rlos_1d)), dtype=bool)]
print(Xi_new.shape)

# note that when you apply mask even in 3D the array gets automatically flattened, so restore correct dimension
Xi_new = Xi_new.reshape(np.sum(mask_rperp_1d), np.sum(mask_rperp_1d), np.sum(mask_rlos_1d))

# save the 3D correlation function
if want_cross:
    np.savez(save_xi_dir + f"/correlations/Xi/z{z:.3f}/Xi_3D_LyAxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz", Xi=Xi_new, rperp_max=rperp_max, rlos_max=rlos_max, rperp_hMpc=rperp_hMpc[mask_rperp_1d], rlos_hMpc=rlos_hMpc[mask_rlos_1d])
else:
    if want_qso:
        np.savez(save_xi_dir + f"/correlations/Xi/z{z:.3f}/Xi_3D_QSOxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz", Xi=Xi_new, rperp_max=rperp_max, rlos_max=rlos_max, rperp_hMpc=rperp_hMpc[mask_rperp_1d], rlos_hMpc=rlos_hMpc[mask_rlos_1d])
    else:
        np.savez(save_xi_dir + f"/correlations/Xi/z{z:.3f}/Xi_3D_LyAxLyA_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz", Xi=Xi_new, rperp_max=rperp_max, rlos_max=rlos_max, rperp_hMpc=rperp_hMpc[mask_rperp_1d], rlos_hMpc=rlos_hMpc[mask_rlos_1d])
