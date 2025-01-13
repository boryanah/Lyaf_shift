#!/usr/bin/env python
import sys

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.io.fits as pyfits
from vega import VegaInterface
from vega.coordinates import Coordinates
from vega.minimizer import Minimizer

"""
TODO: verify, save, share, pair summation; other LOS; SIGMARPRT; EFFECT OF RPMIN

data = np.load(fn)
xi_s_mu = data['xi_s_mu']

data_special = np.load("data/autocorr_rppi_dF_AbacusSummit_base_c000_ph004_Model_4_LOSz_part_143_down8.npz")
npairs = data_special['npairs'] # might need to load from new runs

xi_s_mu = 0.5*(xi_s_mu[:, :npibins//2][:, ::-1] + xi_s_mu[:, npibins//2:])
npairs = 1.0*(npairs[:, :npibins//2][:, ::-1] + npairs[:, npibins//2:])

xi_s_mu = (xi_s_mu*npairs).reshape(nrpbins//2, 2, npibins//2, 1).sum(axis=(1, 3))
npairs = npairs.reshape(nrpbins//2, 2, npibins//2, 1).sum(axis=(1, 3))
xi_s_mu /= npairs

"""

# initialize Vega
want_lcv = True
if want_lcv:
    lcv_str = "_LCV"
    lcv_extra_str = "_nmesh1152"
    #lcv_extra_str = ""
else:
    lcv_str = ""
    lcv_extra_str = ""
want_bb = True
if want_bb:
    bb_str = "_bb"
else:    
    bb_str = ""
want_qso = False
if want_qso:
    #qso_str = "_lyalya_lyaqso"
    qso_str = "_lyaqso"
else:
    qso_str = ""
    vega = VegaInterface(f'configs/main{bb_str}.ini')
vega = VegaInterface(f'configs/main{bb_str}{qso_str}.ini') # makes no difference ask andrei re arinyo maybe different initializing of parameters

# do you want linear density field?
want_linear = True
if want_linear:
    linear_str = "_linear_density"
    #linear_str = "_EFT"
else:
    linear_str = ""

# redefine coordinate grid
#rpbins = np.linspace(0, 148, 38)
rpbins = np.linspace(0, 200, 51)
rpbinc = (rpbins[1:] + rpbins[:-1])*.5
nrpbins = npibins = int(rpbins[-1]/(rpbins[1]-rpbins[0]))*2 # TODO: make prettier
rp_grid = rt_grid = rpbinc

want_fft = True
if want_fft:
    fft_str = "_fft"
else:
    fft_str = ""

# redshift grid
z_eff = 2.5
z_grid = np.ones_like(rp_grid) * z_eff

# define coordinates
new_coords = Coordinates(rp_min=rpbins[0], rp_max=rpbins[-1], rt_max=rpbins[-1], rp_nbins=len(rpbinc), rt_nbins=len(rpbinc), z_eff=z_eff)
if want_qso:
    vega.corr_items['lyaxqso'].init_coordinates(new_coords)
else:
    vega.corr_items['lyaxlya'].init_coordinates(new_coords)

# create mask
rp = np.tile(rpbinc, (rpbinc.shape[0], 1)).T
rt = rp.T
r = np.sqrt(rp**2+rt**2)
"""
# andrei said don't
rp_min = 9.
rp_max = 149.
rt_min = 9.
rt_max = 149.
"""
rp_min = 0.
rp_max = 200.
rt_min = 0.
rt_max = 200.
rmin = float(sys.argv[3]) # 10, 20, 30, 40 # 0 only if turning on arinyo
#rmax = 148.
rmax = 180.
#rmax = 200.
mask = (rp.flatten() > rp_min) & (rp.flatten() < rp_max) & (rt.flatten() > rt_min) & (rt.flatten() < rt_max) & (r.flatten() > rmin) & (r.flatten() < rmax)
mask_2d = mask[:, None] & mask[None, :]

# read and mask covariance
#h = pyfits.open("/global/cfs/cdirs/desicollab/users/jguy/pk2xi/cf_lya_x_lya_desi_y1.fits")
h = pyfits.open("/global/cfs/projectdirs/desi/science/lya/eboss_dr16/public_dr16_correlations/cf_z_0_10-exp.fits")
h["COR"].data["RP"] = np.tile(2+4*np.arange(50),(50,1)).T.ravel()
h["COR"].data["RT"] = np.tile(2+4*np.arange(50),(50,1)).ravel() # unnecessary
R = np.sqrt(h["COR"].data["RP"]**2+h["COR"].data["RT"]**2)
mask_data = (h["COR"].data["RP"] > rp_min) & (h["COR"].data["RP"] < rp_max) & (h["COR"].data["RT"] > rt_min) & (h["COR"].data["RT"] < rt_max) & (R > rmin) & (R < rmax)
mask_2d_data = mask_data[:, None] & mask_data[None, :]
cov = h["COR"].data["CO"] / 20. # matrice de covariance de DESI Y1 CHECK
cov /= 12.  # the original stuff in the paper has this line uncommented TESTING
cov = cov[mask_2d_data].reshape(np.sum(mask_data), np.sum(mask_data))
icov = np.linalg.inv(cov)

# load qso covariance matrix
h = pyfits.open("/global/cfs/cdirs/desicollab/users/jguy/pk2xi/eboss-covariance/eboss-dr16-xcf-2500x2500-covariance.fits")
cov = h[0].data / 20.
#cov /= 12. # in the paper we commented this out; TESTING # I think this is what makes the computation incredibly slow cause it means you need a super high precision
cov = cov[mask_2d_data].reshape(np.sum(mask_data), np.sum(mask_data))
icov_qso = np.linalg.inv(cov)

# Build model for monopole
def monopole_model(params):
    ell = 0
    vega.corr_items['lyaxlya'].config['model']['single_multipole'] = str(ell)
    return vega.compute_model(params, run_init=True)['lyaxlya']

model_no = int(sys.argv[1]) # 1, 2, 3, 4 
N_sims = 6
want_jump = int(sys.argv[2])
if want_jump == 1:
    N_jk = 6*8
elif want_jump == 0:
    N_jk = 6
elif want_jump == 2:
    N_jk = 6
los_dir = sys.argv[4] #"y", "z"
if los_dir == "zy" or los_dir == "yz":
    want_both_los = True
else:
    want_both_los = False
if want_both_los:
    N_jk *= 2
aps = np.zeros(N_jk) 
ats = np.zeros(N_jk)
bias = np.zeros(N_jk) 
beta = np.zeros(N_jk)
sigmap = np.zeros(N_jk)
sigmat = np.zeros(N_jk) 
scale_factor = 8
corr_type = "rppi"
run_init = True

for i_jk in range(N_jk):
    print(i_jk)
    
    count = 0
    for i in range(N_jk):
        match = i == i_jk
        if want_jump != 2: # not jackknife
            if match: continue

        if want_both_los:
            if i >= N_jk // 2:
                i_dir = 1
                los_dir = "y"
            else:
                i_dir = 0
                los_dir = "z"
            i = i - i_dir*(N_jk // 2)
            
        if want_jump == 1:
            i_sim = i // 8
            ijk_grid = i - i_sim * 8
            i_grid, j_grid, k_grid = f'{ijk_grid:03b}'
        elif want_jump == 0:
            i_sim = i
            ijk_grid = i_grid = j_grid = k_grid = None
        elif want_jump == 2:
            i_sim = i
            ijk_grid = i_grid = j_grid = k_grid = None
            
        sim_name = f"AbacusSummit_base_c000_ph{i_sim:03d}"
                    
        if ijk_grid is not None:
            fn = f"data_subs/autocorr_{corr_type}_dF_{sim_name}_Model_{model_no:d}_LOS{los_dir}_part_144_down{scale_factor:d}_i{i_grid}_j{j_grid}_k{k_grid}_jump2.npz"
        else:
            fn = f"data/autocorr_{corr_type}_dF_{sim_name}_Model_{model_no:d}_LOS{los_dir}_part_143_down{scale_factor:d}.npz"
        
        # process
        if want_fft:
            if want_linear:
                data = np.load(f"linear_density/data/Xi_rppi_LyAxLyA{lcv_str}_AbacusSummit_base_c000_ph{i_sim:03d}{linear_str}_Model_{model_no:d}_LOS{los_dir[-1]}_d4.0{lcv_extra_str}.npz")
            else:
                data = np.load(f"data_fft/Xi_rppi_LyAxLyA{lcv_str}_AbacusSummit_base_c000_ph{i_sim:03d}_Model_{model_no:d}_LOS{los_dir[-1]}_d4.0{lcv_extra_str}.npz")

            rp_bins = data['rp_bins']
            pi_bins = data['pi_bins']
            xirppi = data['xirppi']
            
            rp_binc = (rp_bins[1:]+rp_bins[:-1])*.5
            pi_binc = (pi_bins[1:]+pi_bins[:-1])*.5

            choice = (rp_binc < rpbins[-1])[:, None] & (pi_binc < rpbins[-1])[None, :]
            xirppi = xirppi[choice].reshape(np.sum(rp_binc < rpbins[-1]), np.sum(pi_binc < rpbins[-1]))
            xirppi = (xirppi.T)
            xi = xirppi.flatten()

            if want_qso:
                data = np.load(f"data_fft/Xi_rppi_LyAxQSO{lcv_str}_AbacusSummit_base_c000_ph{i_sim:03d}_Model_{model_no:d}_LOS{los_dir[-1]}_d4.0{lcv_extra_str}.npz")
                xirppi = data['xirppi']

                xirppi = xirppi[choice].reshape(np.sum(rp_binc < rpbins[-1]), np.sum(pi_binc < rpbins[-1]))
                xirppi = (xirppi.T)
                xi_qso = xirppi.flatten()
        else:
            data = np.load(fn)
            xi_s_mu = data['xi_s_mu']
            if want_jump == 1:
                data_special = np.load("data_subs/autocorr_rppi_dF_AbacusSummit_base_c000_ph004_Model_4_LOSz_part_144_down8_i1_j1_k1_jump2.npz")
            else:
                data_special = np.load("data/autocorr_rppi_dF_AbacusSummit_base_c000_ph004_Model_4_LOSz_part_143_down8.npz")
            npairs = data_special['npairs'] # might need to load from new runs
            xi_s_mu = (xi_s_mu*npairs)[:, :npibins//2][:, ::-1] + (xi_s_mu*npairs)[:, npibins//2:]
            npairs = npairs[:, :npibins//2][:, ::-1] + npairs[:, npibins//2:]
            xi_s_mu = 0.5*(xi_s_mu[:, :npibins//2][:, ::-1] + xi_s_mu[:, npibins//2:])
            npairs = 1.0*(npairs[:, :npibins//2][:, ::-1] + npairs[:, npibins//2:])
            #xi_s_mu = xi_s_mu.reshape(nrpbins//2, 2, npibins//2, 1).mean(axis=(1, 3))
            xi_s_mu = (xi_s_mu*npairs).reshape(nrpbins//2, 2, npibins//2, 1).sum(axis=(1, 3))
            npairs = npairs.reshape(nrpbins//2, 2, npibins//2, 1).sum(axis=(1, 3))
            xi_s_mu /= npairs
            xi_s_mu = xi_s_mu.T
            xi = xi_s_mu.flatten()
        xi = xi.reshape(len(xi), 1)
        if want_qso:
            xi_qso = xi_qso.reshape(len(xi), 1)
            
        if count == 0:
            xi_final = xi
        else:
            xi_final = np.hstack((xi_final, xi))
        if want_jump == 2 and match:
            xi_this = xi.flatten().copy()

        if want_qso:
            if count == 0:
                xi_qso_final = xi_qso
            else:
                xi_qso_final = np.hstack((xi_qso_final, xi_qso))
            if want_jump == 2 and match:
                xi_qso_this = xi_qso.flatten().copy()
        count += 1

    if want_jump != 2:
        print("should be ", N_jk-1, count)
    else:
        print("should be ", N_jk, count)
    
    """
    cov = np.cov(xi_final)
    cov /= count
    icov = np.diag(1./np.diag(cov))
    """
    
    # average out
    xi_mean = np.mean(xi_final, axis=1)
    if want_jump != 2:
        data = xi_mean[mask]
    else:
        data = xi_this[mask]
        
    if want_qso:
        xi_qso_mean = np.mean(xi_qso_final, axis=1)
        if want_jump != 2:
            data_qso = xi_qso_mean[mask]
        else:
            data_qso = xi_qso_this[mask]
        
    def chisq(params):
        #model = monopole_model(params)
        model = vega.compute_model(params, run_init=run_init)['lyaxlya']
        diff = model[mask] - data
        chisq = diff @ icov @ diff.T
        return chisq

    def chisq_qso(params):
        """
        #model = monopole_model(params)
        model = vega.compute_model(params, run_init=run_init)['lyaxlya']
        diff = model[mask] - data
        chisq = diff @ icov @ diff.T
        """
        model = vega.compute_model(params, run_init=run_init)['lyaxqso']
        diff = model[mask] - data_qso
        chisq = diff @ icov_qso @ diff.T
        return chisq
    
    # initialize and run iminuit
    if want_qso:
        minimizer = Minimizer(chisq_qso, vega.sample_params)
    else:
        minimizer = Minimizer(chisq, vega.sample_params)
    minimizer.minimize()
    run_init = False # to speed up
    #quit() # run with want_jump == 2
    
    # gaussian uncertainties
    #print("errors", minimizer.errors)
    aps[i_jk] = minimizer.values['ap']
    ats[i_jk] = minimizer.values['at']
    bias[i_jk] = minimizer.values['bias_LYA']
    beta[i_jk] = minimizer.values['beta_LYA']
    sigmap[i_jk] = minimizer.values['sigmaNL_par']
    sigmat[i_jk] = minimizer.values['sigmaNL_per']

# bestfit
#print("bestfit", minimizer.values)
if want_jump == 2: # not jackknife
    data = xi_mean[mask]
    if want_qso:
        data_qso = xi_qso_mean[mask]
        minimizer = Minimizer(chisq_qso, vega.sample_params)
    else:
        minimizer = Minimizer(chisq, vega.sample_params)
    minimizer.minimize()

def jk_stats(par_name, pars):
    mean_par = np.mean(pars)
    error_par = np.sqrt(N_jk-1)*np.std(pars, ddof=0)
    shift_perc = (1.-mean_par)*100.
    shift_sign = (1.-mean_par)/error_par
    print("par mean", mean_par)
    print("par error", error_par)
    print("percentage shift", shift_perc)
    print("significance w/ which we measure shift", shift_sign)
    return mean_par, error_par, shift_perc, shift_sign

def avg_stats(par_name, pars):
    mean_par = np.mean(pars)
    error_par = np.std(pars)#, ddof=0)
    shift_perc = (1.-mean_par)*100.
    shift_sign = (1.-mean_par)/error_par
    print("par mean", mean_par)
    print("par error", error_par)
    print("percentage shift", shift_perc)
    print("significance w/ which we measure shift", shift_sign)
    return mean_par, error_par, shift_perc, shift_sign

def stacked_stats(par_name, pars):
    mean_par = minimizer.values[par_name]
    error_par = minimizer.errors[par_name]
    shift_perc = (1.-mean_par)*100.
    shift_sign = (1.-mean_par)/error_par
    print("par mean", mean_par)
    print("par error", error_par)
    print("percentage shift", shift_perc)
    print("significance w/ which we measure shift", shift_sign)
    return mean_par, error_par, shift_perc, shift_sign

if want_jump == 2:
    #stats = avg_stats
    #stats_type = "avg_"
    stats = stacked_stats
    stats_type = "stacked_"
else:
    stats = jk_stats
    stats_type = "jk_"

pars_dict = {}
pars_dict['aps'] = np.array(stats('ap', aps))
pars_dict['ats'] = np.array(stats('at', ats))
pars_dict['bias'] = np.array(stats('bias_LYA', bias))
pars_dict['beta'] = np.array(stats('beta_LYA', beta))
pars_dict['sigmap'] = np.array(stats('sigmaNL_par', sigmap))
pars_dict['sigmat'] = np.array(stats('sigmaNL_per', sigmat))
if want_both_los:
    np.savez(f"data_fits/{stats_type}stats{bb_str}{lcv_str}{qso_str}_Model_{model_no:d}_LOSzy_rpmin{rp_min:.1f}_rpmax{rp_max:.1f}_rtmin{rt_min:.1f}_rtmax{rt_max:.1f}_rmin{rmin:.1f}_rmax{rmax:.1f}_njk{N_jk:d}{fft_str}{linear_str}.npz", param_mean_error_perc_sign=pars_dict, aps=aps, ats=ats, bias=bias, beta=beta, sigmap=sigmap, sigmat=sigmat)
else:
    np.savez(f"data_fits/{stats_type}stats{bb_str}{lcv_str}{qso_str}_Model_{model_no:d}_LOS{los_dir}_rpmin{rp_min:.1f}_rpmax{rp_max:.1f}_rtmin{rt_min:.1f}_rtmax{rt_max:.1f}_rmin{rmin:.1f}_rmax{rmax:.1f}_njk{N_jk:d}{fft_str}{linear_str}.npz", param_mean_error_perc_sign=pars_dict, aps=aps, ats=ats, bias=bias, beta=beta, sigmap=sigmap, sigmat=sigmat)
