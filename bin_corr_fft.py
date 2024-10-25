import sys, gc
sys.path.append("/global/homes/b/boryanah/repos/abacus_tng_lyalpha/")

import numpy as np
from compute_power import get_rp_pi_box_edges, compute_xirppi_from_xi3d, get_s_mu_box_edges, compute_xismu_from_xi3d

"""
python bin_corr_fft.py AbacusSummit_base_c000_ph000 1 losz
"""

# params
z = 2.5
Ndim = 6912
Lbox = 2000.
los_dir = sys.argv[3] #"losz"
sim_name = sys.argv[1]
model = int(sys.argv[2])
want_cross = 1 #int(sys.argv[3])
want_qso = 0
want_xirppi = 1#int(sys.argv[3])#False

# directory to save to
#save_dir = "/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/"
save_dir = "/pscratch/sd/b/boryanah/AbacusLymanAlpha/"

# load 3D Xi
if want_cross:
        data = np.load(save_dir + f"correlations/Xi/z{z:.3f}/Xi_3D_LyAxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz")
else:
        if want_qso:
                data = np.load(save_dir + f"correlations/Xi/z{z:.3f}/Xi_3D_QSOxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz")
        else:
                data = np.load(save_dir + f"correlations/Xi/z{z:.3f}/Xi_3D_LyAxLyA_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz")
        
Xi = data['Xi'][:, :, :]
rlos_max = data['rlos_max'] # 200 Mpc/h
rperp_max = data['rperp_max'] # 200 Mpc/h
rperp_hMpc = data['rperp_hMpc']
rlos_hMpc = data['rlos_hMpc']
del data; gc.collect()

if want_xirppi:
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
        xirppi = compute_xirppi_from_xi3d(Xi, Lbox, Ndim, Ndim, rperp_box, rlos_box, rperp_bin_edges, rlos_bin_edges)

        if want_cross:
                np.savez(save_dir + f"correlations/Xi/z{z:.3f}/Xi_rppi_LyAxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}_d{d:.1f}.npz", xirppi=xirppi, rp_bins=rperp_bin_edges, pi_bins=rlos_bin_edges, rlos_max=rlos_max, rperp_max=rperp_max)
        else:
                if want_qso:
                        np.savez(save_dir + f"correlations/Xi/z{z:.3f}/Xi_rppi_QSOxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}_d{d:.1f}.npz", xirppi=xirppi, rp_bins=rperp_bin_edges, pi_bins=rlos_bin_edges, rlos_max=rlos_max, rperp_max=rperp_max)
                else:
                        np.savez(save_dir + f"correlations/Xi/z{z:.3f}/Xi_rppi_LyAxLyA_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}_d{d:.1f}.npz", xirppi=xirppi, rp_bins=rperp_bin_edges, pi_bins=rlos_bin_edges, rlos_max=rlos_max, rperp_max=rperp_max)

else:
        # define bins in los and perp direction
        d = 2. # Mpc/h
        s_max = rperp_max
        n_s_bins = int(s_max/d) # 100
        n_mu_bins = 60
        mu_max = 1.

        # get 3D separation grid
        s_box, mu_box, s_bin_edges, mu_bin_edges = get_s_mu_box_edges(rperp_hMpc, rlos_hMpc, s_max, mu_max, n_s_bins, n_mu_bins)

        # reduce memory by recasting
        s_box = s_box.astype(np.float32)
        mu_box = mu_box.astype(np.float32)
        s_bin_edges = s_bin_edges.astype(np.float32)
        mu_bin_edges = mu_bin_edges.astype(np.float32)

        # do binning in real space
        xismu = compute_xismu_from_xi3d(Xi, Lbox, Ndim, Ndim, s_box, mu_box, s_bin_edges, mu_bin_edges)

        if want_cross:
                np.savez(save_dir + f"correlations/Xi/z{z:.3f}/Xi_smu_LyAxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz", xismu=xismu, s_bins=s_bin_edges, mu_bins=mu_bin_edges, mu_max=mu_max, s_max=s_max)
        else:
                if want_qso:
                        np.savez(save_dir + f"correlations/Xi/z{z:.3f}/Xi_smu_QSOxQSO_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz", xismu=xismu, s_bins=s_bin_edges, mu_bins=mu_bin_edges, mu_max=mu_max, s_max=s_max)
                else:
                        np.savez(save_dir + f"correlations/Xi/z{z:.3f}/Xi_smu_LyAxLyA_{sim_name}_Model_{model:d}_LOS{los_dir[-1]}.npz", xismu=xismu, s_bins=s_bin_edges, mu_bins=mu_bin_edges, mu_max=mu_max, s_max=s_max)
        
