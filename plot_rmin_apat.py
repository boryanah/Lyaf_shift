import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

"""
python plot_rmin_apat.py z
python plot_rmin_apat.py y
"""

rp_min = rt_min = 0.
rp_max = rt_max = 200.
r_max = 148.
N_jk = 6 *8
#stats_type = "stack_"
stats_type = "jk_"
los_dir = sys.argv[1]
#los_dir = "z"
#los_dir = "y"

r_mins = np.array([10., 20., 30., 40.])
offsets = (np.linspace(0, 1, len(r_mins))-1./2.)#*0.01

cs = ["r", "g", "b", "y"]

plt.subplots(2, 1, figsize=(14,8))
for j in range(4):
    ap_means = np.zeros_like(r_mins)
    at_means = np.zeros_like(r_mins)
    ap_errors = np.zeros_like(r_mins)
    at_errors = np.zeros_like(r_mins)

    for i, r_min in enumerate(r_mins):

        data = np.load(f"data_fits/jk_stats_Model_{j+1:d}_LOS{los_dir}_rpmin{rp_min:.1f}_rpmax{rp_max:.1f}_rtmin{rt_min:.1f}_rtmax{rt_max:.1f}_rmin{r_min:.1f}_rmax{r_max:.1f}_njk{N_jk:d}.npz", allow_pickle=True)
        print(data.files)
        #'param_mean_error_perc_sign', 'aps', 'ats', 'bias', 'beta', 'sigmap', 'sigmat'

        param_mean_error_perc_sign = data['param_mean_error_perc_sign'].item()
        aps = data['aps']
        ats = data['ats']
        ap_mean, ap_error, ap_perc, ap_sign = param_mean_error_perc_sign['aps']
        at_mean, at_error, at_perc, at_sign = param_mean_error_perc_sign['ats']

        ap_means[i] = ap_mean
        at_means[i] = at_mean
        ap_errors[i] = ap_error
        at_errors[i] = at_error

    plt.subplot(2, 1, 1)
    #plt.figure(1, figsize=(9, 7))
    plt.errorbar(r_mins-offsets[j], ap_means, yerr=ap_errors, color=cs[j], ls='', marker='o', capsize=4, label=rf"${{\rm Model}} \ {j+1:d}$")


    plt.subplot(2, 1, 2)
    #plt.figure(2, figsize=(9, 7))
    plt.errorbar(r_mins-offsets[j], at_means, yerr=at_errors, color=cs[j], ls='', capsize=4, marker='o')

plt.subplot(2, 1, 1)
plt.ylabel(r"$\alpha_\parallel$")
plt.ylim([0.985, 1.015])
plt.gca().axhline(y=1, ls='--', color='k')
plt.legend(frameon=False)

plt.subplot(2, 1, 2)
plt.ylabel(r"$\alpha_\perp$")
plt.ylim([0.985, 1.015])
plt.gca().axhline(y=1, ls='--', color='k')
plt.savefig(f"figs/{stats_type}apat_rmin_LOS{los_dir}_njk{N_jk:d}.png")
plt.show()