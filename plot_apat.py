import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

"""
python plot_apat.py 10. 1 z
python plot_apat.py 20. 1 z
python plot_apat.py 30. 1 z
python plot_apat.py 40. 1 z

python plot_apat.py 10. 2 z
python plot_apat.py 20. 2 z
python plot_apat.py 30. 2 z
python plot_apat.py 40. 2 z

python plot_apat.py 10. 3 z
python plot_apat.py 20. 3 z
python plot_apat.py 30. 3 z
python plot_apat.py 40. 3 z

python plot_apat.py 10. 4 z
python plot_apat.py 20. 4 z
python plot_apat.py 30. 4 z
python plot_apat.py 40. 4 z

python plot_apat.py 10. 1 y
python plot_apat.py 20. 1 y
python plot_apat.py 30. 1 y
python plot_apat.py 40. 1 y

python plot_apat.py 10. 2 y
python plot_apat.py 20. 2 y
python plot_apat.py 30. 2 y
python plot_apat.py 40. 2 y

python plot_apat.py 10. 3 y
python plot_apat.py 20. 3 y
python plot_apat.py 30. 3 y
python plot_apat.py 40. 3 y

python plot_apat.py 10. 4 y
python plot_apat.py 20. 4 y
python plot_apat.py 30. 4 y
python plot_apat.py 40. 4 y

"""

rp_min = rt_min = 0.
rp_max = rt_max = 200.
r_max = 148.
N_jk = 6*8
#stats_type = "stack_"
stats_type = "jk_"

r_min = float(sys.argv[1])# 10.
model_no = int(sys.argv[2]) #1
los_dir = (sys.argv[3]) #1
#los_dir = "z"
#los_dir = "y"

data = np.load(f"data_fits/jk_stats_Model_{model_no:d}_LOS{los_dir}_rpmin{rp_min:.1f}_rpmax{rp_max:.1f}_rtmin{rt_min:.1f}_rtmax{rt_max:.1f}_rmin{r_min:.1f}_rmax{r_max:.1f}_njk{N_jk:d}.npz", allow_pickle=True)
print(data.files)
#'param_mean_error_perc_sign', 'aps', 'ats', 'bias', 'beta', 'sigmap', 'sigmat'

param_mean_error_perc_sign = data['param_mean_error_perc_sign'].item()
aps = data['aps']
ats = data['ats']
ap_mean, ap_error, ap_perc, ap_sign = param_mean_error_perc_sign['aps']
at_mean, at_error, at_perc, at_sign = param_mean_error_perc_sign['ats']

# THERE IS SOME KIND OF ISSUE WITH THE LOS_DIR = Y MODEL 4 COULD WE HAVE MESSED UP THE CALCULATION
mask = (aps < 1.5) & (aps > 0.5) & (ats < 1.5) & (ats > 0.5)
if "jk_" == stats_type:
    N = np.sum(mask)
    print(N)
    ap_mean = np.mean(aps[mask])
    at_mean = np.mean(ats[mask])
    ap_error = np.sqrt(N-1.)*np.std(aps[mask], ddof=0)
    at_error = np.sqrt(N-1.)*np.std(ats[mask], ddof=0)

plt.figure(figsize=(9, 7))
plt.scatter(aps, ats)
plt.errorbar([ap_mean], [at_mean], xerr=[ap_error], yerr=[at_error], color='k')
plt.xlabel(r"$\alpha_\parallel$")
plt.ylabel(r"$\alpha_\perp$")
plt.xlim([0.985, 1.015])
plt.ylim([0.985, 1.015])
plt.gca().axhline(y=1, ls='--', color='k')
plt.gca().axvline(x=1, ls='--', color='k')
plt.savefig(f"figs/{stats_type}apat_Model_{model_no:d}_LOS{los_dir}_rmin{r_min:.1f}_njk{N_jk:d}.png")
