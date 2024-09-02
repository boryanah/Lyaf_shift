`downsample.py`: Reduces 6912^3 Lyaf mock into 864^3 to enable brute-force calculation.

`fit_abacus_bao_rppi.py`: Fits BAO peak in 3 modes. `0`: jackknife on the 6 measurements for each model; `1`: jackknife on the 48 measurements for each model (note it is not exactly jackknife); `2`: averaging over fits to each of the 6 individual measurements. Lets you specify `rmin`.

`get_corr_brute.py`: Computes correlation function using `Corrfunc` with 2 Mpc/h difference in the transverse direction and 4 Mpc/h in the parallel direction.
`get_corr_brute_sub.py`: Computes correlation function in a jackknife fashion (skipping 1/8th at a time) for one of the 6 simulations and of the four models.

`plot_apat.py`: Plots `ap` and `at` as 2D scatter plots for diferent values of `rmin` and for one of the four models.
`plot_rmin_apat.py`: Plots `ap` and `at` as a function of `rmin` for the four models.

`tools.py`: Contains a function for computing the correlation function more quickly from the 3D power spectrum (works for galaxies, but breaks for Lyaf possibly due to large-scale numerical instability). Currently unused.
