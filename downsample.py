import gc
import os
import sys

import numpy as np
import asdf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def new_func(array, scale_factor=(2, 2, 2)):
    # Reshape to free dimension of size scale_factor to apply scaledown method to
    m, n, r = np.array(array.shape) // scale_factor
    arr = array.reshape((m, scale_factor[0], n, scale_factor[1], r, scale_factor[2]))
    arr = np.swapaxes(arr, 1, 2).swapaxes(2, 4)
    arr = arr.reshape((m, n, r, np.prod(scale_factor)))
    # Collapse dimensions
    arr = arr.reshape(-1,np.prod(scale_factor))
    # Get blockwise frequencies -> Get most frequent items
    arr = (arr).mean(axis=1)
    arr = arr.reshape((m,n,r))
    return arr

nmesh = 6912
npart = 144
nmesh_y = nmesh//npart
model_no = int(sys.argv[2])
if model_no != 1:
    los_dir = "z"
else:
    los_dir = "y"
sim_name = sys.argv[1]
scale_factor = 8

os.makedirs(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/{sim_name}/", exist_ok=True)
save_fn = f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/{sim_name}/dF_Model_{model_no:d}_LOS{los_dir}_part_143_down{scale_factor:d}.npy"
if os.path.exists(save_fn): quit()

for i in range(npart):
    print(i, npart)
    save_fn = f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/{sim_name}/dF_Model_{model_no:d}_LOS{los_dir}_part_{i:03d}_down{scale_factor:d}.npy"
    data = asdf.open(f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/{sim_name}/z2.500/Model_{model_no:d}/tau_Model_{model_no:d}_LOS{los_dir}_part_{i:03d}.asdf")['data']
    print(data.keys())
    
    tau_down = data[f'tau_rsd_los{los_dir}']
    print(tau_down.dtype)
    F = np.exp(-tau_down)
    mean_F = np.mean(F, dtype=np.float64)
    F /= mean_F
    F -= 1.
    F = new_func(F, scale_factor=(scale_factor, scale_factor, scale_factor))
    print(tau_down.shape, F.shape)

    np.save(save_fn, F)
    del tau_down, F, data; gc.collect()
