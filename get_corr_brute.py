import sys
import time

from Corrfunc.theory import DDsmu, DDrppi
import numpy as np

def get_corr(x1, y1, z1, w1, rpbins, nbins_mu, lbox, Nthread, num_cells = 20, x2 = None, y2 = None, z2 = None, w2=None, corr_type="rppi"):
    if corr_type == "smu": print("need to fix what this function returns"); quit()
    
    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1
    
    # single precision mode
    # to do: make this native 
    rpbins = rpbins.astype(np.float32)
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    w1 = w1.astype(np.float32)
    lbox = np.float32(lbox)
    
    #nbins_mu = 40
    if autocorr == 1:
        if corr_type == "rppi":
            npibins = int(rpbins[-1]/(rpbins[1]-rpbins[0]))
            results = DDrppi(autocorr, Nthread, rpbins, rpbins[-1], npibins, x1, y1, z1, weights1=w1, weight_type='pair_product', periodic = True, boxsize = lbox)#, max_cells_per_dim = num_cells)
        elif corr_type == "smu":
            results = DDsmu(autocorr, Nthread, rpbins, 1, nbins_mu, x1, y1, z1, weights1=w1, weight_type='pair_product', periodic = True, boxsize = lbox)#, max_cells_per_dim = num_cells)
        DD_counts = results['weightavg']#['npairs']
        npairs = results['npairs']
        rpavg = results['rpavg']
        rmax = results['rmax']
        pimax = results['pimax']
    else:
        x2 = x2.astype(np.float32)
        y2 = y2.astype(np.float32)
        z2 = z2.astype(np.float32)
        results = DDsmu(autocorr, Nthread, rpbins, 1, nbins_mu, x1, y1, z1, weights1=w1, X2 = x2, Y2 = y2, Z2 = z2, weights2=w2,
            periodic = True, boxsize = lbox)#, max_cells_per_dim = num_cells)
        DD_counts = results['weightavg']#['npairs']
        npairs = results['npairs']
        
    if corr_type == "rppi":
        DD_counts = DD_counts.reshape((len(rpbins) - 1, npibins))
        npairs = npairs.reshape((len(rpbins) - 1, npibins))
        rpavg = rpavg.reshape((len(rpbins) - 1, npibins))
        rmax = rmax.reshape((len(rpbins) - 1, npibins))
        pimax = pimax.reshape((len(rpbins) - 1, npibins))
    elif corr_type == "smu":
        DD_counts = DD_counts.reshape((len(rpbins) - 1, nbins_mu))
        
    #mu_bins = np.linspace(0, 1, nbins_mu+1)
    #RR_counts = 2*np.pi/3*(rpbins[1:, None]**3 - rpbins[:-1, None]**3)*(mu_bins[None, 1:] - mu_bins[None, :-1]) / lbox**3 * ND1 * ND2 * 2
    #xi_s_mu = DD_counts/RR_counts - 1
    
    return DD_counts, npairs, rpavg, rmax, pimax

# specify directory
sim_name = sys.argv[1] #"AbacusSummit_base_c000_ph000"
lbox = 2000.
scale_factor = 8 #*2
npart = 144
model_no = int(sys.argv[2])
if model_no != 1:
    los_dir = "z"
else:
    los_dir = "y"
ngrid = 6912//scale_factor
cell_size = lbox/ngrid
print(sim_name, model_no, los_dir)
sys.stdout.flush()

# midpoint of each cell along LOS
rsd_bins = np.linspace(0., ngrid, ngrid+1)
rsd_binc = (rsd_bins[1:] + rsd_bins[:-1]) * .5
rsd_binc *= cell_size # cMpc/h
#rsd_binc = (np.arange(ngrid)+0.5)*cell_size

# rsdbins
x1, y1, z1 = np.meshgrid(rsd_binc, rsd_binc, rsd_binc)
    
# load deltaF
#fn = f"tmp_deltaF_yz_losz_ngrid{ngrid:d}.npy"
dF = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
count = 0
for i in range(npart):
    print(i)

    tau = np.load(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/{sim_name}/dF_Model_{model_no:d}_LOS{los_dir}_part_{i:03d}_down{scale_factor:d}.npy")
    dF[count:count+tau.shape[0], :, :] = tau
    
    count += tau.shape[0]
assert count == ngrid, "oopsie daisy"
sys.stdout.flush()

#dF = np.load(fn).astype(np.float32)
print("mean!", np.mean(dF))

# downsample
if los_dir == "y":
    dF = np.transpose(dF, (0, 2, 1)) # wait, no, losz?

print(dF.dtype)

# transpose so that rsd along third axis
x1, y1, z1 = x1.flatten(), y1.flatten(), z1.flatten()
w1 = dF.flatten()

# plot params
rpbins = np.linspace(0, 148, 75)
print(rpbins[-1], len(rpbins)-1)
rpbinc = (rpbins[1:] + rpbins[:-1])*.5
nbins_mu = 1 #4
Nthread = 256 # max for perlmutter
corr_type = "rppi" # smu
t = time.time()
sys.stdout.flush()

save_fn = f"data/autocorr_{corr_type}_dF_{sim_name}_Model_{model_no:d}_LOS{los_dir}_part_{i:03d}_down{scale_factor:d}_try4.npz"

print("starting corrfunc")
xi_s_mu, npairs, rpavg, rmax, pimax = get_corr(x1, y1, z1, w1, rpbins, nbins_mu, lbox, Nthread, num_cells=20, x2=None, y2=None, z2=None, w2=None, corr_type=corr_type)
    
np.savez(save_fn, xi_s_mu=xi_s_mu, rpbinc=rpbinc, npairs=npairs, rpavg=rpavg, rmax=rmax, pimax=pimax)
print(time.time()-t)
