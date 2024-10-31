import sys, gc, os
sys.path.append("/global/homes/b/boryanah/repos/abacus_tng_lyalpha/")

from scipy.interpolate import interpn
import numpy as np
from scipy.ndimage import zoom

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from tools import numba_tsc_3D, numba_tsc_irregular_3D, compress_asdf

from astropy.io import ascii

"""
python get_fft_gal_dens.py AbacusSummit_base_c000_ph000 losy
python get_fft_gal_dens.py AbacusSummit_base_c000_ph001 losy
python get_fft_gal_dens.py AbacusSummit_base_c000_ph002 losy
python get_fft_gal_dens.py AbacusSummit_base_c000_ph003 losy
python get_fft_gal_dens.py AbacusSummit_base_c000_ph004 losy
python get_fft_gal_dens.py AbacusSummit_base_c000_ph005 losy
"""

# param names
sim_name = sys.argv[1] #"AbacusSummit_base_c000_ph000"
z = 2.5
#mock_dir = f"/global/cscratch1/sd/boryanah/Ly_alpha/{sim_name}/mocks/z{z:.3f}/"
mock_dir = f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/mocks/{sim_name}/z{z:.3f}/"
save_dir = f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/mocks/{sim_name}/z{z:.3f}/"
os.makedirs(save_dir, exist_ok=True)
paint_type = "TSC"
los_dir = sys.argv[2] #"losz" # "losy"
NP = 6912
Lbox = 2000.
Ndim_lya = 6912
#Ndim_lya = 8000
if Ndim_lya == 6912:
    n_chunks = 576

    klos_min = 0.
    klos_max = 4. # h/Mpc
    kperp_min = 0.
    kperp_max = 2. # h/Mpc
else:
    n_chunks = 500 # 10 per chunk

    klos_min = 0.
    klos_max = 1.6 # h/Mpc
    kperp_min = 0.
    kperp_max = 1.0 # h/Mpc

# tuks maybe not  + 0.5
cell_size_lya = Lbox/Ndim_lya
bins_lya = (np.arange(Ndim_lya+1))*cell_size_lya
dx_lya = bins_lya[1] - bins_lya[0]
Nchunk_lya = Ndim_lya//n_chunks

# load galaxy positions
"""
pos = np.load(f"{sim_name}/z{z:.3f}/pos.npy")
vel = np.load(f"{sim_name}/z{z:.3f}/vel.npy")
"""
t = ascii.read(f"{mock_dir}/QSOs.dat")
if los_dir == "losy":
    pos = np.vstack((t['x'], t['z'], t['y'])).T
    vel = np.vstack((t['vx'], t['vz'], t['vy'])).T
else:
    pos = np.vstack((t['x'], t['y'], t['z'])).T
    vel = np.vstack((t['vx'], t['vy'], t['vz'])).T
del t; gc.collect()

print("number of particles", pos.shape)
#mean_dens = pos.shape[0]/Lbox**3
#mean_dens = pos.shape[0]/1000.**3
mean_dens = pos.shape[0]/Ndim_lya**3
print("mean dens =", mean_dens)
#print(np.mean(np.load(mock_dir+f"gal_dens_Ndim1000.npy")))

def get_k_hMpc(Ndim, L_hMpc):
    # get frequencies from numpy.fft
    k_hMpc = np.fft.fftfreq(Ndim)
    #k_hMpc = np.fft.rfftfreq(Ndim)
    # normalize using box size (first wavenumber should be 2 pi / L_hMpc)
    k_hMpc *= (2.0*np.pi) * Ndim / L_hMpc    
    return k_hMpc

def get_k_hMpc_real(Ndim, L_hMpc):
    # get frequencies from numpy.fft
    k_hMpc = np.fft.rfftfreq(Ndim)
    # normalize using box size (first wavenumber should be 2 pi / L_hMpc)
    k_hMpc *= (2.0*np.pi) * Ndim / L_hMpc    
    return k_hMpc

# get the wavenumbers
kperp_hMpc = get_k_hMpc(Ndim_lya, Lbox)
klos_hMpc = get_k_hMpc_real(Ndim_lya, Lbox)

# get masks
mask_perp_1d = (np.abs(kperp_hMpc) < kperp_max)
mask_los_1d = (np.abs(klos_hMpc) < klos_max)

# create empty complex array
dens_fft = np.zeros((Ndim_lya, np.sum(mask_perp_1d), np.sum(mask_los_1d)), dtype=np.complex64)
print("dens_fft", dens_fft.shape)

try:
    pos[:, 2]
except:
    print("change pos")
    n = len(pos)//3
    pos_new = np.zeros((n, 3), dtype=np.float32)
    for i in range(3):
        pos_new[:, i] = pos[i*n:(i+1)*n]
    pos = pos_new

# to math stuff
pos += Lbox/2.
print(pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max(), pos[:, 2].min(), pos[:, 2].max())
#test = pos[(pos[:, 0] < 40.) & (pos[:, 0] >= 37.5)]
#print(test.shape, test[:, 0].min(), test[:, 0].max(), test[:, 1].min(), test[:, 1].max(), test[:, 2].min(), test[:, 2].max())
#print("bigger than 1?", 4000*1000/pos.shape[0])
#quit()

# load sim_directory
sim_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/"
halo_dir =  f"{sim_dir}/{sim_name}/halos/z{z:.3f}/halo_info/"
fn = halo_dir+"halo_info_000.asdf"
cat = CompaSOHaloCatalog(fn, subsamples=False, fields=['N'])
header = cat.header
inv_velz2kms = 1./(header['VelZSpace_to_kms']/Lbox)
assert los_dir == "losz" or los_dir == "losy"

pos[:, 2] = pos[:, 2] + vel[:, 2]*inv_velz2kms
pos[:, 2] %= Lbox
del vel, cat; gc.collect()

# loop over all chunks
for i in range(n_chunks):
    print(i, n_chunks)

    L = (Nchunk_lya+2)*cell_size_lya
    choice = (bins_lya[i*Nchunk_lya] < pos[:, 0]) & (bins_lya[(i+1)*Nchunk_lya] >= pos[:, 0])
    xmin = bins_lya[i*Nchunk_lya] - dx_lya
    xmax = bins_lya[(i+1)*Nchunk_lya] + dx_lya
    if xmin < 0.:
        extra_min = (Lbox+xmin < pos[:, 0])
        pos[extra_min, 0] -= Lbox
    else:
        extra_min = (xmin < pos[:, 0]) & (bins_lya[i*Nchunk_lya] >= pos[:, 0])
    print("sum choice", np.sum(choice))
    choice |= extra_min
    if xmax > Lbox:
        extra_max = (xmax-Lbox >= pos[:, 0])
        pos[extra_max, 0] += Lbox
    else:
        extra_max = (xmax >= pos[:, 0]) & (bins_lya[(i+1)*Nchunk_lya] < pos[:, 0])
    choice |= extra_max
    print("sum extra", np.sum(extra_max), np.sum(extra_min))
    p = pos[choice]
    p[:, 0] -= xmin
    if xmin < 0.:
        pos[extra_min, 0] += Lbox
    if xmax > Lbox:
         pos[extra_max, 0] -= Lbox
    print("xmax-xmin, p min, max", xmax-xmin, p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(), p[:, 2].min(), p[:, 2].max())
    
    print("xmin, bins_lya[i*Nchunk_lya], bins_lya[(i+1)*Nchunk_lya, xmax, xmax-xmin, L", xmin, bins_lya[i*Nchunk_lya], bins_lya[(i+1)*Nchunk_lya], xmax, xmax-xmin, L)

    # create and save density field
    gdl = np.zeros((Nchunk_lya+2, Ndim_lya, Ndim_lya), dtype=np.float32)
    numba_tsc_irregular_3D(p, gdl, np.array([L, Lbox, Lbox]))
    print("mean gdl", np.mean(gdl))
    del p; gc.collect()
    gdl = gdl[1:-1, :, :]
    print("mean gdl after", np.mean(gdl))
    gdl /= mean_dens
    gdl -= 1.
    
    # now apply fourier transform in yz direction
    gd_fft = np.fft.rfftn(gdl, axes=(2,))[:, :, mask_los_1d].reshape(Nchunk_lya, Ndim_lya, np.sum(mask_los_1d))
    del gdl; gc.collect()
    print("1st fft")
    
    dens_fft[i*Nchunk_lya:(i+1)*Nchunk_lya, :, :] = np.fft.fftn(gd_fft, axes=(1,))[:, mask_perp_1d, :].reshape(Nchunk_lya, np.sum(mask_perp_1d), np.sum(mask_los_1d))
    del gd_fft; gc.collect()
    print("2nd fft")

# can either do this
"""
dens_fft = np.fft.fftn(dens_fft, axes=(0,))[mask_perp_1d, :, :]
"""

# or this which is safer
dens_fft_new = np.zeros((np.sum(mask_perp_1d), np.sum(mask_perp_1d), np.sum(mask_los_1d)), dtype=np.complex64); print(dens_fft_new.shape)
for i in range(dens_fft.shape[1]):
    for j in range(dens_fft.shape[2]):
        if i%100 == 0 and j%100 == 0: print(i, j)
        dens_fft_new[:, i, j] = np.fft.fftn(dens_fft[:, i, j])[mask_perp_1d]
del dens_fft; gc.collect()
dens_fft = dens_fft_new

header = {}
header['Redshift'] = z
header['Simulation'] = sim_name
header['PaintMode'] = paint_type
header['DirectionsOrder'] = 'XYZ'
header['klos_min'] = klos_min
header['klos_max'] = klos_max
header['klos_hMpc'] = klos_hMpc[mask_los_1d]
header['kperp_min'] = kperp_min
header['kperp_max'] = kperp_max
header['kperp_hMpc'] = kperp_hMpc[mask_perp_1d]
del mask_los_1d, mask_perp_1d, klos_hMpc, kperp_hMpc; gc.collect()

if los_dir == "losy":
    dens_fft = np.transpose(dens_fft, (0, 2, 1))

table = {}
table['ComplexDeltaGalaxyReal'] = np.array(dens_fft.real, dtype=np.float32)
table['ComplexDeltaGalaxyImag'] = np.array(dens_fft.imag, dtype=np.float32)
del dens_fft; gc.collect()
print("compressing")
compress_asdf(save_dir+f'complex_QSO_LOS{los_dir[-1]}.asdf', table, header)
print("done")
