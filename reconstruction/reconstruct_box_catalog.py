from pathlib import Path
import os, gc

import numpy as np
import asdf
import argparse
from astropy.io import ascii

from abacusnbody.metadata import get_meta
from abacusnbody.analysis.tsc import tsc_parallel
from pyrecon import  utils, IterativeFFTParticleReconstruction, MultiGridReconstruction, IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['redshift'] = 0.5 # 0.8
DEFAULTS['tracer'] = "LRG" # "ELG"
DEFAULTS['nmesh'] = 1024
DEFAULTS['sr'] = 12.5 # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"

"""
Usage:
python reconstruct_box_catalog.py --sim_name AbacusSummit_base_c000_ph002 --redshift 0.5 --tracer LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym

# Notes to self (Boryana):
# If you want redshift-dependent bias, then do `recon_tracer.mesh_delta /= b_z` after `recon.set_density_contrast`, where `mesh_delta` is a 512^3 array, and set bias to 1.
# If you want to input your own delta, then initiate `recfunc()` and get rid of assign_data, assign_randoms and set_density_contrast.
# Instead, do `recon_tracer.mesh_delta[...] = delta_new` or `mesh_delta.value = delta_new` (but apply smoothing to delta_new yourself);
# The only thing that MG needs as input is mesh_delta. Finally, read the shifts as before at the positions of interest.
# Note that IFTP updates the positions internally, which complicates the reading of the shifts
"""

# random seed
np.random.seed(300)

def main(sim_name, redshift, tracer, nmesh, sr, rectype, convention):
    # new features
    want_rsd = False
    want_lya = True
    
    # how many processes to use for reconstruction: 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 128

    # TESTING! make thinner
    want_make_thinner = False
    thin_str = "_thin" if want_make_thinner else ""
    
    # additional specs of the tracer
    extra = '_'.join(tracer.split('_')[1:])
    tracer = tracer.split('_')[0]
    if extra != '': extra = '_'+extra

    # reconstruction parameters
    if tracer == "LRG":
        bias = 2.2 # +/- 10%
    elif tracer == "ELG":
        bias = 1.3
    elif tracer == "QSO":
        bias = 2.1
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "IFTP":
        recfunc = IterativeFFTParticleReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction

    # simulation parameters
    Lbox = get_meta(sim_name, 0.1)['BoxSize'] # cMpc/h
    cosmo = DESI()
    ff = cosmo.growth_factor(redshift) # Abacus
    H_z = cosmo.hubble_function(redshift) # Abacus
    los = 'z'

    # directory where mock catalogs are saved
    #mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
    mock_dir = Path(f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/mocks/")
    
    # file names of mocks with and without RSD
    #fn = mock_dir / f"{sim_name}/z{redshift:.3f}/galaxies/{tracer}s.dat"
    fn = mock_dir / f"{sim_name}/z{redshift:.3f}/{tracer}s.dat"

    # directory where the reconstructed mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{redshift:.3f}"
    os.makedirs(save_recon_dir, exist_ok=True)
    final_fn = save_recon_dir / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{redshift:.3f}{thin_str}.npz"
    
    # read pos and vel w/o RSD (centered at 0)
    f = ascii.read(fn)
    Position = np.vstack((f['x'], f['y'], f['z'])).T
    Velocity = np.vstack((f['vx'], f['vy'], f['vz'])).T

    if want_rsd:
        """
        # read pos w/ RSD (centered at 0)
        f = ascii.read(fn_rsd)
        PositionRSD = np.vstack((f['x'], f['y'], f['z'])).T
        """
        pass # TODO
    
    # wrap around box so that pos range from [0, Lbox)
    Position %= Lbox
    if want_rsd:
        PositionRSD %= Lbox

    # generate randoms
    rands_fac = 16 #40
    RandomPosition = np.vstack((np.random.rand(rands_fac*Position.shape[0]), np.random.rand(rands_fac*Position.shape[0]), np.random.rand(rands_fac*Position.shape[0])))*Lbox
    RandomPosition = RandomPosition.T


    if want_make_thinner:
        z_max = 830.
        choice = Position[:, 2] < z_max
        Position = Position[choice]
        Velocity = Velocity[choice]
        PositionRSD = PositionRSD[choice]
        choice = RandomPosition[:, 2] < z_max
        RandomPosition = RandomPosition[choice]
    
    # run reconstruction on the mocks w/o RSD
    print('Recon First tracer')
    if want_make_thinner: # maybe this uses the randoms and otherwise no?
        recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position,
                               nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=False)
    else:
        recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position, boxsize=Lbox, boxcenter=(Lbox/2, Lbox/2, Lbox/2), # boxcenter not needed when wrap=True
                               nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer.assign_data(Position)#, dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    #recon_tracer.mesh_delta[:, :, int(.40 * nmesh):] = 0. # TESTING # a bunch of zero and a bit of wrapping
    #recon_tracer.mesh_delta[:, :, int(.9 * nmesh):] = 0. # TESTING make thinner
    print('density constrast calculated, now doing recon', flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # read the displacements in real space
    if not want_lya:
        if rectype == 'IFTP':
            displacements = recon_tracer.read_shifts('data', field='disp')
        else:
            displacements = recon_tracer.read_shifts(Position, field='disp')
    else:
        #pass # tuks
        lya_dir = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/")
        model_no = 1
        los_dir = "z"
        assert los_dir == "z"
        n_parts = 144
        nmesh_lya = 1024
        ngrid = 6912
        cell_size = Lbox/ngrid
        print(sim_name, model_no, los_dir)
        
        # midpoint of each cell along LOS
        yz_bins = np.linspace(0., ngrid, ngrid+1)
        yz_binc = (yz_bins[1:] + yz_bins[:-1]) * .5
        yz_binc *= cell_size # cMpc/h
        yz_binc = yz_binc.astype(np.float32)

        x_ngrid = ngrid//n_parts
        x_slab = x_ngrid * cell_size
        x_bins = np.linspace(0., x_ngrid, x_ngrid+1)
        x_binc = (x_bins[1:] + x_bins[:-1]) * .5
        x_binc *= cell_size # cMpc/h
        x_binc = x_binc.astype(np.float32)
        
        x, y, z  = np.meshgrid(x_binc, yz_binc, yz_binc)
        pos_lya = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        pos_lya -= Lbox/2.
        pos_lya %= Lbox
        del x, y, z; gc.collect()
        
        density = np.zeros((nmesh_lya, nmesh_lya, nmesh_lya), dtype=np.float32)
        for i_part in range(n_parts):
            print(i_part)
            data = asdf.open(lya_dir / sim_name / f"z{redshift:.3f}" / f"Model_{model_no:d}" / f"tau_Model_{model_no:d}_LOS{los_dir}_part_{i_part:03d}.asdf")['data']
            tau_down = data[f'tau_rsd_los{los_dir}']
            
            F = np.exp(-tau_down)
            mean_F = np.mean(F, dtype=np.float64)
            F /= mean_F
            F -= 1.
            F = F.flatten()
            del tau_down, data; gc.collect()
            print(len(F), pos_lya.shape)
            
            displacements = recon_tracer.read_shifts(pos_lya, field='disp')
            pos_lya[:, 0] += x_slab

            # old code corr cat should be rsd but whatever for now
            tsc_parallel(pos_lya - displacements, density, weights=F)
        np.save(save_recon_dir / f"density_lya_Model_{model_no:d}_LOS{los_dir}_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{redshift:.3f}{thin_str}.npy", density)
    random_displacements = recon_tracer.read_shifts(RandomPosition, field='disp')

    if want_lya:
        density = np.zeros((nmesh_lya, nmesh_lya, nmesh_lya), dtype=np.float32)
        tsc_parallel(RandomPosition - random_displacements, density)
        np.save(save_recon_dir / f"density_ran_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{redshift:.3f}{thin_str}.npy", density)
                         
    if want_rsd:
        # run reconstruction on the mocks w/ RSD
        print('Recon Second tracer')
        if want_make_thinner:
            recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=PositionRSD,
                                   nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=False)
        else:
            recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=PositionRSD, boxsize=Lbox, boxcenter=(Lbox/2, Lbox/2, Lbox/2),
                                   nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
        print('grid set up',flush=True)
        recon_tracer.assign_data(PositionRSD)#, dat_cat['WEIGHT'])
        print('data assigned',flush=True)
        recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
        print('randoms assigned',flush=True)
        recon_tracer.set_density_contrast(smoothing_radius=sr)
        print('density constrast calculated, now doing recon', flush=True)
        recon_tracer.run()
        print('recon has been run',flush=True)

        # read the displacements in real and redshift space (rsd has the (1+f) factor in the LOS direction)
        if rectype == 'IFTP':
            displacements_rsd = recon_tracer.read_shifts('data', field='disp+rsd')
            displacements_rsd_nof = recon_tracer.read_shifts('data', field='disp')
        else:
            displacements_rsd = recon_tracer.read_shifts(PositionRSD, field='disp+rsd')
            displacements_rsd_nof = recon_tracer.read_shifts(PositionRSD, field='disp')
        random_displacements_rsd = recon_tracer.read_shifts(RandomPosition, field='disp+rsd')
        random_displacements_rsd_nof = recon_tracer.read_shifts(RandomPosition, field='disp')

    if want_rsd:
        # save the displacements
        np.savez(final_fn,
                 displacements=displacements, displacements_rsd=displacements_rsd, velocities=Velocity, positions=Position, positions_rsd=PositionRSD,
                 growth_factor=ff, Hubble_z=H_z, random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements,
                 random_positions=RandomPosition, displacements_rsd_nof=displacements_rsd_nof, random_displacements_rsd_nof=random_displacements_rsd_nof)
    else:
        np.savez(final_fn,
                 displacements=displacements, velocities=Velocity, positions=Position,
                 growth_factor=ff, Hubble_z=H_z, random_displacements=random_displacements,
                 random_positions=RandomPosition)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--redshift', help='Redshift', type=float, default=DEFAULTS['redshift'])
    parser.add_argument('--tracer', help='Galaxy type', default=DEFAULTS['tracer'])#, choices=["LRG", "ELG", "QSO", "LRG_high_density", "LRG_bgs", "ELG_high_density"])
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    args = vars(parser.parse_args())
    main(**args)
