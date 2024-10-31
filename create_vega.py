import sys
import numpy as np
import fitsio
import astropy.io.fits as pyfits

def average_2x2_block(array):
    # Get the dimensions of the original array
    n = array.shape[0] // 2
    
    # Reshape and compute the mean across the appropriate axes
    averaged_array = array.reshape(n, 2, n, 2).mean(axis=(1, 3))
    
    return averaged_array

def block(covmat,indices) :
    res=np.zeros((indices.size,indices.size))
    for i in range(indices.size) :
        res[i,:]=covmat[indices[i],indices]
    return res

def compute_wedge(rp,rt,input_xi2d,input_cov,murange=[0.8,1.0],rrange=[10,180],rbin=4,rpmin=None,beta=0,rpmax=None,optimal=True) :

    orp=rp.copy()
    rp=np.abs(rp)

    # indexing
    rstep = rt[1]-rt[0]
    #print("rstep=",rstep)
    rr=np.sqrt(rt**2+rp**2)

    rt_edges=np.zeros((rt.size,2,2))
    rp_edges=np.zeros((rp.size,2,2))
    for i in range(2) :
        for j in range(2) :
            rt_edges[:,i,j]=rt[:]-rstep/2.+i*rstep
            rp_edges[:,i,j]=rp[:]-rstep/2.+j*rstep
    rr_edges=np.sqrt(rt_edges**2+rp_edges**2)
    mu_edges=rp_edges/(rr_edges+(rr_edges==0))

    rr_min=np.min(np.min(rr_edges,axis=-1),axis=-1)
    rr_max=np.max(np.max(rr_edges,axis=-1),axis=-1)
    mu_min=np.min(np.min(mu_edges,axis=-1),axis=-1)
    mu_max=np.max(np.max(mu_edges,axis=-1),axis=-1)

    nr=int((rrange[1]-rrange[0])/rbin)
    r=rrange[0]+rbin/2+np.arange(nr)*rbin

    selection = (mu_max>=murange[0])&(mu_min<=murange[1])&(rr_max>=rrange[0])&(rr_min<=rrange[1])
    if rpmin is not None : selection &= (orp>rpmin)
    if rpmax is not None : selection &= (orp<rpmax)
    wedge_indices=np.where(selection)[0]
    wedge_data=input_xi2d[wedge_indices]
    rr=rr[wedge_indices]
    rt=rt[wedge_indices]
    rp=rp[wedge_indices]
    rr_edges=rr_edges[wedge_indices]
    mu_edges=mu_edges[wedge_indices]
    rr_min=rr_min[wedge_indices]
    rr_max=rr_max[wedge_indices]
    mu_min=mu_min[wedge_indices]
    mu_max=mu_max[wedge_indices]

    ndata=wedge_data.size
    wedge_cov=block(input_cov,wedge_indices)

    var=np.diag(wedge_cov)

    weights=np.zeros((nr,ndata))
    res = np.zeros(nr)
    err = np.zeros(nr)

    for i in range(nr) :
        rmin=r[i]-rbin/2.
        rmax=r[i]+rbin/2.
        bin_indices=np.where((rr_max>=rmin)&(rr_min<=rmax))[0]
        if optimal :
            binfrac = np.zeros(len(bin_indices))
        for k,j in enumerate(bin_indices) :
            # find fraction of each pixel in slice rmin,rmax,mu_min,mu_max with subsampling pixel
            n=7
            rtb=np.tile(np.linspace(rt[j]-rstep/2.+rstep/n/2,rt[j]+rstep/2.-rstep/n/2.,n),(n,1)).ravel()
            rpb=np.tile(np.linspace(rp[j]-rstep/2.+rstep/n/2,rp[j]+rstep/2.-rstep/n/2.,n),(n,1)).T.ravel()
            rrb=np.sqrt(rtb**2+rpb**2)
            mub=rpb/rrb
            if optimal :
                binfrac[k] = np.sum((mub>=murange[0])*(mub<=murange[1])*(rrb>=rmin)*(rrb<rmax))/n**2
            weights[i,j]=np.sum((mub>=murange[0])*(mub<=murange[1])*(rrb>=rmin)*(rrb<rmax)*(1+beta*mub**2)**2)/var[j] # weight with mu if beta>0

        if not optimal : # normalize because we do a simple weighted average
            s=np.sum(weights[i])
            if s>0 :
                weights[i] /= s
        else :
            # actually solve each bin independently
            bin_cov  = block(wedge_cov,bin_indices)
            bin_icov = np.linalg.inv(bin_cov)
            # print(i,binfrac)
            # reduce icov based on intersection of bins
            for k in range(bin_icov.shape[0]) :
                bin_icov[k] *=  binfrac[k]*binfrac[:]
            # solve
            H = np.ones(bin_icov.shape[0])
            B  = np.inner(H,(bin_icov.dot(wedge_data[bin_indices])))
            A  = np.inner(H,bin_icov.dot(H))
            res[i] = B/A
            err[i] = 1./np.sqrt(A)

    if optimal :
        cov = np.diag(err**2) # I don't have the covariance among bins (but the errors of individual bins are correct)
    else :
        res=weights.dot(wedge_data) # pure weighted average
        cov=weights.dot(wedge_cov.dot(weights.T)) # covariance of weighted average
        err=np.sqrt(np.diag(cov).copy())
    return r,res,err,cov


def main():
    want_fft = True
    if want_fft:
        fft_str = "_fft"
    else:
        fft_str = ""
    corr_type = "rppi"
    model_no = int(sys.argv[1])
    scale_factor = 8
    npart = 144
    #sec_tracer = "LyA"
    sec_tracer = "QSO"
    
    
    N_sim = 6
    for i in range(N_sim):
        sim_name = f"AbacusSummit_base_c000_ph00{i:d}"

        # FFT ONLY CARES ABOUT LAST ENTRY
        rpbins = np.linspace(0, 200, 201); assert want_fft
        npibins = int(rpbins[-1]/(rpbins[1]-rpbins[0]))
        rpbinc = (rpbins[1:] + rpbins[:-1])*.5
        rp = np.tile(rpbinc, (rpbinc.shape[0], 1)).T
        rt = rp.T

        for los_dir in ["z", "y"]:

            if want_fft:
                #save_dir = "/pscratch/sd/b/boryanah/AbacusLymanAlpha/correlations/Xi/z2.500/"
                save_dir = "data_fft/"
                data = np.load(save_dir + f"Xi_{corr_type}_LyAx{sec_tracer}_{sim_name}_Model_{model_no:d}_LOS{los_dir}_d4.0.npz")

                rp_bins = data['rp_bins']
                pi_bins = data['pi_bins']
                xirppi = data['xirppi']

                rp_binc = (rp_bins[1:]+rp_bins[:-1])*.5
                pi_binc = (pi_bins[1:]+pi_bins[:-1])*.5

                rp = np.tile(2+4*np.arange(50),(50,1)).T
                rt = np.tile(2+4*np.arange(50),(50,1))

                choice = (rp_binc < rpbins[-1])[:, None] & (pi_binc < rpbins[-1])[None, :]
                rp = rp[choice].reshape(np.sum(rp_binc < rpbins[-1]), np.sum(pi_binc < rpbins[-1]))
                rt = rt[choice].reshape(np.sum(rp_binc < rpbins[-1]), np.sum(pi_binc < rpbins[-1]))
                xirppi = xirppi[choice].reshape(np.sum(rp_binc < rpbins[-1]), np.sum(pi_binc < rpbins[-1]))
                res = (xirppi.T)
            else:
                data = np.load(f"/global/homes/b/boryanah/repos/abacus_tng_lyalpha/julien/autocorr_{corr_type}_dF_{sim_name}_Model_{model_no:d}_LOS{los_dir}_part_143_down{scale_factor:d}.npz")
                xi_s_mu = data['xi_s_mu']

                data_special = np.load("/global/homes/b/boryanah/repos/abacus_tng_lyalpha/julien/autocorr_rppi_dF_AbacusSummit_base_c000_ph004_Model_4_LOSz_part_143_down8.npz")
                npairs = data_special['npairs'] # might need to load from new runs

                xi_s_mu = (xi_s_mu*npairs)[:, :npibins//2][:, ::-1] + (xi_s_mu*npairs)[:, npibins//2:]
                npairs = npairs[:, :npibins//2][:, ::-1] + npairs[:, npibins//2:]
                xi_s_mu = xi_s_mu/npairs
                #xi_s_mu = 0.5*(xi_s_mu[:, :npibins//2][:, ::-1] + xi_s_mu[:, npibins//2:])
                #npairs = 1.0*(npairs[:, :npibins//2][:, ::-1] + npairs[:, npibins//2:])

                xi_s_mu = (xi_s_mu*npairs).reshape(npibins//2, 2, npibins//2, 1).sum(axis=(1, 3))
                npairs = npairs.reshape(npibins//2, 2, npibins//2, 1).sum(axis=(1, 3))
                res = xi_s_mu/npairs
                #xi_s_mu = 0.5*(xi_s_mu[:, :npibins//2][:, ::-1] + xi_s_mu[:, npibins//2:])
                #res = xi_s_mu.reshape((len(rpbins) - 1)//2, 2, npibins//2, 1).mean(axis=(1, 3))

                rp = average_2x2_block(rp)
                rt = average_2x2_block(rt)
                res = res.T # need this because it's rprt
                print(res)

            rpn = np.zeros((50, 50))
            rtn = np.zeros((50, 50))
            resn = np.zeros((50, 50))

            rpn[:rp.shape[0], :rp.shape[1]] = rp
            rtn[:rt.shape[0], :rt.shape[1]] = rt
            resn[:rt.shape[0], :rt.shape[1]] = res

            rp = rpn
            rt = rtn
            res = resn

            # now copy this on DESI data format tuks!!!!!!!!!!!!!!!
            if sec_tracer == "QSO":
                # load qso covariance matrix
                h = pyfits.open("/global/cfs/cdirs/desicollab/users/jguy/pk2xi/eboss-covariance/eboss-dr16-xcf-2500x2500-covariance.fits")
                cov = h[0].data / 20.
            else:
                h = pyfits.open("/global/cfs/cdirs/desicollab/users/jguy/pk2xi/cf_lya_x_lya_desi_y1.fits")
                cov = h["COR"].data["CO"]/20.
            # use this as an empty shell
            h = pyfits.open("/global/cfs/cdirs/desicollab/users/jguy/pk2xi/cf_boryana_from_fft/cf_boryana_model_1_mean.fits")
            h["COR"].data["DA"] = res.ravel()
            #h["COR"].data["CO"] /= 20. # matrice de covariance de DESI Y1
            h["COR"].data["CO"] = cov
            h["COR"].data["RP"] = rp.ravel()
            h["COR"].data["RT"] = rt.ravel()
            h["COR"].data["Z"] = 2.5
            
            h.writeto(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/{sim_name}/cf_lya_x_{sec_tracer.lower()}_abacus_{model_no:d}_LOS{los_dir}{fft_str}.fits", overwrite=True)

    h = None
    corr = None
    ncorr = 0
    for r in range(N_sim):
        for los_dir in ["z", "y"]:
            sim_name = f"AbacusSummit_base_c000_ph00{r:d}"
            filename = f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/{sim_name}/cf_lya_x_{sec_tracer.lower()}_abacus_{model_no:d}_LOS{los_dir}{fft_str}.fits"
            print("reading", filename)
            hr = pyfits.open(filename)
            if h is None:
                h = hr
                corr = hr["COR"].data["DA"]
                ncorr = 1
            else :
                corr  += hr["COR"].data["DA"]
                ncorr += 1

    corr /= ncorr
    h["COR"].data["DA"] = corr
    h["COR"].data["CO"] /= ncorr # TESTING
    h["COR"].data["Z"] = 2.5
    h["COR"].data["RP"] = np.tile(2+4*np.arange(50),(50,1)).T.ravel()
    h["COR"].data["RT"] = np.tile(2+4*np.arange(50),(50,1)).ravel()
    h.writeto(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/cf_lya_x_{sec_tracer.lower()}_abacus_{model_no}_mean{fft_str}.fits", overwrite=True)

    quit()
    # TESTING!!!!!
    r,xim,_,_ = compute_wedge(h["COR"].data["RP"],h["COR"].data["RT"],h["COR"].data["DA"],h["COR"].data["CO"],murange=[0.,0.5],rrange=[0,148])
    fitsio.write(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/cf_lya_x_{sec_tracer.lower()}_abacus_mean_r{fft_str}.fits", r)
    fitsio.write(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/cf_lya_x_{sec_tracer.lower()}_abacus_mean_0_5{fft_str}.fits", xim)

    r,xim,_,_ = compute_wedge(h["COR"].data["RP"],h["COR"].data["RT"],h["COR"].data["DA"],h["COR"].data["CO"],murange=[0.5,0.8],rrange=[0,148])
    fitsio.write(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/cf_lya_x_{sec_tracer.lower()}_abacus_mean_5_8{fft_str}.fits", xim)

    r,xim,_,_ = compute_wedge(h["COR"].data["RP"],h["COR"].data["RT"],h["COR"].data["DA"],h["COR"].data["CO"],murange=[0.8,0.95],rrange=[0,148])
    fitsio.write(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/cf_lya_x_{sec_tracer.lower()}_abacus_mean_8_95{fft_str}.fits", xim)

    r,xim,_,_ = compute_wedge(h["COR"].data["RP"],h["COR"].data["RT"],h["COR"].data["DA"],h["COR"].data["CO"],murange=[0.95,1.],rrange=[0,148])
    fitsio.write(f"/pscratch/sd/b/boryanah/abacus_tng_lyalpha/cf_lya_x_{sec_tracer.lower()}_abacus_mean_95_1{fft_str}.fits", xim)

main()
