import numpy as np
import numba

MAX_THREADS = numba.config.NUMBA_NUM_THREADS

@numba.njit(parallel=True, fastmath=True)
def bin_kmu(
    n1d,
    L,
    kedges,
    muedges,
    weights,
    poles=np.empty(0, 'i8'),
    dtype=np.float32,
    fourier=True,
    nthread=MAX_THREADS,
):

    numba.set_num_threads(nthread)

    #kzlen = n1d // 2 + 1 # tuks
    kzlen = n1d 
    Nk = len(kedges) - 1
    Nmu = len(muedges) - 1
    if fourier:
        dk = 2.0 * np.pi / L
    else:
        dk = L / n1d
    kedges2 = ((kedges / dk) ** 2).astype(dtype)
    muedges2 = (muedges**2).astype(dtype)

    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nk, Nmu), dtype=np.int64)
    weighted_counts = np.zeros((nthread, Nk, Nmu), dtype=dtype)
    Np = len(poles)
    if Np == 0:
        poles = np.empty(0, dtype=np.int64)  # so that compiler does not complain
    else:
        poles = poles.astype(np.int64)
    weighted_counts_poles = np.zeros((nthread, len(poles), Nk), dtype=dtype)
    weighted_counts_k = np.zeros((nthread, Nk, Nmu), dtype=dtype)

    # Loop over all k vectors
    for i_1 in numba.prange(n1d):
        tid = numba.get_thread_id()
        for j_1 in range(n1d):
            for k_1 in range(kzlen):
                for i_2 in range(n1d):
                    if i_2 <= i_1: continue
                    i2 = (i_2-i_1)**2 if i_2 - i_1 < n1d // 2 else (i_2 - i_1 - n1d) ** 2
                    for j_2 in range(n1d):
                        if j_2 <= j_1: continue
                        #bk, bmu = 0, 0 # tuks
                        j2 = (j_2-j_1)**2 if j_2 - j_1 < n1d // 2 else (j_2 - j_1 - n1d) ** 2
                        for k_2 in range(kzlen):
                            bk, bmu = 0, 0 # tuks
                            if k_2 <= k_1: continue
                            k2 = (k_2-k_1)**2 if k_2 - k_1 < n1d // 2 else (k_2 - k_1 - n1d) ** 2
                            kmag2 = dtype(i2 + j2 + k2) # tuks
                            
                            if kmag2 < kedges2[0]:
                                continue

                            if kmag2 >= kedges2[-1]:
                                break

                            while kmag2 > kedges2[bk + 1]:
                                bk += 1
                            
                            counts[tid, bk, bmu] += 1 
                            weighted_counts[tid, bk, bmu] += weights[i_1, j_1, k_1] * weights[i_2, j_2, k_2]
                            weighted_counts_k[tid, bk, bmu] += np.sqrt(kmag2) * dk

                            
    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)
    weighted_counts_poles = weighted_counts_poles.sum(axis=0)
    weighted_counts_k = weighted_counts_k.sum(axis=0)
    counts_poles = counts.sum(axis=1)
    
    for i in range(Nk):
        for j in range(Nmu):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
                weighted_counts_k[i, j] /= dtype(counts[i, j])
    return (
        weighted_counts,
        counts,
        weighted_counts_poles,
        counts_poles,
        weighted_counts_k,
    )
