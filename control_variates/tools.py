import gc

import numpy as np
import numba

MAX_THREADS = numba.config.NUMBA_NUM_THREADS

# the first 20 factorials
FACTORIAL_LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype=np.int64,
)


@numba.njit
def factorial(n):
    r"""
    Compute the factorial for some integer.

    Parameters
    ----------
    n : int
        integer number for which to calculate the factorial.
        Must be less than or equal to 20 and non-negative.

    Returns
    -------
    factorial : int
        the factorial of the requested integer.
    """
    if n > 20 or n < 0:
        raise ValueError
    factorial = FACTORIAL_LOOKUP_TABLE[n]
    return factorial


@numba.njit
def factorial_slow(x):
    r"""
    Brute-force compute the factorial for some integer.

    Parameters
    ----------
    x : int
        integer number for which to calculate the factorial.

    Returns
    -------
    n : int
        the factorial of the requested integer.
    """
    n = 1
    for i in range(2, x + 1):
        n *= i
    return n


@numba.njit
def n_choose_k(n, k):
    r"""
    Compute binomial coefficient for a choice of two integers (n, k).

    Parameters
    ----------
    n : int
        the integer `n` in n-choose-k.
    k : int
        the integer `k` in n-choose-k.

    Returns
    -------
    x : int
        binomial coefficient, n-choose-k.
    """
    x = factorial(n) // (factorial(k) * factorial(n - k))
    return x



@numba.njit
def P_n(x, n, dtype=np.float32):
    r"""
    Computes Legendre polynomial of order n for some squared quantity x. Maximum tested
    order of the polynomial is 10, after which we see deviations from `scipy`.

    Parameters
    ----------
    x : float
        variable in the polynomial.
    n : int
        order of the Legendre polynomial.

    Returns
    -------
    sum : float
        evaluation of the polynomial at `x`.
    """
    sum = dtype(0.0)
    for k in range(n // 2 + 1):
        factor = dtype(n_choose_k(n, k) * n_choose_k(2 * n - 2 * k, n))
        if k % 2 == 0:
            sum += factor * x ** (dtype(0.5 * (n - 2 * k)))
        else:
            sum -= factor * x ** (dtype(0.5 * (n - 2 * k)))
    sum *= dtype(0.5**n)
    return sum

@numba.njit
def linear_interp(xd, x, y):
    r"""
    Custom linear interpolation. Assumes `x` entries are equidistant and monotonically increasing.
    Assigns `y[0]` and `y[-1]` to the leftmost and rightmost edges, respectively.

    Parameters
    ----------
    xd : float
        x-value at which to evaluate function y(x).
    x : array_type
        equidistantly separated x values at which function y is provided.
    y : array_type
        y values at each x.

    Returns
    -------
    yd : float
        linearly interpolated value at `xd`.
    """
    if xd <= x[0]:
        return y[0]
    elif xd >= x[-1]:
        return y[-1]
    dx = x[1] - x[0]
    f = (xd - x[0]) / dx
    fl = np.int64(f)
    yd = y[fl] + (f - fl) * (y[fl + 1] - y[fl])
    return yd

@numba.njit(parallel=True, fastmath=True)
def bin_kmu(weights, kxy, kz, kedges, muedges, poles=np.empty(0, 'i8'), dtype=np.float32, nthread=MAX_THREADS):

    numba.set_num_threads(nthread)

    Nk = len(kedges) - 1
    Nmu = len(muedges) - 1
    kedges2 = (kedges ** 2).astype(dtype)
    muedges2 = (muedges ** 2).astype(dtype)
    kxy2 = (kxy ** 2).astype(dtype)
    kz2 = (kz ** 2).astype(dtype)
    
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
    weighted_counts_mu = np.zeros((nthread, Nk, Nmu), dtype=dtype)

    # Loop over all k vectors
    for i in numba.prange(len(kxy)):
        tid = numba.get_thread_id()
        i2 = kxy2[i]
        for j in range(len(kxy)):
            bk, bmu = 0, 0
            j2 = kxy2[j]
            for k in range(len(kz)):
                k2 = kz2[k]
                kmag2 = dtype(i2 + j2 + k2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k2) * invkmag2
                else:
                    mu2 = dtype(0.0)  # matches nbodykit

                if kmag2 < kedges2[0]:
                    continue

                if kmag2 >= kedges2[-1]:
                    break

                while kmag2 > kedges2[bk + 1]:
                    bk += 1

                while mu2 > muedges2[bmu + 1]:
                    bmu += 1

                counts[tid, bk, bmu] += 1 if k == 0 else 2
                weighted_counts[tid, bk, bmu] += (
                    weights[i, j, k] if k == 0 else dtype(2.0) * weights[i, j, k]
                )
                weighted_counts_k[tid, bk, bmu] += (
                    np.sqrt(kmag2) if k == 0 else dtype(2.0) * np.sqrt(kmag2)
                )
                weighted_counts_mu[tid, bk, bmu] += (
                    np.sqrt(mu2) if k == 0 else dtype(2.0) * np.sqrt(mu2)
                )
                if Np > 0:
                    for ip in range(len(poles)):
                        pole = poles[ip]
                        if pole != 0:
                            pw = dtype(2 * pole + 1) * P_n(mu2, pole)
                            weighted_counts_poles[tid, ip, bk] += (
                                weights[i, j, k] * pw
                                if k == 0
                                else dtype(2.0) * weights[i, j, k] * pw
                            )

    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)
    weighted_counts_poles = weighted_counts_poles.sum(axis=0)
    weighted_counts_k = weighted_counts_k.sum(axis=0)
    weighted_counts_mu = weighted_counts_mu.sum(axis=0)
    counts_poles = counts.sum(axis=1)

    for ip, pole in enumerate(poles):
        if pole == 0:
            weighted_counts_poles[ip] = weighted_counts.sum(axis=1)

    for i in range(Nk):
        if Np > 0:
            if counts_poles[i] != 0:
                weighted_counts_poles[:, i] /= dtype(counts_poles[i])
        for j in range(Nmu):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
                weighted_counts_k[i, j] /= dtype(counts[i, j])
                weighted_counts_mu[i, j] /= dtype(counts[i, j])
    return (
        weighted_counts,
        counts,
        weighted_counts_poles,
        counts_poles,
        weighted_counts_k,
        weighted_counts_mu,
    )


@numba.njit(parallel=True, fastmath=True)
def bin_rppi(weights, rxy, rz, rpedges, piedges, dtype=np.float32, nthread=MAX_THREADS):

    numba.set_num_threads(nthread)

    Nrp = len(rpedges) - 1
    Npi = len(piedges) - 1
    rpedges2 = (rpedges ** 2).astype(dtype)
    piedges2 = (piedges ** 2).astype(dtype)
    rxy2 = (rxy ** 2).astype(dtype)
    rz2 = (rz ** 2).astype(dtype)
    
    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nrp, Npi), dtype=np.int64)
    weighted_counts = np.zeros((nthread, Nrp, Npi), dtype=dtype)

    # Loop over all k vectors
    for i in numba.prange(len(rxy)):
        tid = numba.get_thread_id()
        i2 = rxy2[i]
        for j in range(len(rxy)):
            brp, bpi = 0, 0 # I think you can move brp = 0 one or two? up
            j2 = rxy2[j]
            rpmag2 = dtype(i2 + j2)

            if rpmag2 < rpedges2[0]:
                continue

            if rpmag2 >= rpedges2[-1]:
                break

            while rpmag2 > rpedges2[brp + 1]:
                brp += 1
            
            for k in range(len(rz)):
                k2 = rz2[k]
                pi2 = dtype(k2)

                while pi2 > piedges2[bpi + 1]:
                    bpi += 1

                counts[tid, brp, bpi] += 1 if k == 0 else 2
                weighted_counts[tid, brp, bpi] += (
                    weights[i, j, k] if k == 0 else dtype(2.0) * weights[i, j, k]
                )


    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)

    for i in range(Nrp):
        for j in range(Npi):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
    return (
        weighted_counts,
        counts
    )


@numba.njit(parallel=True, fastmath=True)
def expand_poles_to_3d(k_ell, P_ell, kxy, kz, poles, dtype=np.float32):

    assert np.abs((k_ell[1] - k_ell[0]) - (k_ell[-1] - k_ell[-2])) < 1.0e-6
    numba.get_num_threads()
    Pk = np.zeros((len(kxy), len(kxy), len(kz)), dtype=dtype)
    k_ell = k_ell.astype(dtype)
    P_ell = P_ell.astype(dtype)
    kxy2 = (kxy ** 2).astype(dtype)
    kz2 = (kz ** 2).astype(dtype)
    
    # Loop over all k vectors
    for i in numba.prange(len(kxy)):
        numba.get_thread_id()
        i2 = kxy2[i]
        for j in range(len(kxy)):
            j2 = kxy2[j]
            for k in range(len(kz)):
                k2 = kz2[k]
                kmag2 = dtype(i2 + j2 + k2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k2) * invkmag2
                else:
                    mu2 = dtype(0.0)  # matches nbodykit
                for ip in range(len(poles)):
                    if poles[ip] == 0:
                        Pk[i, j, k] += linear_interp(
                            np.sqrt(kmag2), k_ell, P_ell[ip]
                        )
                    else:
                        Pk[i, j, k] += linear_interp(
                            np.sqrt(kmag2), k_ell, P_ell[ip]
                        ) * P_n(mu2, poles[ip])
    return Pk


def get_rp_pi_box_edges(rperp_hMpc, rlos_hMpc, rperp_max, rlos_max, n_rp_bins, n_pi_bins):
    # recast
    rperp_hMpc = rperp_hMpc.astype(np.float32)
    rlos_hMpc = rlos_hMpc.astype(np.float32)
    
    # this stores *all* Fourier wavenumbers in the box (no binning)
    rx = rperp_hMpc[:, np.newaxis, np.newaxis]
    ry = rperp_hMpc[np.newaxis, :, np.newaxis]
    rz = rlos_hMpc[np.newaxis, np.newaxis, :]
    rp_box = np.sqrt(rx**2 + ry**2)
    del rx, ry; gc.collect()
    
    # construct mu in two steps, without NaN warnings
    pi_box = rz / np.ones_like(rp_box)
    rp_box = rp_box * np.ones_like(pi_box)
    rp_box = rp_box.flatten()
    pi_box = pi_box.flatten()
    print(rp_box.shape, pi_box.shape)

    # define mu-binning
    rp_bin_edges = np.linspace(0., rperp_max, n_rp_bins + 1)
    pi_bin_edges = np.linspace(0., rlos_max, n_pi_bins + 1)

    return rp_box, pi_box, rp_bin_edges, pi_bin_edges

def get_rp_pi_edges(rperp_max, rlos_max, n_rp_bins, n_pi_bins):

    # define mu-binning
    rp_bin_edges = np.linspace(0., rperp_max, n_rp_bins + 1)
    pi_bin_edges = np.linspace(0., rlos_max, n_pi_bins + 1)
    return rp_bin_edges, pi_bin_edges

def compute_xirppi_from_xi3d(Xi, L_hMpc, nperp, nlos, rp_box, pi_box, rp_bin_edges, pi_bin_edges):
    """Actually measure Xi3D from skewers grid (in h/Mpc units)"""

    # flatten to match box
    Xi = Xi.flatten()

    # for the histograming
    ranges = ((rp_bin_edges[0], rp_bin_edges[-1]),(pi_bin_edges[0], pi_bin_edges[-1]))
    nbins2d = (len(rp_bin_edges)-1, len(pi_bin_edges)-1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    
    # compute mean number of modes
    print("binning")
    print(rp_box.shape, pi_box.shape)
    rp, pi, binned_p3d = mean2d_numba_seq(np.array([rp_box, pi_box]), bins=nbins2d, ranges=ranges, weights=Xi, logk=False)

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    xirppi = binned_p3d * nperp**2 * nlos
    return rp, pi, xirppi

@numba.njit(nogil=True, parallel=False)
def mean2d_numba_seq(tracks, bins, ranges, weights=np.empty(0), dtype=np.float32, logk=True):
    """
    This is 8-9 times faster than np.histogram
    We give the signature here so it gets precompmiled
    In theory, this could even be threaded (nogil!)
    """
    H = np.zeros((bins[0], bins[1]), dtype=dtype)
    N = np.zeros((bins[0], bins[1]), dtype=dtype)
    RP = np.zeros((bins[0], bins[1]), dtype=dtype)
    PI = np.zeros((bins[0], bins[1]), dtype=dtype)
    if logk:
        delta0 = 1/(np.log(ranges[0,1]/ranges[0,0]) / bins[0])
    else:
        delta0 = 1/((ranges[0,1] - ranges[0,0]) / bins[0])
    delta1 = 1/((ranges[1,1] - ranges[1,0]) / bins[1])
    Nw = len(weights)

    for t in range(tracks.shape[1]):
        if logk:
            i = np.log(tracks[0,t]/ranges[0,0]) * delta0
        else:
            i = (tracks[0,t] - ranges[0,0]) * delta0
        j = (tracks[1,t] - ranges[1,0]) * delta1
        if 0 <= i < bins[0] and 0 <= j < bins[1]:

            N[int(i),int(j)] += 1.
            H[int(i),int(j)] += weights[t]
            RP[int(i),int(j)] += tracks[0,t]
            PI[int(i),int(j)] += tracks[1,t]

    for i in range(bins[0]):
        for j in range(bins[1]):
            if N[i, j] > 0.:
                H[i, j] /= N[i, j]
                RP[i, j] /= N[i, j]
                PI[i, j] /= N[i, j]
    return RP, PI, H
