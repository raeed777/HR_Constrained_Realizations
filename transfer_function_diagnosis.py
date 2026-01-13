import numpy as np
from helper_tools import kgrid_rfft3d
def rfft_w_last(n):
    w = np.ones(n//2 + 1, dtype=float)
    w[1:n//2] = 2.0
    if n % 2 == 0:
        w[n//2] = 1.0
    return w

def kgrid_rfft_pp(n, L):
    # returns kx, ky, kz (rfft), k, mu
    dx = L / n
    kx = 2*np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K  = np.sqrt(K2)
    mu = np.zeros_like(K)
    m = K > 0
    mu[m] = np.abs(KZ[m]) / K[m]  # abs for PP symmetry
    return KX, KY, KZ, K, mu

def bin_transfer_H_emp(phi_true, phi_hat, Lbox, kbins, mubins=None):
    """
    Empirical transfer H_emp = <phi_hat phi_true*> / <|phi_true|^2>
    computed on rFFT grid with multiplicity weights.

    phi_true, phi_hat : (n,n,n) real arrays
    kbins : 1D bin edges in k (same units as 2pi/Lbox)
    mubins: 1D bin edges in mu in [0,1] or None (if None => isotropic bins)

    Returns dict with bin centers and H_emp
    """
    n = phi_true.shape[0]
    assert phi_true.shape == (n,n,n) and phi_hat.shape == (n,n,n)

    # FFTs
    T = np.fft.rfftn(phi_true)
    H = np.fft.rfftn(phi_hat)

    # k, mu grid
    _, _, _, K, MU = kgrid_rfft_pp(n, Lbox)

    # multiplicity weights
    w_last = rfft_w_last(n)[None, None, :]
    Wk = w_last * np.ones_like(K)

    # drop DC
    mask = K > 0

    # numerator/denominator per mode
    num = (H * np.conj(T)).real
    den = (T * np.conj(T)).real

    if mubins is None:
        # isotropic binning
        kbin = np.digitize(K[mask], kbins) - 1
        nb = len(kbins) - 1
        Num = np.zeros(nb)
        Den = np.zeros(nb)
        Wsum = np.zeros(nb)

        wk = Wk[mask]
        for b in range(nb):
            sel = (kbin == b)
            if np.any(sel):
                ww = wk[sel]
                Num[b] = np.sum(ww * num[mask][sel])
                Den[b] = np.sum(ww * den[mask][sel])
                Wsum[b] = np.sum(ww)

        Hemp = np.divide(Num, Den, out=np.zeros_like(Num), where=(Den > 0))
        kcen = 0.5*(kbins[:-1] + kbins[1:])
        return {"k": kcen, "H_emp": Hemp, "Wsum": Wsum}

    else:
        # (k, mu) binning
        kbin  = np.digitize(K[mask],  kbins) - 1
        mubin = np.digitize(MU[mask], mubins) - 1
        nbk = len(kbins) - 1
        nbm = len(mubins) - 1

        Num = np.zeros((nbk, nbm))
        Den = np.zeros((nbk, nbm))
        Wsum = np.zeros((nbk, nbm))

        wk = Wk[mask]
        num_m = num[mask]
        den_m = den[mask]

        for i in range(nbk):
            for j in range(nbm):
                sel = (kbin == i) & (mubin == j)
                if np.any(sel):
                    ww = wk[sel]
                    Num[i,j]  = np.sum(ww * num_m[sel])
                    Den[i,j]  = np.sum(ww * den_m[sel])
                    Wsum[i,j] = np.sum(ww)

        Hemp = np.divide(Num, Den, out=np.zeros_like(Num), where=(Den > 0))
        kcen = 0.5*(kbins[:-1] + kbins[1:])
        mucen = 0.5*(mubins[:-1] + mubins[1:])
        return {"k": kcen, "mu": mucen, "H_emp": Hemp, "Wsum": Wsum}

def H_theory_pp(k, mu, b, Dphi, f, a0H0f0, Pphi, sigma):
    """
    k, mu: arrays broadcastable to same shape
    Pphi: P_phi(k) evaluated at k (isotropic spectrum)
    """
    L = (k**2 * (1.0 + f*mu**2)) / (a0H0f0 + 1e-30)  # magnitude (sign irrelevant)
    A = (b**2) * (Dphi**2) * (L**2) * Pphi
    return A / (A + sigma**2 + 1e-30)

def rfft_multiplicity_last_axis(n):
    w = np.full(n//2+1, 2.0, dtype=float)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w

import numpy as np

def H_theory_binned_pp_rfft(
    Lbox, n, Pk_callable,  # Pdelta(k) callable
    b, f0, a0H0f0,
    sigma_x, mask=None,
    Dphi=1.0,
    NBINS=20,
    convention="sampler",
    mu_min=None,  # e.g. 0.95 for "mu≈1"; None for mu-averaged
):
    """
    Compute binned theoretical H(k) by evaluating H(kx,ky,kz) on the rFFT grid and binning.

    Assumptions:
      - PP operator along z
      - constant f0, a0H0f0, scalar Dphi
      - noise treated as white with effective sigma^2 = 1/<M/sigma_x^2>
    """
    V = float(Lbox**3)
    Npts = float(n**3)

    dx = Lbox / n
    kx = 2*np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K  = np.sqrt(K2)

    # mu grid (PP along z)
    MU = np.zeros_like(K)
    mK = K > 0
    MU[mK] = np.abs(KZ[mK]) / K[mK]

    # rFFT multiplicity weights (same idea as your shell_power_rfft)
    wlast = np.ones(n//2 + 1, dtype=float)
    wlast[1:n//2] = 2.0
    if n % 2 == 0:
        wlast[n//2] = 1.0
    wfull = np.broadcast_to(wlast[None, None, :], K.shape)

    # Effective scalar noise variance from your actual weighting
    sig2 = np.asarray(sigma_x, float)**2
    W_x = 1.0 / (sig2 + 1e-30)
    if mask is not None:
        W_x *= np.asarray(mask, float)
    Wbar = float(np.mean(W_x))
    sigma_eff2 = 1.0 / (Wbar + 1e-30)

    # Build Pphi(k) from Pdelta(k) consistently with your model
    Pdelta = Pk_callable(K)  # evaluate on grid
    Pphi   = (a0H0f0**2) * Pdelta / (K2**2 + 1e-30)  # uses k^4 via K2^2

    # PP L(k): magnitude (sign irrelevant)
    Lk = (K2 + f0*(KZ**2)) / (a0H0f0 + 1e-30)

    # A(k) and H(k) per mode
    A = (b**2) * (Dphi**2) * (Lk**2) * Pphi
    Hmodes = A / (A + sigma_eff2 + 1e-30)

    # Optionally restrict to mu≈1 modes
    mode_mask = (K > 0)
    if mu_min is not None:
        mode_mask &= (MU >= float(mu_min))

    # Bin edges exactly like shell_power_rfft (0..K.max)
    edges = np.linspace(0.0, K.max(), NBINS + 1)
    bins  = np.digitize(K.ravel(), edges) - 1
    valid = mode_mask.ravel() & (bins >= 0) & (bins < NBINS)

    if convention == "sampler":
        num_w = (wfull * Hmodes).ravel()
        den_w = np.ones_like(num_w)
    elif convention == "weighted":
        num_w = (wfull * Hmodes).ravel()
        den_w = wfull.ravel()
    else:
        raise ValueError("convention must be 'sampler' or 'weighted'")

    Hsum = np.bincount(bins[valid], weights=num_w[valid], minlength=NBINS)
    Wsum = np.bincount(bins[valid], weights=den_w[valid], minlength=NBINS)

    Hbin = np.where(Wsum > 0, Hsum / Wsum, np.nan)
    kcen = 0.5*(edges[:-1] + edges[1:])
    return kcen, Hbin, sigma_eff2


def shell_cross_power_rfft(field_a: np.ndarray, field_b: np.ndarray, L: float, N: int,
                           convention: str = "sampler",
                           return_counts: bool = True):
    """
    Shell-averaged cross power spectrum P_ab(k) in continuous units [(Mpc/h)^3]
    using NumPy rFFT on a periodic n^3 grid.

    Matches shell_power_rfft() normalization and conventions exactly:
      Pmodes_ab = (V/Npts^2) * Re[A_k B_k*]
      then applies multiplicity weights wfull in the numerator and optionally denominator.
    """
    n = field_a.shape[0]
    assert field_a.shape == (n, n, n) and field_b.shape == (n, n, n), "fields must be cubic (n,n,n)"
    V = L**3
    Npts = n**3

    Ak = np.fft.rfftn(field_a)
    Bk = np.fft.rfftn(field_b)

    # k-grid (h/Mpc)
    kx = 2*np.pi * np.fft.fftfreq(n, d=L/n)
    ky = 2*np.pi * np.fft.fftfreq(n, d=L/n)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=L/n)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Km = np.sqrt(KX**2 + KY**2 + KZ**2)

    # multiplicity weights
    wlast = rfft_multiplicity_last_axis(n)[None, None, :]
    wfull = np.broadcast_to(wlast, Km.shape)

    # per-mode cross power in continuous units
    Pmodes = (V / Npts**2) * (Ak * Bk.conj()).real

    # choose averaging convention
    if convention == "sampler":
        num_arr = wfull * Pmodes
        den_arr = np.ones_like(wfull)
    elif convention == "weighted":
        num_arr = wfull * Pmodes
        den_arr = wfull
    else:
        raise ValueError("convention must be 'sampler' or 'weighted'")

    # binning (matches your shell_power_rfft)
    edges = np.linspace(0.0, Km.max(), N + 1)
    bins = np.digitize(Km.ravel(), edges) - 1
    valid = (bins >= 0) & (bins < N)

    Psum = np.bincount(bins[valid], weights=num_arr.ravel()[valid], minlength=N)
    Wsum = np.bincount(bins[valid], weights=den_arr.ravel()[valid], minlength=N)

    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = np.where(Wsum > 0, Psum / Wsum, np.nan)

    k_centers = 0.5 * (edges[:-1] + edges[1:])
    if return_counts:
        return k_centers, Pk, Wsum
    return k_centers, Pk



def shell_auto_power_rfft_phys(a, L, kbins):
    n = a.shape[0]
    V = float(L**3)
    N = float(n**3)
    pref = V / (N**2)

    Ak = np.fft.rfftn(a)

    dx = L / n
    kx = 2*np.pi*np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi*np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    w_last = np.ones(n//2 + 1, dtype=float)
    w_last[1:n//2] = 2.0
    if n % 2 == 0:
        w_last[n//2] = 1.0
    Wk = w_last[None, None, :] * np.ones_like(K)

    power = (Ak * np.conj(Ak)).real

    mask = (K > 0)
    ind = np.digitize(K[mask], kbins) - 1
    nb = len(kbins) - 1

    P = np.zeros(nb)
    Wsum = np.zeros(nb)

    wk = Wk[mask]
    xm = power[mask]

    for i in range(nb):
        sel = (ind == i)
        if np.any(sel):
            ww = wk[sel]
            P[i] = pref * (np.sum(ww * xm[sel]) / np.sum(ww))
            Wsum[i] = np.sum(ww)

    kcen = 0.5*(kbins[:-1] + kbins[1:])
    return kcen, P, Wsum


import numpy as np

def shell_cross_power_rfft_phys(a, b, L, kbins):
    """
    Cross power P_ab(k) estimated from rFFT with physical-style normalization:
        P_ab(k) = (V/N^2) * < w_k * Re[A_k B_k*] >_shell
    where:
      - V = L^3
      - N = n^3
      - w_k is rFFT multiplicity (2 interior kz, 1 on kz=0/Nyquist)
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.shape[0]
    assert a.shape == (n,n,n) and b.shape == (n,n,n)

    V = float(L**3)
    N = float(n**3)
    pref = V / (N**2)

    Ak = np.fft.rfftn(a)
    Bk = np.fft.rfftn(b)

    # k grid
    dx = L / n
    kx = 2*np.pi*np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi*np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # multiplicity weights
    w_last = np.ones(n//2 + 1, dtype=float)
    w_last[1:n//2] = 2.0
    if n % 2 == 0:
        w_last[n//2] = 1.0
    Wk = w_last[None, None, :] * np.ones_like(K)

    # mode cross
    cross = (Ak * np.conj(Bk)).real

    # binning (exclude DC)
    mask = (K > 0)
    ind = np.digitize(K[mask], kbins) - 1
    nb = len(kbins) - 1

    P = np.zeros(nb)
    Wsum = np.zeros(nb)

    # apply BOTH prefactor and rFFT multiplicity weight
    wk = Wk[mask]
    xm = cross[mask]

    for i in range(nb):
        sel = (ind == i)
        if np.any(sel):
            ww = wk[sel]
            P[i] = pref * (np.sum(ww * xm[sel]) / np.sum(ww))
            Wsum[i] = np.sum(ww)

    kcen = 0.5*(kbins[:-1] + kbins[1:])
    return kcen, P, Wsum


import numpy as np

def H_theory_binned_pp_rfft_match_emp(
    Lbox, n, Pk_callable,
    b, f0, a0H0f0,
    sigma_x, mask=None,
    Dphi=1.0,
    NBINS=20,
    mu_min=None,     # e.g. 0.95 for mu≈1; None for mu-avg
):
    """
    Binned PP theory transfer matched to the empirical estimator
    H_emp = (binned cross)/(binned auto).

    Returns (k_centers, H_bin, sigma_eff2).
    """
    dx = Lbox / n
    kx = 2*np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K  = np.sqrt(K2)

    # mu
    MU = np.zeros_like(K)
    mK = (K > 0)
    MU[mK] = np.abs(KZ[mK]) / K[mK]

    # rFFT multiplicity weights
    wlast = np.ones(n//2 + 1, dtype=float)
    wlast[1:n//2] = 2.0
    if n % 2 == 0:
        wlast[n//2] = 1.0
    wfull = np.broadcast_to(wlast[None, None, :], K.shape)

    # effective sigma^2 from your real-space weights
    sig2 = np.asarray(sigma_x, float)**2
    W_x = 1.0 / (sig2 + 1e-30)
    if mask is not None:
        W_x *= np.asarray(mask, float)
    Wbar = float(np.mean(W_x))
    sigma_eff2 = 1.0 / (Wbar + 1e-30)

    # spectra on grid
    Pdelta = np.asarray(Pk_callable(K), float)
    Pdelta = np.maximum(Pdelta, 0.0)

    # Pphi(k) (continuum spectral version)
    Pphi = (a0H0f0**2) * Pdelta / (K2**2 + 1e-30)

    # PP operator magnitude
    Lk = (K2 + f0*(KZ**2)) / (a0H0f0 + 1e-30)

    # per-mode transfer
    A = (b**2) * (Dphi**2) * (Lk**2) * Pphi
    Hmodes = A / (A + sigma_eff2 + 1e-30)

    # mode selection
    mode_mask = (K > 0) & np.isfinite(Hmodes) & np.isfinite(Pphi)
    if mu_min is not None:
        mode_mask &= (MU >= float(mu_min))

    # bin edges exactly like your shell_power_rfft (0..K.max)
    edges = np.linspace(0.0, K.max(), NBINS + 1)
    bins = np.digitize(K.ravel(), edges) - 1
    valid = mode_mask.ravel() & (bins >= 0) & (bins < NBINS)

    # --- THIS is the crucial estimator match ---
    # H_bin = sum( w * H * Pphi ) / sum( w * Pphi )
    num = (wfull * Hmodes * Pphi).ravel()
    den = (wfull * Pphi).ravel()

    Hsum = np.bincount(bins[valid], weights=num[valid], minlength=NBINS)
    Dsum = np.bincount(bins[valid], weights=den[valid], minlength=NBINS)

    Hbin = np.where(Dsum > 0, Hsum / Dsum, np.nan)
    kcen = 0.5*(edges[:-1] + edges[1:])
    return kcen, Hbin, sigma_eff2

def H_theory_binned_match_emp_using_kbins(
    box, Pk_callable, b, f0, a0H0f0, Dphi,
    sigma_x, mask, kbins
):
    n, L = box.n, box.L
    dx = L / n
    dx3 = box.dx**3

    # k-grid
    kx = 2*np.pi*np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi*np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K  = np.sqrt(K2)

    # rFFT multiplicity
    wlast = rfft_multiplicity_last_axis(n)[None, None, :]
    wfull = np.broadcast_to(wlast, K.shape).astype(float)

    # effective sigma^2 from W = mask/sigma^2
    sig2 = np.asarray(sigma_x, float)**2
    W_x = 1.0 / (sig2 + 1e-30)
    if mask is not None:
        W_x *= np.asarray(mask, float)
    Wbar = float(np.mean(W_x))
    sigma_like2 = 1.0 / (Wbar + 1e-30)

    # match "physical P(k)" convention:
    sigma_eff2 = sigma_like2 * dx3

    # spectra
    Pdelta = np.maximum(np.asarray(Pk_callable(K), float), 0.0)
    Pphi   = (a0H0f0**2) * Pdelta / (K2**2 + 1e-30)

    # PP operator magnitude
    Lk = (K2 + f0*(KZ**2)) / (a0H0f0 + 1e-30)

    A = (b**2) * (Dphi**2) * (Lk**2) * Pphi
    Hm = A / (A + sigma_eff2 + 1e-30)

    # binning on your kbins
    bins = np.digitize(K.ravel(), kbins) - 1
    valid = (K.ravel() > 0) & (bins >= 0) & (bins < len(kbins)-1)

    num = (wfull * Hm * Pphi).ravel()
    den = (wfull * Pphi).ravel()

    Hsum = np.bincount(bins[valid], weights=num[valid], minlength=len(kbins)-1)
    Dsum = np.bincount(bins[valid], weights=den[valid], minlength=len(kbins)-1)

    Hbin = np.where(Dsum > 0, Hsum / Dsum, np.nan)
    kcen = 0.5*(kbins[:-1] + kbins[1:])
    return kcen, Hbin


import numpy as np

def H_theory_binned_from_ops_pp(
    box,
    ops, kbins,
    sigma_x, b, a, H, f,
    mask=None,
    Dphi_scalar=1.0,
):
    K = ops.K
    wfull = np.broadcast_to(ops.w_k, K.shape).astype(float)

    # sigma_like^2 = 1 / < M / sigma_x^2 >
    sig2 = np.asarray(sigma_x, float)**2
    W_x = 1.0 / (sig2 + 1e-30)
    if mask is not None:
        W_x *= np.asarray(mask, float)
    Wbar = float(np.mean(W_x))
    sigma_like2 = 1.0 / (Wbar + 1e-30)


    Lk = np.abs(apply_Lk_pp(a, H, f, b, box))

    # discrete prior covariance in rFFT convention
    Sphi = np.asarray(ops.Sphi_k, float)

    R2 = (float(Dphi_scalar)**2) * (Lk**2)
    A  = R2 * Sphi
    Hm = A / (A + sigma_like2 + 1e-30)

    # bin
    bins  = np.digitize(K.ravel(), kbins) - 1
    nbin  = len(kbins) - 1
    valid = (K.ravel() > 0) & (bins >= 0) & (bins < nbin) & np.isfinite(Hm.ravel()) & (Sphi.ravel() > 0)

    num = (wfull * Hm * Sphi).ravel()
    den = (wfull * Sphi).ravel()

    Hsum = np.bincount(bins[valid], weights=num[valid], minlength=nbin)
    Dsum = np.bincount(bins[valid], weights=den[valid], minlength=nbin)

    Hbin = np.where(Dsum > 0, Hsum / Dsum, np.nan)
    kcen = 0.5 * (kbins[:-1] + kbins[1:])
    return kcen, Hbin

def apply_Lk_pp(a, H, f, b_bias, box):

    KX, KY, KZ, K, K2 = kgrid_rfft3d(box)
    Lk_pp = -(b_bias*K2 + f * (KZ**2)) / (a * H * f + 1e-30)
    Lk_pp[0, 0, 0] = 0.0
    return Lk_pp