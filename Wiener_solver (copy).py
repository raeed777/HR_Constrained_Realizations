import numpy as np
from Box import Box
from Cosmology import Cosmology
from helper_tools import kgrid_rfft3d, Pphi_from_Pdelta
import pickle, time



class Operators:
    def __init__(self, box: Box, cosmo: Cosmology, Pdelta_callable, use_lattice_in_fft=True):
        a, H, f= cosmo.a, cosmo.H, cosmo.f
        self.box = box
        self.a, self.H, self.f = a, H, f
        self.KX, self.KY, self.KZ, self.K, self.K2 = kgrid_rfft3d(box)

        # Prior spectra
        self.Pdelta = Pdelta_callable(self.K)          # (Mpc/h)^3
        self.Pphi   = Pphi_from_Pdelta(self.K, self.Pdelta, a, H, f)

        # FFT prior precision S^{-1}_φ = dx^3 / Pφ
        dx3 = box.dx**3
        self.Sphi_inv_k_spec = np.zeros_like(self.Pphi)
        np.divide(dx3, self.Pphi, out=self.Sphi_inv_k_spec, where=(self.Pphi>0))
        self.Sphi_inv_k_spec[0,0,0] = 0.0

        # Lattice Laplacian symbol: k̃^2 = (2/Δx^2) Σ_i (1 - cos k_i Δx)
        dx = box.dx
        cosx = np.cos(self.KX * dx)
        cosy = np.cos(self.KY * dx)
        cosz = np.cos(self.KZ * dx)
        self.KT2  = (2.0 / dx**2) * (3.0 - cosx - cosy - cosz)
        self.KTZ2 = (2.0 / dx**2) * (1.0 - cosz)  # only z-component

        self._dx = box.dx
        self._laplace_coeff = 1.0 / (self._dx**2)

        # φ -> δ (real, continuum vs lattice)
        self.Lk_real_spec = -(self.K2)  / (a*H*f)                 # δ = -(k^2/(aHf)) φ
        self.Lk_real_lat  = -(self.KT2) / (a*H*f)

        # φ -> δ_s (PP along z, continuum vs lattice)
        self.Lk_pp_spec = -(self.K2 + f*(self.KZ**2))   / (a*H*f) # δ_s = -(k^2+f k_z^2)/(aHf) φ
        self.Lk_pp_lat  = -(self.KT2 + f*self.KTZ2)     / (a*H*f)

        # choose active symbols
        self.Lk_real = self.Lk_real_lat if use_lattice_in_fft else self.Lk_real_spec
        self.Lk_pp   = self.Lk_pp_lat   if use_lattice_in_fft else self.Lk_pp_spec

        # (optional) belt-and-suspenders: zero DC
        self.Lk_real[0,0,0] = 0.0
        self.Lk_pp[0,0,0]   = 0.0

        # Stencil-consistent Pφ (k̃): use in stencil preconditioner if desired
        KT2_safe = np.maximum(self.KT2, 1e-30)
        self.Pphi_lat = (a * H * f) ** 2 * self.Pdelta / (KT2_safe ** 2)
        self.Sphi_inv_k_sten = np.zeros_like(self.Pphi_lat)
        np.divide(dx3, self.Pphi_lat, out=self.Sphi_inv_k_sten, where=(self.Pphi_lat>0))
        self.Sphi_inv_k_sten[0,0,0] = 0.0

        # RSD operator symbol in k-space:
        if use_lattice_in_fft:
            # match the stencil numerics
            self.Lk = -(self.KT2 + f * self.KTZ2) / max(a*H*f, 1e-30)
        else:
            self.Lk = -(self.K2 + f * (self.KZ**2)) / max(a*H*f, 1e-30)

        # Real-space stencil needs dx
        self._dx = dx


    # ----- prior terms -----
    def apply_Sphi_inv_fft(self, x):
        Xk = np.fft.rfftn(x)
        Yk = self.Sphi_inv_k_spec * Xk
        Yk[0,0,0] = 0.0
        return np.fft.irfftn(Yk, s=x.shape)

    def apply_Sphi_inv_stencil(self, x):
        Xk = np.fft.rfftn(x)
        Yk = self.Sphi_inv_k_sten * Xk
        Yk[0,0,0] = 0.0
        return np.fft.irfftn(Yk, s=x.shape)

    # ----- spectral L_rsd_pp (FFT) -----
    def apply_L_rsd_pp_fft(self, phi):
        Phik = np.fft.rfftn(phi)
        return np.fft.irfftn(self.Lk_pp * Phik, s=phi.shape)


    @staticmethod
    def _d2_axis(u, axis, dx):
        return (np.roll(u, +1, axis) + np.roll(u, -1, axis) - 2.0*u) / (dx*dx)
    
    # ----- spectral L_rsd_pp (stencil) -----
    def apply_L_rsd_pp_stencil(self, phi):
        # PP δ_s = -(∇²φ + f ∂_z^2 φ)/(aHf)
        dx = self._dx
        lap = (self._d2_axis(phi,0,dx) + self._d2_axis(phi,1,dx) + self._d2_axis(phi,2,dx))
        dzz = self._d2_axis(phi,2,dx)
        return (lap + self.f * dzz) / (self.a*self.H*self.f)
    

    # ----- spectral L real space (FFt) -----
    def apply_L_real_fft(self, phi):
        Phik = np.fft.rfftn(phi)
        return np.fft.irfftn(self.Lk_real * Phik, s=phi.shape)

    # ----- spectral L real space (stencil) -----
    def apply_L_stencil(self, x):
        # real-space δ = -∇²φ / (aHf)
        dx = self._dx
        lap = (self._d2_axis(x,0,dx) + self._d2_axis(x,1,dx) + self._d2_axis(x,2,dx))
        return lap / (self.a*self.H*self.f)

# -----------------------
# CG / PCG implementation
# -----------------------
def make_precond_Sphi_spectral(ops):
    n = ops.box.n
    def M(r):
        Rk = np.fft.rfftn(r)
        Zk = (ops.Pphi / (ops.box.dx**3)) * Rk  # ≈ S_φ
        Zk[0,0,0] = 0.0
        return np.fft.irfftn(Zk, s=(n,n,n))
    return M

def make_precond_Sphi_stencil(ops):
    n = ops.box.n
    dx3 = ops.box.dx ** 3
    def M(r):
        Rk = np.fft.rfftn(r)
        Zk = (ops.Pphi_lat / dx3) * Rk         # ≈ S_φ (lattice)
        Zk[0,0,0] = 0.0
        return np.fft.irfftn(Zk, s=(n,n,n))
    return M

def pcg(apply_A, b, apply_Minv=None, rtol=1e-6, maxit=200, verbose=True):
    x = np.zeros_like(b)
    r = b - apply_A(x)
    z = apply_Minv(r) if apply_Minv is not None else r.copy()
    p = z.copy()
    rz_old = np.vdot(r, z).real
    norm_b = np.linalg.norm(b)
    t0 = time.perf_counter()
    for it in range(1, maxit+1):
        Ap = apply_A(p)
        alpha = rz_old / np.vdot(p, Ap).real
        x += alpha * p
        r -= alpha * Ap
        res = np.linalg.norm(r)
        if verbose and (it==1 or it%10==0):
            print(f"[PCG] it={it:3d}  |r|/|b|={res/(norm_b+1e-30):.3e}")
        if res <= rtol*(norm_b+1e-30):
            if verbose:
                dt = time.perf_counter()-t0
                print(f"[PCG] converged in {it} iters, time {dt:.2f}s")
            return x
        z = apply_Minv(r) if apply_Minv is not None else r
        rz_new = np.vdot(r, z).real
        beta = rz_new / (rz_old+1e-30)
        p = z + beta * p
        rz_old = rz_new
    print(f"[PCG] reached maxit={maxit}, |r|/|b|={res/(norm_b+1e-30):.3e} in {time.perf_counter()-t0:.2f}s")
    return x

# -----------------------
# Shared helper for weights
# -----------------------
def _make_Wx(sigma_x, M=None, eps=0.0):
    """
    sigma_x: scalar or (n,n,n)
    M: None or (n,n,n) mask in {0,1} or [0,1]
    returns W_x with same broadcastable shape as inputs
    """
    sigma2 = np.asarray(sigma_x, dtype=float)**2 + eps
    Wx = 1.0 / sigma2
    if M is not None:
        Wx = np.asarray(M, dtype=float) * Wx
    return Wx

# -----------------------
# Build matvecs (A·x) & RHS for PP–RSD (spectral L)
# -----------------------
def make_matvec_and_rhs_rsd_spectral(ops: Operators, b_bias, sigma_x, d, M=None, eps=0.0):
    """
    Optional mask M. Per-voxel sigma_x allowed. L is PP–RSD in k-space; W multiplies in real space.
    """
    W_x = _make_Wx(sigma_x, M, eps)

    def apply_A(x):
        y_prior = ops.apply_Sphi_inv_fft(x)
        yL      = ops.apply_L_rsd_pp_fft(x)
        WyL     = W_x * yL
        LtWL    = ops.apply_L_rsd_pp_fft(WyL)  # L^T = L (periodic)
        return y_prior + (b_bias**2) * LtWL

    rhs = b_bias * ops.apply_L_rsd_pp_fft(W_x * d)
    return apply_A, rhs

# -----------------------
# Build matvecs (A·x) & RHS for PP–RSD (stencil L)
# -----------------------
def make_matvec_and_rhs_rsd_stencil(ops: Operators, b_bias, sigma_x, d, M=None, eps=0.0):
    W_x = _make_Wx(sigma_x, M, eps)

    def apply_A(x):
        y_prior = ops.apply_Sphi_inv_stencil(x)
        yL      = ops.apply_L_rsd_pp_stencil(x)
        WyL     = W_x * yL
        LtWL    = ops.apply_L_rsd_pp_stencil(WyL)  # L^T = L (periodic)
        return y_prior + (b_bias**2) * LtWL

    rhs = b_bias * ops.apply_L_rsd_pp_stencil(W_x * d)
    return apply_A, rhs

# -----------------------
# Build matvecs (A·x) & RHS for REAL (spectral L)
# -----------------------
def make_matvec_and_rhs_realspace_spectral(ops: Operators, b_bias, sigma_x, d, M=None, eps=0.0):
    W_x = _make_Wx(sigma_x, M, eps)

    def apply_A(x):
        y_prior = ops.apply_Sphi_inv_fft(x)
        yL      = ops.apply_L_real_fft(x)
        WyL     = W_x * yL
        LtWL    = ops.apply_L_real_fft(WyL)
        return y_prior + (b_bias**2) * LtWL

    rhs = b_bias * ops.apply_L_real_fft(W_x * d)
    return apply_A, rhs

# -----------------------
# Build matvecs (A·x) & RHS for REAL (stencil L)
# -----------------------
def make_matvec_and_rhs_realspace_stencil(ops: Operators, b_bias, sigma_x, d, M=None, eps=0.0):
    """
    7-point Laplacian in real space; optional mask M.
    """
    W_x = _make_Wx(sigma_x, M, eps)

    def apply_A(x):
        y_prior = ops.apply_Sphi_inv_stencil(x)   # spectral prior
        yL      = ops.apply_L_stencil(x)          # L in real space
        WyL     = W_x * yL
        LtWL    = ops.apply_L_stencil(WyL)        # L^T = L
        return y_prior + (b_bias**2) * LtWL

    rhs = b_bias * ops.apply_L_stencil(W_x * d)
    return apply_A, rhs

############################# solvers ##############################
from Observed_Data import ObservedData
from Data import Data

# --- helper to fetch sigma robustly ---
def _get_sigma_x(obs):
    sig = getattr(obs, "sigma", None)
    if sig is None:
        sig = getattr(obs, "sigma_noise", None)
    if sig is None:
        raise AttributeError("ObservedData must have .sigma or .sigma_noise")
    return sig

# --- Wiener mean (REAL space, spectral L) ---
def Wiener_solve_realspace_fft(
    ops_fft: Operators,
    obs_data: ObservedData,
    rtol=1e-6, maxit=300, verbose=True, return_precond=False
):
    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)

    precond = make_precond_Sphi_spectral(ops_fft)
    # If you named the builder differently, adjust here:
    A, rhs  = make_matvec_and_rhs_realspace_spectral(ops_fft, b_bias=b_bias, sigma_x=sigma_x, d=d, M=M)

    t0 = time.perf_counter()
    phi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[Spectral L] total solve time: {time.perf_counter()-t0:.2f}s")

    return (phi, A, rhs, precond) if return_precond else (phi, A, rhs)


# --- One HR constrained realization (REAL space, spectral L) ---
def Constrained_realization_real_space_fft(
    ops_fft: Operators,
    obs_data: ObservedData,
    rng=None, rtol=1e-6, maxit=300, verbose=False,
    reuse=None     # optionally pass (A_fft, precond_fft) for many HR draws
):
    """
    Hoffman–Ribak constrained realization using the spectral branch.
    Consistent with:
      - prior S_phi(k) = (a H f)^2 P_delta(k) / k^4
      - forward L_real_fft: δ = -(k^2/(a H f)) φ
    """
    rng = np.random.default_rng() if rng is None else rng

    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)
    box     = obs_data.box
    cosmo   = obs_data.cosmology

    # 1) prior draw via your Data generator; ensure phi_fft exists
    truth = Data(box, cosmo)
    truth.generate_mock_fields(rng=rng)                 # builds delta_r, etc.
    if getattr(truth, "phi_fft", None) is None:
        if hasattr(truth, "calc_phi"):
            truth.calc_phi()                     # fills phi_fft & phi_sten
        else:
            raise AttributeError("Data lacks calc_phi(); needed to compute phi_fft.")
    phi_rand = truth.phi_fft

    # 2) mock observation with spectral forward model and fresh noise
    nshape = (box.n, box.n, box.n)
    if np.isscalar(sigma_x):
        n_rand = rng.normal(0.0, float(sigma_x), size=nshape)
    else:
        sig = np.asarray(sigma_x, float)
        n_rand = rng.normal(0.0, 1.0, size=sig.shape) * sig

    # y_rand = M [ b L φ_rand + n_rand ]
    y_rand = b_bias * ops_fft.apply_L_real_fft(phi_rand) + n_rand
    if M is not None:
        y_rand = np.asarray(M, float) * y_rand

    # 3) residual and Wiener correction with the SAME system
    residual = d - y_rand

    if reuse is not None:
        A_fft, precond_fft = reuse
        # Rebuild the RHS for the new residual directly (same as your builder)
        eps = 1e-30
        W_x = (np.asarray(M, float) if M is not None else 1.0) / (np.asarray(sigma_x, float)**2 + eps)
        rhs_fft = b_bias * ops_fft.apply_L_real_fft(W_x * residual)
        phi_corr = pcg(A_fft, rhs_fft, apply_Minv=precond_fft,
                       rtol=rtol, maxit=maxit, verbose=verbose)
    else:
        phi_corr, A_fft, rhs_fft = Wiener_solve_realspace_fft(
            ops_fft, obs_data=obs_data.__class__(**{**obs_data.__dict__, "d": residual}), rtol=rtol, maxit=maxit, verbose=verbose
        )
        # ^ light trick: reuse same builder by passing a shallow copy with d=residual

    # 4) constrained realization
    return phi_rand + phi_corr


def Wiener_solve_realspace_stencils(
    ops_sten: Operators,
    obs_data: ObservedData,
    *,
    rtol=1e-6, maxit=300, verbose=True, return_precond=False
):
    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)

    precond = make_precond_Sphi_stencil(ops_sten)
    # If you renamed: make_matvec_and_rhs_realspace_stencil(...)
    A, rhs  = make_matvec_and_rhs_realspace_stencil(ops_sten, b_bias=b_bias, sigma_x=sigma_x, d=d, M=M)

    t0 = time.perf_counter()
    phi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[Stencil L] total solve time: {time.perf_counter()-t0:.2f}s")

    return (phi, A, rhs, precond) if return_precond else (phi, A, rhs)


# --- One HR constrained realization (REAL space, stencil L) ---
def Constrained_realization_real_space_stencils(
    ops_sten: Operators,
    obs_data: ObservedData,
    rng=None, rtol=1e-6, maxit=300, verbose=False
):
    rng = np.random.default_rng() if rng is None else rng


    b_bias = float(obs_data.b_bias)
    # tolerate either .sigma or .sigma_noise
    sigma_x = getattr(obs_data, "sigma", None)
    if sigma_x is None:
        sigma_x = getattr(obs_data, "sigma_noise", None)
    if sigma_x is None:
        raise AttributeError("ObservedData must have .sigma or .sigma_noise")
    d   = obs_data.d
    M   = getattr(obs_data, "mask", None)
    box = obs_data.box
    cosmo = obs_data.cosmology

    # 1) prior draw: generate mock truth and make sure we have φ_sten
    uncon = Data(box, cosmo)
    uncon.generate_mock_fields(rng)       # builds delta_r (and likely v, etc.)
    # ensure φ_sten exists (call your dual-φ method if needed)
    if getattr(uncon, "phi_sten", None) is None:
        if hasattr(uncon, "calc_phi"):
            uncon.calc_phi()           # fills phi_fft and phi_sten
        else:
            raise AttributeError("Data lacks calc_phi() needed to compute phi_sten.")
    phi_rand = uncon.phi_sten          # lattice-consistent φ for stencil path

    # 2) mock observation with the SAME forward model as the solver (stencil L)
    nshape = (box.n, box.n, box.n)
    if np.isscalar(sigma_x):
        n_rand = rng.normal(0.0, float(sigma_x), size=nshape)
    else:
        sig = np.asarray(sigma_x, float)
        n_rand = rng.normal(0.0, 1.0, size=sig.shape) * sig

    y_rand = b_bias * ops_sten.apply_L_stencil(phi_rand) + n_rand
    if M is not None:
        y_rand = np.asarray(M, float) * y_rand

    # 3) residual and Wiener correction (solve the same system with d - y_rand)
    residual = d - y_rand

    from types import SimpleNamespace
    # 1) build a tiny obs-like object for the residual
    obs_resid = SimpleNamespace(
        box=box,
        cosmology=cosmo,
        d=residual,          # <-- the residual you computed
        mask=M,
        b_bias=b_bias,
        sigma_noise=sigma_x  # used by _get_sigma_x(...)
    )

    # 2) call with the correct signature
    phi_corr, _, _ = Wiener_solve_realspace_stencils(
        ops_sten,
        obs_resid,
        rtol=rtol, maxit=maxit, verbose=verbose
    )


    # 4) constrained realization
    phi_cr = phi_rand + phi_corr
    return phi_cr