import numpy as np
from Box import Box
from Cosmology import Cosmology
from Obs_Geometry import Obs_Geometry
from helper_tools import kgrid_rfft3d, Pphi_from_Pdelta, los_unit_and_radius
import pickle, time, copy
from helper_tools import rfft_multiplicity_last_axis

class Operators:
    def __init__(self,
                 box: Box,
                 cosmo: Cosmology,
                 b_bias,
                 Pdelta_callable,
                 obs_geometry: Obs_Geometry
                 ):
        a, H, f = cosmo.a, cosmo.H, cosmo.f
        self.box = box
        self.cosmology = cosmo
        self.a, self.H, self.f = a, H, f
        self.b_bias = b_bias

        # --- k-grid for rFFT ---
        self.KX, self.KY, self.KZ, self.K, self.K2 = kgrid_rfft3d(box)

        # --- Prior spectra: P_δ(k) and P_φ(k) ---
        self.Pdelta = Pdelta_callable(self.K)          # (Mpc/h)^3
        self.Pphi   = Pphi_from_Pdelta(self.K, self.Pdelta, a, H, f)

        dx3 = box.dx**3
        self.dx3 = dx3

        Pphi = np.asarray(self.Pphi, dtype=float)

        # rFFT multiplicity weights w(kz): shape (1,1,n//2+1), broadcasts over (kx,ky)
        w_last = rfft_multiplicity_last_axis(box.n)[None, None, :].astype(float)
        self.w_k = w_last

        # S(k) = Pphi / (dx3 * w)
        self.Sphi_k = np.zeros_like(Pphi)
        np.divide(Pphi, dx3 * w_last, out=self.Sphi_k, where=(Pphi > 0.0))
        self.Sphi_k[0, 0, 0] = 0.0

        # S^{-1}(k) = dx3 * w / Pphi
        self.Sphi_inv_k = np.zeros_like(Pphi)
        np.divide(dx3 * w_last, Pphi, out=self.Sphi_inv_k, where=(Pphi > 0.0))
        self.Sphi_inv_k[0, 0, 0] = 0.0

        # S^{+1/2}(k)
        self.Sphi_sqrt_k = np.zeros_like(Pphi)
        np.sqrt(self.Sphi_k, out=self.Sphi_sqrt_k, where=(self.Sphi_k > 0.0))
        self.Sphi_sqrt_k[0, 0, 0] = 0.0

        # S^{-1/2}(k)
        self.Sphi_sqrt_inv_k = np.zeros_like(Pphi)
        np.sqrt(self.Sphi_inv_k, out=self.Sphi_sqrt_inv_k, where=(self.Sphi_inv_k > 0.0))
        self.Sphi_sqrt_inv_k[0, 0, 0] = 0.0



        # --- φ -> δ and φ -> δ_s (PP) symbols, purely spectral ---
        self.Lk_real = -self.b_bias*(self.K2) / (a * H * f + 1e-30)
        self.Lk_real[0, 0, 0] = 0.0

        self.Lk_pp = -(self.b_bias*self.K2 + f * (self.KZ**2)) / (a * H * f + 1e-30)
        self.Lk_pp[0, 0, 0] = 0.0

        # --- Attach observation geometry (n̂, r, Dφ, maybe z) ---
        self.obs_geometry = obs_geometry

        # LOS unit vector n̂ and 1/r
        self._nhat = obs_geometry.nhat_grid          # typically a tuple (nx, ny, nz)
        self._invR = (1.0 / np.maximum(obs_geometry.r_grid, 1e-30)).astype(np.float32)
        # --- DEBUG: freeze 1/r to a constant ---
        # invR0 = 1.0 / 2500.0
        # self._invR = np.full_like(obs_geometry.r_grid, invR0, dtype=np.float32)

        # Growth of the velocity potential on the grid
        if getattr(obs_geometry, "Dphi_grid", None) is None:
            raise ValueError("obs_geometry.Dphi_grid is None. "
                             "Call obs_geometry.compute_D_grid() and "
                             "obs_geometry.compute_Dphi_grid() first.")
        self.Dphi_grid = np.asarray(obs_geometry.Dphi_grid, dtype=float)

        # (optional) keep z_grid if you want it later
        self.z_grid = getattr(obs_geometry, "z_grid", None)
        self.a_grid = getattr(obs_geometry, "a_grid", None)
        self.H_grid = getattr(obs_geometry, "H_grid", None)
        self.f_grid = getattr(obs_geometry, "f_grid", None)
        self.aHf_grid = (self.a_grid * self.H_grid * self.f_grid).astype(np.float64)


    # ---------- Prior operators in Fourier space ----------

    def apply_Sphi_inv_fft(self, x):
        """
        Apply S_φ^{-1} to a real-space φ(x):
            y = S_φ^{-1} φ
        via rFFT:   φ̂(k) -> dx^3 / P_φ(k) * φ̂(k) -> y(x).
        """
        Xk = np.fft.rfftn(x)
        Yk = self.Sphi_inv_k * Xk
        Yk[0, 0, 0] = 0.0
        return np.fft.irfftn(Yk, s=x.shape).real

    def apply_Sphi_sqrt_fft(self, psi):
        """
        Map ψ(x) -> φ(x) using:
            φ̂(k) = S_φ^{1/2}(k) ψ̂(k),
        where S_φ^{1/2}(k) = sqrt(P_φ(k)/dx^3*w).
        """
        Psi_k = np.fft.rfftn(psi)
        Phi_k = self.Sphi_sqrt_k * Psi_k
        Phi_k[0, 0, 0] = 0.0
        return np.fft.irfftn(Phi_k, s=psi.shape).real

    def apply_Sphi_sqrt_inv_fft(self, phi):
        """
        Map φ(x) -> ψ(x) using:
            ψ̂(k) = φ̂(k) / S_φ^{1/2}(k),
        mainly for diagnostics (ψ “whiteness”).
        """
        Phi_k = np.fft.rfftn(phi)
        Psi_k = self.Sphi_sqrt_inv_k * Phi_k
        Psi_k[0, 0, 0] = 0.0
        return np.fft.irfftn(Psi_k, s=phi.shape).real

    # ---------- Spectral forward models on φ ----------

    def apply_L_real_fft(self, phi):
        """
        δ = -(k^2/(a H f)) φ   in Fourier space.
        """
        Phi_k = np.fft.rfftn(phi)
        Delta_k = self.Lk_real * Phi_k
        return np.fft.irfftn(Delta_k, s=phi.shape).real

    def apply_L_rsd_pp_fft(self, phi):
        """
        Plane-parallel RSD along z:
            δ_s = -(k^2 + f k_z^2)/(a H f) φ.
        """
        Phi_k = np.fft.rfftn(phi)
        Delta_s_k = self.Lk_pp * Phi_k
        return np.fft.irfftn(Delta_s_k, s=phi.shape).real

    # ========= Growth operator in φ-space =========
    def apply_Dphi(self, u):
        """
        Apply the velocity-potential growth factor D_φ(x) in real space:
            (D_φ u)(x) = D_φ(x) * u(x).

        Assumes Obs_Geometry has been attached and Dphi_grid initialized.
        """
        if self.Dphi_grid is None:
            raise ValueError("Dphi_grid is not set on Operators. "
                             "Attach Obs_Geometry and call compute_Dphi_grid().")
        Dφ = self.Dphi_grid
        # Broadcasting-safe: ensure same shape
        if Dφ.shape != u.shape:
            raise ValueError(f"Dphi_grid shape {Dφ.shape} does not match input {u.shape}")
        return Dφ * u

    def apply_Dphi_T(self, u):
        """
        Adjoint of D_φ under the standard real inner product.
        For real, diagonal D_φ this is identical to apply_Dphi.
        """
        return self.apply_Dphi(u)


    # ========= NEW: radial RSD apply (hybrid spectral) =========
    def apply_L_rsd_radial_fft(self, phi, include_geom=True):
        a, H, f = self.a_grid, self.H_grid, self.f_grid
        KX, KY, KZ, K2 = self.KX, self.KY, self.KZ, self.K2
        b_bias = self.b_bias
        Phik = np.fft.rfftn(phi, s=phi.shape)

        lap = np.fft.irfftn(-K2 * Phik, s=phi.shape).real
        Hxx = np.fft.irfftn(-(KX*KX) * Phik, s=phi.shape).real
        Hyy = np.fft.irfftn(-(KY*KY) * Phik, s=phi.shape).real
        Hzz = np.fft.irfftn(-(KZ*KZ) * Phik, s=phi.shape).real
        Hxy = np.fft.irfftn(-(KX*KY) * Phik, s=phi.shape).real
        Hxz = np.fft.irfftn(-(KX*KZ) * Phik, s=phi.shape).real
        Hyz = np.fft.irfftn(-(KY*KZ) * Phik, s=phi.shape).real

        nx, ny, nz = self._nhat
        dnn = (nx*nx)*Hxx + (ny*ny)*Hyy + (nz*nz)*Hzz + 2*(nx*ny)*Hxy + 2*(nx*nz)*Hxz + 2*(ny*nz)*Hyz

        inner = b_bias*lap + self.f_grid * dnn

        if include_geom:
            # v_r = -∂_n φ computed spectrally
            phi_x = np.fft.irfftn((1j*KX) * Phik, s=phi.shape).real
            phi_y = np.fft.irfftn((1j*KY) * Phik, s=phi.shape).real
            phi_z = np.fft.irfftn((1j*KZ) * Phik, s=phi.shape).real
            vr = -(nx*phi_x + ny*phi_y + nz*phi_z)
            inner = inner - 2.0 * self.f_grid * vr * self._invR

        return inner / (self.aHf_grid + 1e-30)

    
    def apply_Lt_rsd_radial_fft(self, y, include_geom=True):
        # If you REALLY want exact self-adjointness for constant f, use this branch
        b_bias = self.b_bias
        if not include_geom:
            # Use scalar background values here for debugging
            a0, H0, f0 = self.a, self.H, self.f   # scalars
            
            Yk  = np.fft.rfftn(y, s=y.shape)
            lap = np.fft.irfftn(-self.K2 * Yk, s=y.shape).real

            nx, ny, nz = self._nhat
            a_xx = (nx*nx) * y;  Axx = np.fft.rfftn(a_xx, s=y.shape)
            a_yy = (ny*ny) * y;  Ayy = np.fft.rfftn(a_yy, s=y.shape)
            a_zz = (nz*nz) * y;  Azz = np.fft.rfftn(a_zz, s=y.shape)
            a_xy = (nx*ny) * y;  Axy = np.fft.rfftn(a_xy, s=y.shape)
            a_xz = (nx*nz) * y;  Axz = np.fft.rfftn(a_xz, s=y.shape)
            a_yz = (ny*nz) * y;  Ayz = np.fft.rfftn(a_yz, s=y.shape)

            d2_xx = np.fft.irfftn(-(self.KX*self.KX) * Axx, s=y.shape).real
            d2_yy = np.fft.irfftn(-(self.KY*self.KY) * Ayy, s=y.shape).real
            d2_zz = np.fft.irfftn(-(self.KZ*self.KZ) * Azz, s=y.shape).real
            d2_xy = np.fft.irfftn(-(self.KX*self.KY) * Axy, s=y.shape).real
            d2_xz = np.fft.irfftn(-(self.KX*self.KZ) * Axz, s=y.shape).real
            d2_yz = np.fft.irfftn(-(self.KY*self.KZ) * Ayz, s=y.shape).real

            term2 = f0 * (d2_xx + d2_yy + d2_zz + 2*d2_xy + 2*d2_xz + 2*d2_yz)
            return (b_bias*lap + term2) / (a0*H0*f0 + 1e-30)

        # --- include_geom=True: grid-based approximate adjoint ---
        KX, KY, KZ, K2 = self.KX, self.KY, self.KZ, self.K2
        nx, ny, nz = self._nhat
        invR = self._invR

        Yk  = np.fft.rfftn(y, s=y.shape)
        lap = np.fft.irfftn(-K2 * Yk, s=y.shape).real

        a_xx = (nx*nx) * y;  Axx = np.fft.rfftn(a_xx, s=y.shape)
        a_yy = (ny*ny) * y;  Ayy = np.fft.rfftn(a_yy, s=y.shape)
        a_zz = (nz*nz) * y;  Azz = np.fft.rfftn(a_zz, s=y.shape)
        a_xy = (nx*ny) * y;  Axy = np.fft.rfftn(a_xy, s=y.shape)
        a_xz = (nx*nz) * y;  Axz = np.fft.rfftn(a_xz, s=y.shape)
        a_yz = (ny*nz) * y;  Ayz = np.fft.rfftn(a_yz, s=y.shape)

        d2_xx = np.fft.irfftn(-(KX*KX) * Axx, s=y.shape).real
        d2_yy = np.fft.irfftn(-(KY*KY) * Ayy, s=y.shape).real
        d2_zz = np.fft.irfftn(-(KZ*KZ) * Azz, s=y.shape).real
        d2_xy = np.fft.irfftn(-(KX*KY) * Axy, s=y.shape).real
        d2_xz = np.fft.irfftn(-(KX*KZ) * Axz, s=y.shape).real
        d2_yz = np.fft.irfftn(-(KY*KZ) * Ayz, s=y.shape).real

        term2 = self.f_grid * (d2_xx + d2_yy + d2_zz + 2*d2_xy + 2*d2_xz + 2*d2_yz)

        qx = (y * nx) * invR; QX = np.fft.rfftn(qx, s=y.shape)
        qy = (y * ny) * invR; QY = np.fft.rfftn(qy, s=y.shape)
        qz = (y * nz) * invR; QZ = np.fft.rfftn(qz, s=y.shape)

        div_q = np.fft.irfftn(1j*KX*QX + 1j*KY*QY + 1j*KZ*QZ, s=y.shape).real
        term3 = -2.0 * self.f_grid * div_q

        coreT = b_bias*lap + term2 + term3
        return coreT / (self.aHf_grid + 1e-30)

    
#############################################
# -----------------------
# CG / PCG implementation
# -----------------------
def make_precond_Sphi_spectral(ops):
    n = ops.box.n
    def M(r):
        Rk = np.fft.rfftn(r)
        Zk = ops.Sphi_k * Rk
        Zk[0,0,0] = 0.0
        return np.fft.irfftn(Zk, s=(n,n,n)).real

    return M

def make_precond_psi_spectral(ops, sigma_x, M=None):
    """
    Diagonal-in-k preconditioner for the ψ-system (PP approximation).
    Uses the *galaxy* PP symbol: L_gal(k)=-(b k^2 + f k_z^2)/(aHf),
    consistent with bias only on the density term.
    """
    n = ops.box.n
    K2 = ops.K2
    KZ = ops.KZ
    a, H, f = ops.a, ops.H, ops.f

    b = float(ops.b_bias)  # stored in Operators now

    # mean weight <W_x> ~ < M / sigma^2 >
    sigma2 = np.asarray(sigma_x, float)**2
    W_x = 1.0 / (sigma2 + 1e-30)
    if M is not None:
        W_x *= np.asarray(M, float)
    Wbar = float(np.mean(W_x))

    # PP galaxy L(k) magnitude-squared
    Lgal = (b * K2 + f * (KZ**2)) / (a * H * f + 1e-30)
    lamL2 = (Lgal**2)  # sign irrelevant

    # approximate eigenvalues of A_psi(k)
    Apsi_diag = 1.0 + Wbar * lamL2 * ops.Sphi_k

    Apsi_diag = np.maximum(Apsi_diag, 1e-20)
    Apsi_diag[0, 0, 0] = 1.0  # keep DC benign

    def M_inv(r):
        Rk = np.fft.rfftn(r, s=(n, n, n))
        Zk = Rk / Apsi_diag
        Zk[0, 0, 0] = 0.0
        return np.fft.irfftn(Zk, s=(n, n, n)).real

    return M_inv


def make_precond_psi_kdiag(ops, b_bias, sigma_x, M=None, eps=1e-30):
    n = ops.box.n
    # assume sigma_x is scalar or nearly constant; otherwise use mean
    if np.isscalar(sigma_x):
        sigma2_eff = float(sigma_x)**2
    else:
        sigma2_eff = float(np.mean(np.asarray(sigma_x, float)**2))

    Pdelta = ops.Pdelta  # (n,n,n//2+1) on rFFT grid

    lam_k = 1.0 + (b_bias**2) * Pdelta / (sigma2_eff + eps)
    Minv_k = np.zeros_like(lam_k, dtype=float)
    np.divide(1.0, lam_k, out=Minv_k, where=(lam_k > 0.0))
    Minv_k[0, 0, 0] = 1.0  # don't touch DC

    def M(z):
        Zk = np.fft.rfftn(z)
        Zk *= Minv_k
        return np.fft.irfftn(Zk, s=z.shape).real

    return M



def pcg(apply_A, b, apply_Minv=None, rtol=1e-10, maxit=1000, verbose=True):
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

import numpy as np

def _make_Wx(sigma_x, M=None, eps=1e-30):
    sig2 = np.asarray(sigma_x, float)**2
    W = 1.0 / (sig2 + eps)
    if M is not None:
        W *= np.asarray(M, float)
    return W


def make_matvec_and_rhs_psi_radial_fft(
    ops,
    sigma_x,
    d,
    M=None,
    eps=0.0,
    include_geom=True,
):
    """
    Build (A·psi) and rhs for the ψ-based Wiener system in radial RSD (FFT branch):

        A_psi psi = psi + b^2 S_phi^{1/2} L^T W L S_phi^{1/2} psi
        rhs       = b S_phi^{1/2} L^T W d

    where:
      - L is the radial RSD operator on φ (ops.apply_L_rsd_radial_fft),
      - S_phi^{1/2} is implemented by ops.apply_Sphi_sqrt_fft,
      - W is a diagonal weight matrix in real space: W_x = mask / sigma_x^2.

    Parameters
    ----------
    ops : Operators
        Must provide:
          - apply_Sphi_sqrt_fft(psi)
          - apply_L_rsd_radial_fft(phi, include_geom=...)
          - apply_Lt_rsd_radial_fft(y, include_geom=...)
    sigma_x : float or (n,n,n)
        Noise RMS per voxel (or scalar).
    d : (n,n,n) array
        Observed data field (masked density contrast).
    M : (n,n,n) array or None
        Mask / completeness in [0,1]. If None, taken as 1 everywhere.
    eps : float
        Small regularization added to sigma_x^2 in the denominator.
    include_geom : bool
        Whether to include the geometric 2 v_r / r term in L and L^T.

    Returns
    -------
    apply_A : callable
        Function mapping psi -> A_psi psi.
    rhs : (n,n,n) array
        Right-hand side for the ψ system.
    """
    d = np.asarray(d, dtype=float)

    # Real-space weights W_x = M / (sigma^2 + eps)
    W_x = _make_Wx(sigma_x, M=M, eps=eps)

    def apply_A(psi):
        """
        A_psi psi = psi + b^2 S_phi^{1/2} D_phi L^T W L D_phi S_phi^{1/2} psi
        """
        # prior term (identity in ψ)
        y_prior = psi

        # 1) ψ -> φ_0 via S_φ^{1/2}   (reference-epoch φ)
        phi0 = ops.apply_Sphi_sqrt_fft(psi)

        # 2) evolve φ_0 -> φ_z via D_φ(x)
        phi_z = ops.apply_Dphi(phi0)

        # 3) φ_z -> data space via radial RSD operator
        yL = ops.apply_L_rsd_radial_fft(phi_z, include_geom=include_geom)

        # 4) apply weights W in real space
        WyL = W_x * yL

        # 5) L^T W yL back to φ-space
        if include_geom:
            LtWyL = ops.apply_Lt_rsd_radial_fft(WyL, include_geom=True)
        else:
            # if we ignore geom, L is self-adjoint: L^T = L
            LtWyL = ops.apply_L_rsd_radial_fft(WyL, include_geom=False)

        # 6) apply D_φ again on the back leg: D_φ L^T W L D_φ φ_0
        Dphi_LtWyL = ops.apply_Dphi(LtWyL)

        # 7) apply S_φ^{1/2} to go back to ψ-space
        SDphiLtWyL = ops.apply_Sphi_sqrt_fft(Dphi_LtWyL)

        # 8) multiply by b^2
        y_data = SDphiLtWyL

        return y_prior + y_data
    
    # Build RHS: rhs = b S_φ^{1/2} L^T W d
    rhs_core = W_x * d
    if include_geom:
        LtWd = ops.apply_Lt_rsd_radial_fft(rhs_core, include_geom=True)
    else:
        LtWd = ops.apply_L_rsd_radial_fft(rhs_core, include_geom=False)
    DphiLtWd = ops.apply_Dphi(LtWd)
    rhs =  ops.apply_Sphi_sqrt_fft(DphiLtWd)
    return apply_A, rhs


def _get_sigma_x(obs_data):
    # just return the field or scalar you stored above
    sig = getattr(obs_data, "sigma_noise", None)
    if sig is None:
        raise ValueError("obs_data.sigma_noise is None; generate noise first.")
    return sig


def Wiener_solve_psi_radial_fft(
    ops_fft: "Operators",
    obs_data,
    *,
    include_geom=True,
    rtol=1e-5,
    maxit=200,
    verbose=True,
    return_precond=False,
):
    """
    Wiener mean for radial RSD using ψ as the reconstruction variable.

    System in ψ-space:
        A_psi ψ = ψ + b^2 S_φ^{1/2} L^T W L S_φ^{1/2} ψ
                = b S_φ^{1/2} L^T W d

    where:
      - L is the radial RSD operator on φ (ops_fft.apply_L_rsd_radial_fft),
      - S_φ^{1/2} is implemented by ops_fft.apply_Sphi_sqrt_fft,
      - W is diagonal in real space: W_x = mask / sigma_x^2.

    Returns
    -------
    psi : (n,n,n) array
        Wiener mean in ψ-space.
    phi : (n,n,n) array
        Wiener mean in φ-space, phi = S_φ^{1/2} ψ.
    A : callable
        Matvec function for A_psi.
    rhs : (n,n,n) array
        Right-hand side b S_φ^{1/2} L^T W d.
    precond : callable, optional
        Preconditioner used in PCG (S_φ-like smoothing).
    """
    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)

    # Build A_psi and rhs for the current data
    A, rhs = make_matvec_and_rhs_psi_radial_fft(
        ops_fft,
        sigma_x=sigma_x,
        d=d,
        M=M,
        eps=1e-30,
        include_geom=include_geom,
    )

    # Preconditioner: spectral S_φ-like smoother (optional but helpful)
    precond = make_precond_psi_spectral(ops_fft, sigma_x, M)

    # Solve A_psi ψ = rhs with PCG
    t0 = time.perf_counter()
    psi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[ψ • Radial RSD • FFT] solve time: {time.perf_counter()-t0:.2f}s")

    # Map back to φ
    phi = ops_fft.apply_Sphi_sqrt_fft(psi)

    if return_precond:
        return psi, phi, A, rhs, precond
    else:
        return psi, phi, A, rhs


def Constrained_realization_psi_radial_fft(
    ops_fft: "Operators",
    obs_data,
    obs_geometry:Obs_Geometry,
    *,
    include_geom=True,
    rng=None,
    rtol=1e-5,
    maxit=200,
    verbose=False,
    reuse=None,   # optionally pass (A_psi, precond_psi) for many HR draws
):
    """
    Hoffman–Ribak constrained realization for radial RSD (FFT branch),
    using ψ as the solver variable but returning a constrained φ field.

    Steps:
      1) Draw a prior φ_rand ~ S_φ from your Data generator.
      2) Forward model to mock observation:
           y_rand = M [ b L φ_rand + n_rand ].
      3) Residual: r = d - y_rand.
      4) Solve A_psi ψ_corr = rhs_psi(r), then φ_corr = S_φ^{1/2} ψ_corr.
      5) Constrained realization: φ_CR = φ_rand + φ_corr.
    """
    from Data import Data  # if not already imported

    rng = np.random.default_rng() if rng is None else rng

    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)
    box     = obs_data.box
    cosmo   = obs_data.cosmology

    # 1) Prior draw: φ_rand from your existing Data generator
    truth = Data(box, cosmo)
    truth.generate_mock_fields(b_bias, obs_geometry, rng=rng)          # builds delta_r, phi_fft, etc.
    if getattr(truth, "phi_fft", None) is None:
        truth.calc_phi()
    phi_rand = truth.phi_0

    # 2) Mock observation: y_rand = M [ b L φ_rand + n_rand ]
    nshape = (box.n, box.n, box.n)
    if np.isscalar(sigma_x):
        n_rand = rng.normal(0.0, float(sigma_x), size=nshape)
    else:
        sig = np.asarray(sigma_x, float)
        n_rand = rng.normal(0.0, 1.0, size=sig.shape) * sig

    # y_rand = b_bias * ops_fft.apply_L_rsd_radial_fft(phi_rand, include_geom=include_geom) + n_rand
    # if M is not None:
    #     y_rand = np.asarray(M, float) * y_rand

    phi_z_rand = ops_fft.apply_Dphi(phi_rand)
    y_rand = ops_fft.apply_L_rsd_radial_fft(phi_z_rand, include_geom=include_geom) + n_rand
    if M is not None:
        y_rand = M * y_rand

    # 3) Residual between real data and mock data
    residual = d - y_rand

    # 4) Wiener correction in ψ-space using the SAME A_psi
    if reuse is not None:
        A_psi, precond_psi = reuse

        # Rebuild RHS for residual: rhs_psi = b S_φ^{1/2} L^T W residual
        eps   = 1e-30
        W_x   = _make_Wx(sigma_x, M=M, eps=eps)
        core  = W_x * residual
        if include_geom:
            LtWres = ops_fft.apply_Lt_rsd_radial_fft(core, include_geom=True)
        else:
            LtWres = ops_fft.apply_L_rsd_radial_fft(core, include_geom=False)
        rhs_psi = b_bias * ops_fft.apply_Sphi_sqrt_fft(ops_fft.apply_Dphi(LtWres))

        psi_corr = pcg(A_psi, rhs_psi, apply_Minv=precond_psi,
                       rtol=rtol, maxit=maxit, verbose=verbose)
    else:
        # Build A_psi and rhs_psi fresh for the residual
        A_psi, rhs_psi = make_matvec_and_rhs_psi_radial_fft(
            ops_fft,
            sigma_x=sigma_x,
            d=residual,   # <-- note: using residual instead of d
            M=M,
            eps=1e-30,
            include_geom=include_geom,
        )
        precond_psi = make_precond_psi_spectral(ops_fft, sigma_x, M)
        #precond_psi = make_precond_psi_kdiag(ops_fft, b_bias, sigma_x, M)
        psi_corr = pcg(A_psi, rhs_psi, apply_Minv=precond_psi,
                       rtol=rtol, maxit=maxit, verbose=verbose)

    # 5) Back to φ-space and form constrained realization
    phi_corr = ops_fft.apply_Sphi_sqrt_fft(psi_corr)
    phi_cr   = phi_rand + phi_corr

    return phi_cr
