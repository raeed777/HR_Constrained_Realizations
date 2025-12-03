import numpy as np
from Box import Box
from Cosmology import Cosmology
from helper_tools import kgrid_rfft3d, Pphi_from_Pdelta, los_unit_and_radius, d1_4, d2_4, d2_mixed_4
import pickle, time, copy



class Operators:
    def __init__(self, box: Box, cosmo: Cosmology, Pdelta_callable, use_lattice_in_fft=True,
                 # new: observer setup for radial RSD
                 radial_observer_offset_L=5.0,  # observer at center - offset*L * los_dir
                 radial_los_dir="z"):          # or a 3-vector):
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

        # Real-space stencil needs dx
        self._dx = dx

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

        # --- Radial geometry (use EXACT same helper you used to generate RSD) ---
        def _as_dir(dspec):
            if isinstance(dspec, str):
                return dict(x=np.array([1,0,0.], float),
                            y=np.array([0,1,0.], float),
                            z=np.array([0,0,1.], float))[dspec.lower()]
            v = np.asarray(dspec, float)
            return v / np.linalg.norm(v)

        # choose LOS base direction and observer position
        los_dir_vec = _as_dir(radial_los_dir)  # e.g., "z" or a 3-vector
        center = np.array([box.L/2, box.L/2, box.L/2], float)
        observer_xyz = center - radial_observer_offset_L * box.L * los_dir_vec

        # use the SAME routine as in your RSD generator
        nx, ny, nz, r = los_unit_and_radius(box, observer_xyz, periodic=False, pad=0)

        # store unit LOS and (optional) 1/r (single precision is fine)
        self._nhat = (nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32))
        self._invR = (1.0 / np.maximum(r, 1e-30)).astype(np.float32)

#############################################

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


#############################################

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
    
#############################################

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

#############################################

# ========= NEW: radial RSD apply (hybrid spectral) =========
    def apply_L_rsd_radial_fft(self, phi, include_geom=True):
        a, H, f = self.a, self.H, self.f
        KX, KY, KZ, K2 = self.KX, self.KY, self.KZ, self.K2

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

        inner = lap + f * dnn

        if include_geom:
            # v_r = -∂_n φ computed spectrally
            phi_x = np.fft.irfftn((1j*KX) * Phik, s=phi.shape).real
            phi_y = np.fft.irfftn((1j*KY) * Phik, s=phi.shape).real
            phi_z = np.fft.irfftn((1j*KZ) * Phik, s=phi.shape).real
            vr = -(nx*phi_x + ny*phi_y + nz*phi_z)
            inner = inner - 2.0 * f * vr * self._invR

        return inner / (a*H*f + 1e-30)
    
    def apply_Lt_rsd_radial_fft(self, y, include_geom=True):
        """
        Adjoint of the radial RSD operator under periodic BCs.

        If include_geom=False:
            L = -(∇² + f n_i n_j ∂i∂j)/(a H f) is self-adjoint ⇒ L^T = L.

        If include_geom=True:
            Core Cφ = ∇²φ + f (n_i n_j) ∂i∂j φ + 2 f (n·∇φ)/r
            Then L = -(1/(a H f)) C, and L^T y = -(1/(a H f)) C^T y with
                C^T y = ∇² y + f ∂i∂j( (n_i n_j) y ) - 2 f ∇·( (y/r) n )
        """
        a, H, f = self.a, self.H, self.f
        if not include_geom:
            # proper adjoint without the geometric term:
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

            term2 = self.f * (d2_xx + d2_yy + d2_zz + 2*d2_xy + 2*d2_xz + 2*d2_yz)
            return (lap + term2) / (self.a*self.H*self.f + 1e-30)



        KX, KY, KZ, K2 = self.KX, self.KY, self.KZ, self.K2
        nx, ny, nz = self._nhat
        invR = self._invR

        Yk  = np.fft.rfftn(y, s=y.shape)
        lap = np.fft.irfftn(-K2 * Yk, s=y.shape).real

        # f ∂i∂j( (n_i n_j) y )  via FFT on the product a_ij = (n_i n_j) y
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

        term2 = f * (d2_xx + d2_yy + d2_zz + 2*d2_xy + 2*d2_xz + 2*d2_yz)

        # - 2 f ∇·( (y/r) n )  via FFT divergence
        qx = (y * nx) * invR; QX = np.fft.rfftn(qx, s=y.shape)
        qy = (y * ny) * invR; QY = np.fft.rfftn(qy, s=y.shape)
        qz = (y * nz) * invR; QZ = np.fft.rfftn(qz, s=y.shape)

        div_q = np.fft.irfftn(1j*KX*QX + 1j*KY*QY + 1j*KZ*QZ, s=y.shape).real
        term3 = -2.0 * f * div_q

        coreT = lap + term2 + term3
        return coreT / (a*H*f + 1e-30)



    # ========= NEW: radial RSD apply (real-space stencil) =========
    def apply_L_rsd_radial_stencil(self, phi, include_geom=True):
        """
        4th-order stencil version of the radial RSD operator:
            δ_s = -(∇²φ + f ∂_n^2 φ)/(a H f)        [linear radial RSD]
        with optional geometric correction (include_geom=True):
            δ_s = -(∇²φ + f ∂_n^2 φ - 2 f v_r / r)/(a H f)
        where v_r = -∂_n φ and ∂_n = n_x ∂_x + n_y ∂_y + n_z ∂_z.

        Requires self._nhat = (nx, ny, nz) and self._invR = 1/r arrays
        defined on the same cell-centre grid as `phi`.
        """
        a, H, f = self.a, self.H, self.f
        dx = self._dx

        # 4th-order second derivatives (diagonal Hessian terms)
        Hxx = d2_4(phi, 0, dx)
        Hyy = d2_4(phi, 1, dx)
        Hzz = d2_4(phi, 2, dx)

        # Laplacian (4th-order)
        lap = Hxx + Hyy + Hzz

        # 4th-order mixed Hessian terms
        Hxy = d2_mixed_4(phi, 0, 1, dx)
        Hxz = d2_mixed_4(phi, 0, 2, dx)
        Hyz = d2_mixed_4(phi, 1, 2, dx)

        # Contract with n_i n_j
        nx, ny, nz = self._nhat
        dnn = (nx*nx)*Hxx + (ny*ny)*Hyy + (nz*nz)*Hzz \
            + 2.0*(nx*ny)*Hxy + 2.0*(nx*nz)*Hxz + 2.0*(ny*nz)*Hyz

        # Base linear mapping
        inner = lap + f * dnn

        if include_geom:
            # geometric correction: - 2 f v_r / r
            # v_r = -∂_n φ with 4th-order first derivatives
            phi_x = d1_4(phi, 0, dx)
            phi_y = d1_4(phi, 1, dx)
            phi_z = d1_4(phi, 2, dx)
            vr = -(nx*phi_x + ny*phi_y + nz*phi_z)          # v_r = -∂_n φ
            inner = inner - 2.0 * f * vr * self._invR       # subtract inside bracket

        return inner / (a*H*f + 1e-30)

    def apply_Lt_rsd_radial_stencil(self, y, include_geom=True):
        """
        4th-order real-space adjoint under periodic BCs.

        include_geom=False:
            C^T y = ∇² y + f ∂i∂j( (n_i n_j) y )
        include_geom=True:
            C^T y = ∇² y + f ∂i∂j( (n_i n_j) y ) - 2 f ∇·((y/r) n)

        L^T y = -(1/(a H f)) * C^T y
        """
        a, H, f = self.a, self.H, self.f
        dx = self._dx
        nx, ny, nz = self._nhat
        invR = self._invR

        # Laplacian (4th-order)
        Hxx = d2_4(y, 0, dx); Hyy = d2_4(y, 1, dx); Hzz = d2_4(y, 2, dx)
        lap = Hxx + Hyy + Hzz

        # f ∂i∂j( (n_i n_j) y )  with 4th-order second derivatives
        d2_xx = d2_4((nx*nx)*y, 0, dx)
        d2_yy = d2_4((ny*ny)*y, 1, dx)
        d2_zz = d2_4((nz*nz)*y, 2, dx)
        d2_xy = d2_mixed_4((nx*ny)*y, 0, 1, dx)
        d2_xz = d2_mixed_4((nx*nz)*y, 0, 2, dx)
        d2_yz = d2_mixed_4((ny*nz)*y, 1, 2, dx)
        term2 = f * (d2_xx + d2_yy + d2_zz + 2*d2_xy + 2*d2_xz + 2*d2_yz)

        if not include_geom:
            coreT = lap + term2
            return coreT / (a*H*f + 1e-30)

        # - 2 f ∇·( (y/r) n )  with 4th-order divergence
        qx = (y * nx) * invR
        qy = (y * ny) * invR
        qz = (y * nz) * invR
        div_q = d1_4(qx, 0, dx) + d1_4(qy, 1, dx) + d1_4(qz, 2, dx)
        term3 = -2.0 * f * div_q

        coreT = lap + term2 + term3
        return coreT / (a*H*f + 1e-30)


#############################################
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
# Build matvecs (A·x) & RHS for radial–RSD (spectral L)
# -----------------------
def make_matvec_and_rhs_radial_rsd_spectral(
    ops: "Operators", b_bias, sigma_x, d, M=None, eps=0.0, include_geom=True
):
    """
    A x = S^{-1}_phi x + b^2 L^T W L x
    rhs = b L^T W d
    L is the *radial* RSD operator (FFT branch). If include_geom=True,
    uses the true adjoint L^T; else L^T=L.
    """
    W_x = _make_Wx(sigma_x, M, eps)

    def apply_A(x):
        y_prior = ops.apply_Sphi_inv_fft(x)
        yL      = ops.apply_L_rsd_radial_fft(x, include_geom=include_geom)
        WyL     = W_x * yL
        LtWL    = (ops.apply_Lt_rsd_radial_fft(WyL, include_geom=include_geom)
                   if include_geom else
                   ops.apply_L_rsd_radial_fft(WyL, include_geom=False))
        return y_prior + (b_bias**2) * LtWL

    rhs_core = W_x * d
    rhs = b_bias * (ops.apply_Lt_rsd_radial_fft(rhs_core, include_geom=include_geom)
                    if include_geom else
                    ops.apply_L_rsd_radial_fft(rhs_core, include_geom=False))
    return apply_A, rhs

# -----------------------
# Build matvecs (A·x) & RHS for radial–RSD (stencil L)
# -----------------------
def make_matvec_and_rhs_radial_rsd_stencil(
    ops: "Operators", b_bias, sigma_x, d, M=None, eps=0.0, include_geom=True
):
    """
    Same as above, but 4th-order real-space stencil branch.
    """
    W_x = _make_Wx(sigma_x, M, eps)

    def apply_A(x):
        y_prior = ops.apply_Sphi_inv_stencil(x)
        yL      = ops.apply_L_rsd_radial_stencil(x, include_geom=include_geom)
        WyL     = W_x * yL
        LtWL    = (ops.apply_Lt_rsd_radial_stencil(WyL, include_geom=include_geom)
                   if include_geom else
                   ops.apply_L_rsd_radial_stencil(WyL, include_geom=False))
        return y_prior + (b_bias**2) * LtWL

    rhs_core = W_x * d
    rhs = b_bias * (ops.apply_Lt_rsd_radial_stencil(rhs_core, include_geom=include_geom)
                    if include_geom else
                    ops.apply_L_rsd_radial_stencil(rhs_core, include_geom=False))
    return apply_A, rhs


# -----------------------
# Build matvecs (A·x) & RHS for PP–RSD (spectral L)
# -----------------------
def make_matvec_and_rhs_pp_rsd_spectral(ops: Operators, b_bias, sigma_x, d, M=None, eps=0.0):
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
def make_matvec_and_rhs_pp_rsd_stencil(ops: Operators, b_bias, sigma_x, d, M=None, eps=0.0):
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

#############################  real space solvers ##############################
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


################################ pp rsd solver ####################################
# --- Wiener mean (z rsd, spectral L) ---
def Wiener_solve_pp_rsd_fft(
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
    A, rhs  = make_matvec_and_rhs_pp_rsd_spectral(ops_fft, b_bias=b_bias, sigma_x=sigma_x, d=d, M=M)

    t0 = time.perf_counter()
    phi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[Spectral L] total solve time: {time.perf_counter()-t0:.2f}s")

    return (phi, A, rhs, precond) if return_precond else (phi, A, rhs)


# --- One HR constrained realization (REAL space, spectral L) ---
def Constrained_realization_pp_rsd_fft(
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
    y_rand = b_bias * ops_fft.apply_L_rsd_pp_fft(phi_rand) + n_rand
    if M is not None:
        y_rand = np.asarray(M, float) * y_rand

    # 3) residual and Wiener correction with the SAME system
    residual = d - y_rand

    if reuse is not None:
        A_fft, precond_fft = reuse
        # Rebuild the RHS for the new residual directly (same as your builder)
        eps = 1e-30
        W_x = (np.asarray(M, float) if M is not None else 1.0) / (np.asarray(sigma_x, float)**2 + eps)
        rhs_fft = b_bias * ops_fft.apply_L_rsd_pp_fft(W_x * residual)
        phi_corr = pcg(A_fft, rhs_fft, apply_Minv=precond_fft,
                       rtol=rtol, maxit=maxit, verbose=verbose)
    else:
        phi_corr, A_fft, rhs_fft = Wiener_solve_pp_rsd_fft(
            ops_fft, obs_data=obs_data.__class__(**{**obs_data.__dict__, "d": residual}), rtol=rtol, maxit=maxit, verbose=verbose
        )
        # ^ light trick: reuse same builder by passing a shallow copy with d=residual

    # 4) constrained realization
    return phi_rand + phi_corr

# --- Wiener mean (z rsd, stencil L) ---
def Wiener_solve_pp_rsd_stencils(
    ops_fft: Operators,
    obs_data: ObservedData,
    rtol=1e-6, maxit=300, verbose=True, return_precond=False
):
    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)

    precond = make_precond_Sphi_stencil(ops_fft)
    # If you named the builder differently, adjust here:
    A, rhs  = make_matvec_and_rhs_pp_rsd_stencil(ops_fft, b_bias=b_bias, sigma_x=sigma_x, d=d, M=M)

    t0 = time.perf_counter()
    phi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[Spectral L] total solve time: {time.perf_counter()-t0:.2f}s")

    return (phi, A, rhs, precond) if return_precond else (phi, A, rhs)


# --- One HR constrained realization (REAL space, spectral L) ---
def Constrained_realization_pp_rsd_stencils(
    ops_sten: Operators,
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
    if getattr(truth, "phi_sten", None) is None:
        if hasattr(truth, "calc_phi"):
            truth.calc_phi()                     # fills phi_fft & phi_sten
        else:
            raise AttributeError("Data lacks calc_phi(); needed to compute phi_fft.")
    phi_rand = truth.phi_sten

    # 2) mock observation with spectral forward model and fresh noise
    nshape = (box.n, box.n, box.n)
    if np.isscalar(sigma_x):
        n_rand = rng.normal(0.0, float(sigma_x), size=nshape)
    else:
        sig = np.asarray(sigma_x, float)
        n_rand = rng.normal(0.0, 1.0, size=sig.shape) * sig

    # y_rand = M [ b L φ_rand + n_rand ]
    y_rand = b_bias * ops_sten.apply_L_rsd_pp_stencil(phi_rand) + n_rand
    if M is not None:
        y_rand = np.asarray(M, float) * y_rand

    # 3) residual and Wiener correction with the SAME system
    residual = d - y_rand

    if reuse is not None:
        A_sten, precond_sten = reuse
        # Rebuild the RHS for the new residual directly (same as your builder)
        eps = 1e-30
        W_x = (np.asarray(M, float) if M is not None else 1.0) / (np.asarray(sigma_x, float)**2 + eps)
        rhs_sten = b_bias * ops_sten.apply_L_rsd_pp_stencil(W_x * residual)
        phi_corr = pcg(A_sten, rhs_sten, apply_Minv=precond_sten,
                       rtol=rtol, maxit=maxit, verbose=verbose)
    else:
        phi_corr, A_sten, rhs_sten = Wiener_solve_pp_rsd_stencils(
            ops_sten, obs_data=obs_data.__class__(**{**obs_data.__dict__, "d": residual}), rtol=rtol, maxit=maxit, verbose=verbose
        )
        # ^ light trick: reuse same builder by passing a shallow copy with d=residual

    # 4) constrained realization
    return phi_rand + phi_corr


############################## radial rsd solver #############################################

# -----------------------
# Helpers
# -----------------------
def _get_sigma_x(obs_data):
    sig = getattr(obs_data, "sigma_noise", None)
    if sig is None:
        raise ValueError("obs_data.sigma_noise is missing.")
    return sig

def _clone_with_d(obs_data, new_d):
    """
    Shallow clone of obs_data that preserves its class and attributes
    without calling __init__, then replaces only `.d`.
    Avoids constructor signature mismatches.
    """
    clone = copy.copy(obs_data)
    # ensure we don't alias the big array inadvertently
    clone.d = np.asarray(new_d, dtype=float).copy()
    return clone

# =======================
# Radial RSD — FFT branch
# =======================
def Wiener_solve_radial_rsd_fft(
    ops_fft: "Operators",
    obs_data,
    *,
    include_geom=True,
    rtol=1e-6, maxit=300, verbose=True, return_precond=False
):
    """
    Wiener mean for radial RSD using the spectral (FFT) operator.

    A x = S_phi^{-1} x + b^2 L^T W L x
    rhs = b L^T W d

    If include_geom=True, uses the true adjoint L^T; else L^T=L.
    """
    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)

    precond = make_precond_Sphi_spectral(ops_fft)

    # builder uses the correct adjoint internally
    A, rhs  = make_matvec_and_rhs_radial_rsd_spectral(
        ops_fft, b_bias=b_bias, sigma_x=sigma_x, d=d, M=M, include_geom=include_geom
    )

    t0 = time.perf_counter()
    phi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[Radial RSD • FFT] solve time: {time.perf_counter()-t0:.2f}s")

    return (phi, A, rhs, precond) if return_precond else (phi, A, rhs)


def Constrained_realization_radial_rsd_fft(
    ops_fft: "Operators",
    obs_data,
    *,
    include_geom=True,
    rng=None, rtol=1e-6, maxit=300, verbose=False,
    reuse=None   # optionally pass (A_fft, precond_fft) for many HR draws
):
    """
    Hoffman–Ribak constrained realization for radial RSD (FFT branch).
    """
    rng = np.random.default_rng() if rng is None else rng

    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)
    box     = obs_data.box
    cosmo   = obs_data.cosmology

    # 1) prior draw (ensure phi_fft exists)
    truth = Data(box, cosmo)
    truth.generate_mock_fields(rng=rng)
    if getattr(truth, "phi_fft", None) is None:
        truth.calc_phi()
    phi_rand = truth.phi_fft

    # 2) mock observation: y_rand = M [ b L φ_rand + n_rand ]
    nshape = (box.n, box.n, box.n)
    if np.isscalar(sigma_x):
        n_rand = rng.normal(0.0, float(sigma_x), size=nshape)
    else:
        sig = np.asarray(sigma_x, float)
        n_rand = rng.normal(0.0, 1.0, size=sig.shape) * sig

    y_rand = b_bias * ops_fft.apply_L_rsd_radial_fft(phi_rand, include_geom=include_geom) + n_rand
    if M is not None:
        y_rand = np.asarray(M, float) * y_rand

    # 3) residual and Wiener correction with the SAME system
    residual = d - y_rand

    if reuse is not None:
        A_fft, precond_fft = reuse
        # rebuild RHS: rhs = b L^T (W * residual)
        eps = 1e-30
        W_x = _make_Wx(sigma_x, M=M, eps=eps)
        rhs_fft = b_bias * (
            ops_fft.apply_Lt_rsd_radial_fft(W_x * residual, include_geom=include_geom)
            if include_geom else
            ops_fft.apply_L_rsd_radial_fft(W_x * residual, include_geom=False)
        )
        phi_corr = pcg(A_fft, rhs_fft, apply_Minv=precond_fft,
                       rtol=rtol, maxit=maxit, verbose=verbose)
    else:
        # call the same Wiener solver on a shallow copy with d = residual
        phi_corr, A_fft, rhs_fft = Wiener_solve_radial_rsd_fft(
            ops_fft, _clone_with_d(obs_data, residual),
            include_geom=include_geom, rtol=rtol, maxit=maxit, verbose=verbose
        )

    # 4) constrained realization
    return phi_rand + phi_corr


# =========================
# Radial RSD — Stencil branch
# =========================
def Wiener_solve_radial_rsd_stencil(
    ops_sten: "Operators",
    obs_data,
    *,
    include_geom=True,
    rtol=1e-6, maxit=300, verbose=True, return_precond=False
):
    """
    Wiener mean for radial RSD using the 4th-order stencil operator.
    """
    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)

    precond = make_precond_Sphi_stencil(ops_sten)

    A, rhs  = make_matvec_and_rhs_radial_rsd_stencil(
        ops_sten, b_bias=b_bias, sigma_x=sigma_x, d=d, M=M, include_geom=include_geom
    )

    t0 = time.perf_counter()
    phi = pcg(A, rhs, apply_Minv=precond, rtol=rtol, maxit=maxit, verbose=verbose)
    if verbose:
        print(f"[Radial RSD • Stencil] solve time: {time.perf_counter()-t0:.2f}s")

    return (phi, A, rhs, precond) if return_precond else (phi, A, rhs)


def Constrained_realization_radial_rsd_stencil(
    ops_sten: "Operators",
    obs_data,
    *,
    include_geom=True,
    rng=None, rtol=1e-6, maxit=300, verbose=False,
    reuse=None
):
    """
    Hoffman–Ribak constrained realization for radial RSD (stencil branch).
    """
    rng = np.random.default_rng() if rng is None else rng

    b_bias  = float(obs_data.b_bias)
    sigma_x = _get_sigma_x(obs_data)
    d       = obs_data.d
    M       = getattr(obs_data, "mask", None)
    box     = obs_data.box
    cosmo   = obs_data.cosmology

    # 1) prior draw (ensure phi_sten exists)
    truth = Data(box, cosmo)
    truth.generate_mock_fields(rng=rng)
    if getattr(truth, "phi_sten", None) is None:
        truth.calc_phi()
    phi_rand = truth.phi_sten

    # 2) mock observation: y_rand = M [ b L φ_rand + n_rand ]
    nshape = (box.n, box.n, box.n)
    if np.isscalar(sigma_x):
        n_rand = rng.normal(0.0, float(sigma_x), size=nshape)
    else:
        sig = np.asarray(sigma_x, float)
        n_rand = rng.normal(0.0, 1.0, size=sig.shape) * sig

    y_rand = b_bias * ops_sten.apply_L_rsd_radial_stencil(phi_rand, include_geom=include_geom) + n_rand
    if M is not None:
        y_rand = np.asarray(M, float) * y_rand

    # 3) residual and Wiener correction
    residual = d - y_rand

    if reuse is not None:
        A_sten, precond_sten = reuse
        eps = 1e-30
        W_x = _make_Wx(sigma_x, M=M, eps=eps)
        rhs_sten = b_bias * (
            ops_sten.apply_Lt_rsd_radial_stencil(W_x * residual, include_geom=include_geom)
            if include_geom else
            ops_sten.apply_L_rsd_radial_stencil(W_x * residual, include_geom=False)
        )
        phi_corr = pcg(A_sten, rhs_sten, apply_Minv=precond_sten,
                       rtol=rtol, maxit=maxit, verbose=verbose)
    else:
        phi_corr, A_sten, rhs_sten = Wiener_solve_radial_rsd_stencil(
            ops_sten, _clone_with_d(obs_data, residual),
            include_geom=include_geom, rtol=rtol, maxit=maxit, verbose=verbose
        )

    # 4) constrained realization
    return phi_rand + phi_corr