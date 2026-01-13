from dataclasses import dataclass, field
import numpy as np
from helper_tools import kgrid_rfft3d, rfft_multiplicity_last_axis, spectral_d_dz, make_triangular_rays_mask, cdiff4, grad4_scalar, radial_rsd_diagnostics, L_rsd_radial_fft_operator
from helper_tools import  div4_vector, los_unit_and_radius, radial_divergence_flux4, radial_divergence_identity4, divergence_of_radial_flux_highorder, radial_linear_rsd_highorder
from Box import Box
from Cosmology import Cosmology
from pspectra import Pk_phys_nowiggle, Pk_phys_at_z_from_P0
from typing import Optional
from pspectra_camb import build_camb_pk_callable
from Obs_Geometry import Obs_Geometry

@dataclass
class Data:
    box: Box
    cosmology: Cosmology
    delta_r: Optional[np.ndarray]    = field(default=None, repr=False)
    phi_fft: Optional[np.ndarray]    = field(default=None, repr=False)
    phi_0: Optional[np.ndarray]    = field(default=None, repr=False)
    phi_sten: Optional[np.ndarray]   = field(default=None, repr=False)
    v_fft: Optional[np.ndarray]      = field(default=None, repr=False)
    v_sten: Optional[np.ndarray]     = field(default=None, repr=False)
    delta_s_z: Optional[np.ndarray]  = field(default=None, repr=False)
    delta_s_r: Optional[np.ndarray]  = field(default=None, repr=False)
    psi: Optional[np.ndarray]        = field(default=None, repr=False)
    observer: Optional[np.ndarray]   = field(default=None, repr=False)
    b_bias = 1.7
    def sample_delta_from_Pk(self, rng=None, Pk_callable=None):
        if rng is None:
            rng = np.random.default_rng()
        n, N, V = self.box.n, self.box.N, self.box.V
        _, _, _, K, _ = kgrid_rfft3d(self.box)  # K in h/Mpc

        # 1) If no P(k) callable provided, build one from CAMB using your cosmology
        if Pk_callable is None:
            Om = getattr(self.cosmology, "Om", 0.315)
            Ob = getattr(self.cosmology, "Ob", 0.049)   # add to Cosmology if not present
            h  = getattr(self.cosmology, "h",  0.674)
            ns = getattr(self.cosmology, "ns", 0.965)
            s8 = getattr(self.cosmology, "sigma8", 0.811)
            print('Creating Pk callable')
            # kmax_h should comfortably cover your grid's max |k|
            kmax_h = float(K.max()) * 1.1
            Pk_callable, _ = build_camb_pk_callable(
                Om=Om, Ob=Ob, h=h, ns=ns,
                sigma8_target=s8,
                z=0.0,
                kmax_h=kmax_h,
                nonlinear=False
            )

        # 2) Evaluate the power on the rFFT grid
        Pk = Pk_callable(K)                      # shape (n, n, n//2+1), units (Mpc/h)^3

        # 3) Multiplicity for last axis in rFFT
        w_last = rfft_multiplicity_last_axis(n)[None, None, :]  # (1,1,n//2+1)

        #    Target variance per Fourier mode:
        #    E[|D_k|^2] = (N/V) * P(k) / w_k  (matches your pipeline)
        P_rfft = (N / V) * (Pk / w_last)
        P_rfft = np.maximum(P_rfft, 0.0)         # guard tiny negatives from rounding

        # 4) White Gaussian in real space -> rFFT
        xi = rng.normal(0.0, 1.0, size=(n, n, n))
        Xk = np.fft.rfftn(xi)                    # E[|Xk|^2] ∝ N

        # 5) Scale by sqrt of target variance
        Fk = np.sqrt(P_rfft, dtype=float) * Xk
        Fk[0, 0, 0] = 0.0                        # zero DC for safety

        # 6) Back to real space
        self.delta_r = np.fft.irfftn(Fk, s=(n, n, n)).real

    def calc_phi(self):
        """
        Compute φ and v from δ using both conventions and store:
        - self.phi_fft  : continuum/spectral inverse (k^2)
        - self.phi_sten : lattice/stencil inverse (tilde{k}^2)
        - self.v_fft    : velocity from spectral gradient (shape: 3,n,n,n)
        - self.v_sten   : velocity from forward-difference gradient (shape: 3,n,n,n)
        Assumes v = -∇φ and δ = -(∇^2 φ)/(a H f).
        """
        n = self.box.n
        dx = self.box.dx
        a, H, f = self.cosmology.a, self.cosmology.H, self.cosmology.f

        assert self.delta_r is not None and self.delta_r.shape == (n, n, n), \
            "delta_r must be a (n,n,n) array. Run your field generator first."

        # rFFT k-grids & FFT of δ
        KX, KY, KZ, K, K2 = kgrid_rfft3d(self.box)
        deltak = np.fft.rfftn(self.delta_r)

        # -------- spectral (continuum) φ:  φ_k = -(a H f) δ_k / k^2 --------
        invK2 = np.zeros_like(K2, dtype=float)
        np.divide(1.0, K2, out=invK2, where=(K2 > 0))
        phik_spec = -(a * H * f) * invK2 * deltak
        phik_spec[0, 0, 0] = 0.0
        self.phi_fft = np.fft.irfftn(phik_spec, s=(n, n, n)).real

        # velocities (spectral): v̂_i = -i k_i φ̂
        vxk = -1j * KX * phik_spec
        vyk = -1j * KY * phik_spec
        vzk = -1j * KZ * phik_spec
        vx  = np.fft.irfftn(vxk, s=(n, n, n)).real
        vy  = np.fft.irfftn(vyk, s=(n, n, n)).real
        vz  = np.fft.irfftn(vzk, s=(n, n, n)).real
        self.v_fft = np.stack([vx, vy, vz], axis=0)  # (3, n, n, n)

        # -------- stencil (lattice) φ:  φ_k = -(a H f) δ_k / \tilde{k}^2 --------
        KT2 = (2.0 / dx**2) * (3.0 - np.cos(KX * dx) - np.cos(KY * dx) - np.cos(KZ * dx))
        invKT2 = np.zeros_like(KT2, dtype=float)
        np.divide(1.0, KT2, out=invKT2, where=(KT2 > 0))
        phik_lat = -(a * H * f) * invKT2 * deltak
        phik_lat[0, 0, 0] = 0.0
        self.phi_sten = np.fft.irfftn(phik_lat, s=(n, n, n)).real

        # velocities (stencil): forward-difference gradient, v = -∇^+ φ
        def d_forward(u, axis, h):
            return (np.roll(u, -1, axis=axis) - u) / h

        vx_s = -d_forward(self.phi_sten, axis=0, h=dx)
        vy_s = -d_forward(self.phi_sten, axis=1, h=dx)
        vz_s = -d_forward(self.phi_sten, axis=2, h=dx)
        self.v_sten = np.stack([vx_s, vy_s, vz_s], axis=0)

    def calc_psi(self, Pk_callable=None):
        """
        Compute the whitened potential ψ such that, in Fourier space,
            φ̂(k) = sqrt(P_φ(k)) ψ̂(k),
        with
            P_φ(k) = (a H f)^2 P_δ(k) / k^4.

        Assumes:
          - self.phi_fft is already filled with φ(x) from calc_phi()
          - same cosmology as used to build P_δ
        Stores:
          - self.psi : ψ(x) in real space
        """
        if self.phi_fft is None:
            raise ValueError("phi_fft is None. Run calc_phi() first.")

        box = self.box
        cosmo = self.cosmology
        a, H, f = cosmo.a, cosmo.H, cosmo.f
        n = box.n
        dx3 = box.dx**3
        # --- k-grid and |k|^2 ---
        KX, KY, KZ, K, K2 = kgrid_rfft3d(box)

        # --- If no P(k) callable provided, build one as in sample_delta_from_Pk ---
        if Pk_callable is None:
            Om = getattr(cosmo, "Om", 0.315)
            Ob = getattr(cosmo, "Ob", 0.049)
            h  = getattr(cosmo, "h",  0.674)
            ns = getattr(cosmo, "ns", 0.965)
            s8 = getattr(cosmo, "sigma8", 0.811)
            print("calc_psi: building Pk_callable from CAMB")
            kmax_h = float(K.max()) * 1.1
            Pk_callable, _ = build_camb_pk_callable(
                Om=Om, Ob=Ob, h=h, ns=ns,
                sigma8_target=s8,
                z=0.0,
                kmax_h=kmax_h,
                nonlinear=False
            )

        # --- Matter power on the rFFT grid ---
        P_delta = Pk_callable(K)   # (Mpc/h)^3, same shape as K

        # --- φ power spectrum on the grid: P_φ(k) = (aHf)^2 P_δ / k^4 ---
        P_phi = np.zeros_like(P_delta, dtype=float)
        # avoid k=0 blow-up
        mask = (K2 > 0)
        P_phi[mask] = (a*H*f)**2 * P_delta[mask] / (K2[mask]**2)

        # --- φ̂(k) from φ(x) ---
        phik = np.fft.rfftn(self.phi_fft, s=(n, n, n))

        # --- Whiten: ψ̂ = φ̂ / sqrt(P_φ) ---
        w_last = rfft_multiplicity_last_axis(n)[None, None, :]  # shape (1,1,n//2+1)

        sqrt_inv = np.zeros_like(P_phi, dtype=float)
        np.sqrt(dx3 * w_last / P_phi, out=sqrt_inv, where=(P_phi > 0))

        psik = phik * sqrt_inv
        psik[0,0,0] = 0.0
        self.psi = np.fft.irfftn(psik, s=(n,n,n)).real



    def calc_lin_z_rsd_delta(self):
        a, H = self.cosmology.a, self.cosmology.H
        dx = self.box.dx
        vz = self.v_fft[2]
        disp_phys = vz / (a * H)            # Mpc/h
        d_vz_dz = spectral_d_dz(vz, dx)
        self.delta_s_z = self.delta_r - (1.0/(a*H)) * d_vz_dz

    def calc_lin_r_rsd_delta(self, nhat, r, a, H, f, b_bias):
        self.delta_s_r = L_rsd_radial_fft_operator(self.phi_fft, nhat, r, a, H, f, b_bias, self.box, include_geom=True)
    
    def evolve_phi_field(self, Dphi_grid):
        """
        Apply the velocity-potential growth factor D_φ(x) in real space:
            (D_φ u)(x) = D_φ(x) * u(x).

        Assumes Obs_Geometry has been attached and Dphi_grid initialized.
        """
        if Dphi_grid is None:
            raise ValueError("Dphi_grid is none, please pass a Dphi_grid.")
        Dφ = Dphi_grid
        # Broadcasting-safe: ensure same shape
        if Dφ.shape != self.phi_fft.shape:
            raise ValueError(f"Dphi_grid shape {Dφ.shape} does not match input {self.phi_fft.shape}")
        self.phi_0 = self.phi_fft
        self.phi_fft = Dφ * self.phi_fft

    def generate_mock_fields(self, b_bias, obs_geometry:Obs_Geometry, rng=None, Pk_callable=None):

        nhat = obs_geometry.nhat_grid
        r = obs_geometry.r_grid
        a = obs_geometry.a_grid
        f = obs_geometry.f_grid
        H = obs_geometry.H_grid
        Dphi_grid = obs_geometry.Dphi_grid

        self.sample_delta_from_Pk(rng, Pk_callable=Pk_callable)
        self.calc_phi()
        self.calc_psi()
        self.evolve_phi_field(Dphi_grid)
        self.calc_lin_z_rsd_delta()
        
        self.calc_lin_r_rsd_delta(nhat, r, a, H, f, b_bias)

    

    