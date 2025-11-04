from dataclasses import dataclass, field
import numpy as np
from helper_tools import kgrid_rfft3d, rfft_multiplicity_last_axis, spectral_d_dz, make_triangular_rays_mask
from Box import Box
from Cosmology import Cosmology
from pspectra import Pk_phys_nowiggle, Pk_phys_at_z_from_P0
from typing import Optional
from pspectra_camb import build_camb_pk_callable

@dataclass
class Data:
    box: Box
    cosmology: Cosmology
    delta_r: Optional[np.ndarray] = field(default=None, repr=False)
    phi_fft: Optional[np.ndarray]     = field(default=None, repr=False)
    phi_sten: Optional[np.ndarray]     = field(default=None, repr=False)
    v_fft: Optional[np.ndarray]     = field(default=None, repr=False)
    v_sten: Optional[np.ndarray]     = field(default=None, repr=False)
    delta_s_z: Optional[np.ndarray] = field(default=None, repr=False)
    delta_s_r: Optional[np.ndarray] = field(default=None, repr=False)
    
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


    def calc_lin_z_rsd_delta(self):
        a, H = self.cosmology.a, self.cosmology.H
        dx = self.box.dx
        vz = self.v_fft[2]
        disp_phys = vz / (a * H)            # Mpc/h
        d_vz_dz = spectral_d_dz(vz, dx)
        self.delta_s_z = self.delta_r - (1.0/(a*H)) * d_vz_dz
    
    def generate_mock_fields(self, rng=None, Pk_callable=None):
        self.sample_delta_from_Pk(rng, Pk_callable=Pk_callable)
        self.calc_phi()
        self.calc_lin_z_rsd_delta()

    