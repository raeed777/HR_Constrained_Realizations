import numpy as np
from Box import Box
from Cosmology import Cosmology
from helper_tools import kgrid_rfft3d, Pphi_from_Pdelta, los_unit_and_radius

class Operators:
    """
    Spectral operators for ψ / φ reconstruction:
      - builds P_δ(k) and P_φ(k)
      - stores S_φ^{-1}(k) (for φ-based solvers) and S_φ^{1/2}(k) (for ψ-based solvers)
      - provides k-grids and radial LOS geometry
      - defines spectral φ→δ and φ→δ_s (PP) symbols
    """

    def __init__(self, box: Box, cosmo: Cosmology, Pdelta_callable,
                 # radial RSD geometry
                 radial_observer_offset_L=5.0,   # observer at center - offset*L * los_dir
                 radial_los_dir="z"):           # or a 3-vector
        a, H, f = cosmo.a, cosmo.H, cosmo.f
        self.box = box
        self.cosmology = cosmo
        self.a, self.H, self.f = a, H, f

        # --- k-grid for rFFT ---
        self.KX, self.KY, self.KZ, self.K, self.K2 = kgrid_rfft3d(box)

        # --- Prior spectra: P_δ(k) and P_φ(k) ---
        self.Pdelta = Pdelta_callable(self.K)          # (Mpc/h)^3
        self.Pphi   = Pphi_from_Pdelta(self.K, self.Pdelta, a, H, f)

        # --- S_φ^{-1}(k) = dx^3 / P_φ(k)  (for φ-based solvers) ---
        dx3 = box.dx**3
        self.Sphi_inv_k = np.zeros_like(self.Pphi, dtype=float)
        np.divide(dx3, self.Pphi, out=self.Sphi_inv_k, where=(self.Pphi > 0.0))
        self.Sphi_inv_k[0, 0, 0] = 0.0   # no DC prior

        # --- S_φ^{1/2}(k) = sqrt(P_φ(k)) (for ψ-based solvers: φ̂ = S_φ^{1/2} ψ̂) ---
        self.Sphi_sqrt_k = np.zeros_like(self.Pphi, dtype=float)
        np.sqrt(self.Pphi, out=self.Sphi_sqrt_k, where=(self.Pphi > 0.0))
        self.Sphi_sqrt_k[0, 0, 0] = 0.0

        # Optional: S_φ^{-1/2}(k) for diagnostics / φ→ψ
        self.Sphi_sqrt_inv_k = np.zeros_like(self.Pphi, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.Sphi_sqrt_inv_k = np.where(self.Sphi_sqrt_k > 0.0,
                                            1.0 / self.Sphi_sqrt_k,
                                            0.0)
        self.Sphi_sqrt_inv_k[0, 0, 0] = 0.0

        # --- φ -> δ and φ -> δ_s (PP) symbols, purely spectral ---
        # real-space density: δ = -(k^2/(a H f)) φ
        self.Lk_real = -(self.K2) / (a * H * f + 1e-30)
        self.Lk_real[0, 0, 0] = 0.0

        # plane-parallel RSD along z: δ_s = -(k^2 + f k_z^2)/(a H f) φ
        self.Lk_pp = -(self.K2 + f * (self.KZ**2)) / (a * H * f + 1e-30)
        self.Lk_pp[0, 0, 0] = 0.0

        # --- Radial geometry (same helper as in your RSD generator) ---
        def _as_dir(dspec):
            if isinstance(dspec, str):
                return dict(
                    x=np.array([1, 0, 0.], float),
                    y=np.array([0, 1, 0.], float),
                    z=np.array([0, 0, 1.], float),
                )[dspec.lower()]
            v = np.asarray(dspec, float)
            return v / np.linalg.norm(v)

        los_dir_vec = _as_dir(radial_los_dir)  # e.g., "z" or a 3-vector
        center = np.array([box.L/2, box.L/2, box.L/2], float)
        observer_xyz = center - radial_observer_offset_L * box.L * los_dir_vec

        nx, ny, nz, r = los_unit_and_radius(box, observer_xyz, periodic=True, pad=0)

        # Store LOS unit vector and 1/r on the grid
        self._nhat = (nx.astype(np.float32),
                      ny.astype(np.float32),
                      nz.astype(np.float32))
        self._invR = (1.0 / np.maximum(r, 1e-30)).astype(np.float32)

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
        where S_φ^{1/2}(k) = sqrt(P_φ(k)).
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
