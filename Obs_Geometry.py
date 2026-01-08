from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from colossus.cosmology import cosmology as col_cosmo
from Box import Box
from helper_tools import los_unit_and_radius


def _growth_factor(col, z, z_norm=0.0):
    """
    D(z) normalized to D(z_norm) = 1.
    """
    D_znorm = col.growthFactor(z_norm)
    D_z     = col.growthFactor(z)
    return D_z / D_znorm


def _growth_rate(col, z):
    """
    f(z) ≈ Ω_m(z)^γ with γ ≃ 0.545.
    """
    Omz   = col.Om(z)
    gamma = 0.545
    return Omz**gamma


@dataclass
class Obs_Geometry:
    box: Box
    observer_xyz: Optional[np.ndarray] = None
    z_ref: float = 0.0   # redshift at which your fields are defined (usually 0)

    r_grid:    Optional[np.ndarray] = field(default=None, repr=False)
    nhat_grid: Optional[np.ndarray] = field(default=None, repr=False)
    z_grid:    Optional[np.ndarray] = field(default=None, repr=False)
    D_grid:    Optional[np.ndarray] = field(default=None, repr=False)
    f_grid:    Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------ Observer placement ------------------ #
    def set_observer(self, offset_L=5.0, los_dir="z"):
        """
        Put the observer at:
          x_obs = box_center - offset_L * L * los_dir_hat
        """
        def _as_dir(dspec):
            if isinstance(dspec, str):
                return dict(
                    x=np.array([1., 0., 0.]),
                    y=np.array([0., 1., 0.]),
                    z=np.array([0., 0., 1.]),
                )[dspec.lower()]
            v = np.asarray(dspec, float)
            return v / np.linalg.norm(v)

        los_dir_vec = _as_dir(los_dir)
        center = np.array([self.box.L/2, self.box.L/2, self.box.L/2], float)
        self.observer_xyz = center - offset_L * self.box.L * los_dir_vec

    # ------------------ Geometry: r, n̂ ------------------ #
    def compute_r_grid(self):
        """
        Compute n̂(x) and r(x) on the grid.
        """
        if self.observer_xyz is None:
            raise ValueError("observer_xyz not set. Call set_observer() first.")

        nx, ny, nz, r = los_unit_and_radius(
            self.box, self.observer_xyz,
            periodic=True, pad=0
        )
        self.nhat_grid = (nx, ny, nz)
        self.r_grid    = r

    # ------------------ z(r) inversion ------------------ #
    def build_z_of_r_interpolator(self, z_max=4.0, Nz=2048):
        """
        Build z_of_r(r) by tabulating comovingDistance and inverting it.

        z_max must be large enough that χ(z_max) > max(r_grid).
        """
        col = col_cosmo.getCurrent()  # assumes setCosmology('planck18') already called

        # 1) Sample z and compute χ(z) *safely* (scalar loop)
        z_samples  = np.linspace(0.0, z_max, Nz)
        chi_samples = np.empty_like(z_samples)

        for i, zi in enumerate(z_samples):
            # Most robust Colossus call is usually comovingDistance(z1, z2)
            # Distance from 0 -> zi:
            chi_samples[i] = col.comovingDistance(0.0, zi)

        # 2) Make sure z_max is big enough for your box
        r_max = float(self.r_grid.max()) if self.r_grid is not None else 0.0
        if r_max > chi_samples.max():
            raise ValueError(
                f"z_max={z_max} not large enough: "
                f"r_max={r_max:.3f} > chi(z_max)={chi_samples.max():.3f}. "
                "Increase z_max."
            )

        # 3) Build interpolator r -> z
        def z_of_r(r):
            r_arr = np.asarray(r, float)
            return np.interp(r_arr, chi_samples, z_samples)

        return z_of_r

    def compute_z_grid(self, z_max=4.0, Nz=2048):
        """
        Compute and store z(x) on the grid via the inverse χ(z).
        """
        if self.r_grid is None:
            raise ValueError("r_grid not set. Call compute_r_grid() first.")

        z_of_r = self.build_z_of_r_interpolator(z_max=z_max, Nz=Nz)
        self.z_grid = z_of_r(self.r_grid)

    # ------------------ D(z) and f(z) on the grid ------------------ #
    def compute_D_grid(self, z_max=4.0, Nz=2048):
        """
        Compute growth factor D(z(x))/D(z_ref) on the box grid.
        """
        if self.r_grid is None:
            raise ValueError("r_grid not set. Call compute_r_grid() first.")

        col = col_cosmo.getCurrent()

        if self.z_grid is None:
            self.compute_z_grid(z_max=z_max, Nz=Nz)

        z_flat = self.z_grid.ravel()
        D_flat = _growth_factor(col, z_flat, z_norm=self.z_ref)
        self.D_grid = D_flat.reshape(self.r_grid.shape)

    def compute_f_grid(self, z_max=4.0, Nz=2048):
        """
        Compute growth rate f(z(x)) ≈ Ω_m(z)^γ on the box grid.
        """
        if self.r_grid is None:
            raise ValueError("r_grid not set. Call compute_r_grid() first.")

        col = col_cosmo.getCurrent()

        if self.z_grid is None:
            self.compute_z_grid(z_max=z_max, Nz=Nz)

        z_flat = self.z_grid.ravel()
        f_flat = _growth_rate(col, z_flat)
        self.f_grid = f_flat.reshape(self.r_grid.shape)

    def compute_Dphi_grid(self):
        """
        Build the grid of velocity–potential growth factors D_φ(r) from the
        density growth grid D_δ(r) using

            D_φ(z) = [a(z) H(z) f(z) / (a0 H0 f0)] * D_δ(z),

        with the same normalization convention as D_δ:
            D_δ(z_ref) = 1  ⇒  D_φ(z_ref) = 1.
        """
        if self.z_grid is None:
            raise ValueError("z_grid not set. Call compute_z_grid() first.")
        if self.D_grid is None:
            raise ValueError("D_grid not set. Call compute_D_grid() first.")
        if self.f_grid is None:
            raise ValueError("f_grid not set. Call compute_f_grid() first.")

        # Colossus cosmology currently in use (global)
        col = col_cosmo.getCurrent()

        z   = np.asarray(self.z_grid, float)   # shape (n,n,n)
        Dδ  = np.asarray(self.D_grid, float)   # D_δ(z) / D_δ(z_ref)
        f_z = np.asarray(self.f_grid, float)   # f(z)

        # a(z) and H(z)
        a_z = 1.0 / (1.0 + z)
        H_z = col.Hz(z)                        # vectorized in Colossus

        # reference values at z_ref
        z0 = float(self.z_ref)
        a0 = 1.0 / (1.0 + z0)
        H0 = col.Hz(z0)
        f0 = _growth_rate(col, z0)              # your f(z) ≈ Ω_m(z)^γ

        # ratio R(z) = (a H f) / (a0 H0 f0)
        eps = 1e-30
        R_z = (a_z * H_z * f_z) / (a0 * H0 * f0 + eps)

        # final D_φ grid, normalized so D_φ(z_ref) = 1
        self.Dphi_grid = R_z * Dδ


    # ------------------ Convenience wrapper ------------------ #
    def initialize(self, z_max=4.0, Nz=2048):
        """
        One-shot: r, z, D, f on the grid.
        """
        self.compute_r_grid()
        self.compute_z_grid(z_max=z_max, Nz=Nz)
        self.compute_D_grid(z_max=z_max, Nz=Nz)
        self.compute_f_grid(z_max=z_max, Nz=Nz)
        self.compute_Dphi_grid()
