from dataclasses import dataclass, field
from typing import Optional, Literal, Dict
import numpy as np
from Box import Box
from Cosmology import Cosmology
from helper_tools import kgrid_rfft3d

Array = np.ndarray

def delta_from_phi(phi, box, a, H, f, mode="stencil"):
    n = box.n; dx = box.dx
    assert phi is not None, "delta_from_phi: 'phi' is None"
    phi = np.asarray(phi)
    assert phi.shape == (n, n, n), f"delta_from_phi: expected phi shape {(n,n,n)}, got {phi.shape}"

    KX, KY, KZ, K, K2 = kgrid_rfft3d(box)

    if mode == "stencil":
        KT2 = (2.0/dx**2) * (3.0 - np.cos(KX*dx) - np.cos(KY*dx) - np.cos(KZ*dx))
        Lk = -KT2 / (a*H*f + 1e-300)
    elif mode == "spectral":
        Lk = -K2  / (a*H*f + 1e-300)
    else:
        raise ValueError("mode must be 'stencil' or 'spectral'")

    Phik = np.fft.rfftn(phi, s=(n, n, n))
    Dk   = Lk * Phik
    Dk[0,0,0] = 0.0
    return np.fft.irfftn(Dk, s=(n, n, n)).real

def velocity_from_phi(phi: Array, box: Box, mode: str = "stencil") -> Optional[Array]:
    """
    Compute v = -∇φ on an n^3 periodic grid.
      - spectral: v̂_i(k) = -i k_i φ̂(k)
      - stencil : central differences (f_{+1}-f_{-1})/(2Δx)
    Returns (3, n, n, n) array or None if phi is None.
    """
    if phi is None:
        # silent None passthrough so callers can guard easily
        return None

    n, dx = box.n, box.dx

    def vel_from_phi_stencil(phi_local: Array) -> Array:
        inv2dx = 1.0 / (2.0 * dx)
        vx = -(np.roll(phi_local, -1, axis=0) - np.roll(phi_local,  1, axis=0)) * inv2dx
        vy = -(np.roll(phi_local, -1, axis=1) - np.roll(phi_local,  1, axis=1)) * inv2dx
        vz = -(np.roll(phi_local, -1, axis=2) - np.roll(phi_local,  1, axis=2)) * inv2dx
        return np.stack([vx, vy, vz], axis=0)

    if mode == "stencil":
        return vel_from_phi_stencil(phi)

    elif mode == "spectral":
        KX, KY, KZ, _, _ = kgrid_rfft3d(box)
        Phik = np.fft.rfftn(phi)
        vx = np.fft.irfftn(-1j * KX * Phik, s=(n, n, n)).real
        vy = np.fft.irfftn(-1j * KY * Phik, s=(n, n, n)).real
        vz = np.fft.irfftn(-1j * KZ * Phik, s=(n, n, n)).real
        return np.stack([vx, vy, vz], axis=0)

    else:
        raise ValueError("mode must be 'stencil' or 'spectral'")

@dataclass
class ReconstructedField:
    box: Box
    cosmology: Cosmology

    # φ (reconstructed)
    Wiener_phi_rec_sten: Optional[Array] = field(default=None, repr=False)
    Wiener_phi_rec_fft:  Optional[Array] = field(default=None, repr=False)
    constrained_realization_phi_sten: Optional[Array] = field(default=None, repr=False)
    constrained_realization_phi_fft:  Optional[Array] = field(default=None, repr=False)

    # δ (reconstructed)
    Wiener_delta_rec_sten: Optional[Array] = field(default=None, repr=False)
    Wiener_delta_rec_fft:  Optional[Array] = field(default=None, repr=False)
    constrained_realization_delta_sten: Optional[Array] = field(default=None, repr=False)
    constrained_realization_delta_fft:  Optional[Array] = field(default=None, repr=False)

    # v (reconstructed) -> (3, n, n, n)
    Wiener_vel_rec_sten: Optional[Array] = field(default=None, repr=False)
    Wiener_vel_rec_fft:  Optional[Array] = field(default=None, repr=False)
    constrained_realization_vel_sten: Optional[Array] = field(default=None, repr=False)
    constrained_realization_vel_fft:  Optional[Array] = field(default=None, repr=False)

    def calc_delta(self):
        a, H, f = self.cosmology.a, self.cosmology.H, self.cosmology.f
        if self.Wiener_phi_rec_sten is not None:
            self.Wiener_delta_rec_sten = delta_from_phi(self.Wiener_phi_rec_sten, self.box, a, H, f, mode="stencil")
        if self.Wiener_phi_rec_fft is not None:
            self.Wiener_delta_rec_fft  = delta_from_phi(self.Wiener_phi_rec_fft,  self.box, a, H, f, mode="spectral")
        if self.constrained_realization_phi_sten is not None:
            self.constrained_realization_delta_sten = delta_from_phi(self.constrained_realization_phi_sten, self.box, a, H, f, mode="stencil")
        if self.constrained_realization_phi_fft is not None:
            self.constrained_realization_delta_fft  = delta_from_phi(self.constrained_realization_phi_fft,  self.box, a, H, f, mode="spectral")

    def calc_vel(self):
        # Wiener
        self.Wiener_vel_rec_sten = velocity_from_phi(self.Wiener_phi_rec_sten, self.box, mode="stencil")   if self.Wiener_phi_rec_sten is not None else None
        self.Wiener_vel_rec_fft  = velocity_from_phi(self.Wiener_phi_rec_fft,  self.box, mode="spectral")  if self.Wiener_phi_rec_fft  is not None else None
        # Constrained realization
        self.constrained_realization_vel_sten = velocity_from_phi(self.constrained_realization_phi_sten, self.box, mode="stencil")  if self.constrained_realization_phi_sten is not None else None
        self.constrained_realization_vel_fft  = velocity_from_phi(self.constrained_realization_phi_fft,  self.box, mode="spectral") if self.constrained_realization_phi_fft  is not None else None



