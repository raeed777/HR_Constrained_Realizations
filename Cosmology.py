from dataclasses import dataclass
from colossus.cosmology import cosmology as col_cosmo
import numpy as np

@dataclass(frozen=True)
class Cosmology:
    a: float     # scale factor at the chosen redshift
    H: float     # H(z) [km/s/Mpc]
    f: float     # growth rate f = d ln D / d ln a at that z
    Om: float    # Ω_m,0 today
    h: float     # h = H0 / (100 km/s/Mpc)
    ns: float    # scalar spectral index
    A: float     # here: we'll store sigma8 for convenience


def make_planck15_cosmology(z: float = 0.0) -> Cosmology:
    """
    Build a Cosmology instance using the Colossus 'planck15' cosmology,
    evaluated at redshift z.
    """
    # 1) Set global cosmology in Colossus
    col = col_cosmo.setCosmology('planck15')

    # 2) Background quantities at this z
    a = 1.0 / (1.0 + z)
    H = col.Hz(z)              # km/s/Mpc

    # Growth rate using the standard f ≈ Ω_m(z)^γ with γ ≈ 0.545
    Om_z = col.Om(z)           # Ω_m(z)
    gamma = 0.545
    fz = Om_z**gamma

    # 3) "Today" parameters (z=0) from Colossus
    Om0 = col.Om0
    h   = col.h
    ns  = col.ns

    # 4) Amplitude A: use sigma8(today) so existing code still has something sensible
    A = col.sigma8

    return Cosmology(
        a=a,
        H=H,
        f=fz,
        Om=Om0,
        h=h,
        ns=ns,
        A=A,
    )
