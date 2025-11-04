# pspectra.py
import numpy as np

def eisenstein_hu_nowiggle_T(k, Om=0.315, h=0.674):
    Gamma = Om * h
    q = np.where(Gamma > 0, k / Gamma, 0.0)
    L0 = np.log(1 + 2.34*q)
    C0 = 14.2 + 731.0/(1 + 62.5*q)
    denom = L0 + C0*q*q
    return np.where(k > 0, L0/denom, 1.0)

def Pk_phys_nowiggle(k, Om=0.315, h=0.674, ns=0.965, A=1.0):
    """Return P_m(k,z=0) in (Mpc/h)^3 using no-wiggle EH T(k)."""
    T = eisenstein_hu_nowiggle_T(k, Om=Om, h=h)
    return A * np.where(k>0, k**ns * T**2, 0.0)

def Pk_phys_at_z_from_P0(k, P0_callable, growth_ratio):
    """
    Given a P0(k) callable at z=0 and D(z)/D(0), return P(k,z) = [D(z)/D(0)]^2 P0(k).
    """
    return (growth_ratio**2) * P0_callable(k)
