import numpy as np
import camb


def build_camb_pk_callable(
    Om=0.315, Ob=0.049, h=0.674, ns=0.965,
    sigma8_target=0.811,           # set None to use As instead
    As=None,                       # only used if sigma8_target is None
    z=0.0,                         # redshift for P(k,z)
    kmax_h=10.0,                   # max k in h/Mpc for the interpolator
    mnu=0.0, tau=0.06,             # neutrino mass and tau (don’t matter much for P_lin shape here)
    nonlinear=False
):
    """
    Returns:
      Pdelta_callable(K) -> P(k,z) with units (Mpc/h)^3 for K in h/Mpc (same shape as K).
      Also returns the CAMB interpolator in case you want direct access.
    """
    # === set up cosmology ===
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0*h,
        ombh2=Ob*h*h,
        omch2=(Om-Ob)*h*h,
        mnu=mnu, omk=0.0, tau=tau
    )
    pars.InitPower.set_params(As=2e-9 if As is None else As, ns=ns)
    pars.set_matter_power(redshifts=[z], kmax= kmax_h / (1.0) )  # kmax here is in 1/Mpc unless using k_hunit in interpolator

    # compute once to get sigma8 (today) if we want to rescale to a target
    results = camb.get_results(pars)
    if sigma8_target is not None:
        # sigma8 ∝ sqrt(As), so As_new = As_old * (σ8_target/σ8_old)^2
        sigma8_now = results.get_sigma8_0()  # today’s σ8 from current As
        scale = (sigma8_target / float(sigma8_now))**2
        pars.InitPower.set_params(As=float(pars.InitPower.As) * scale, ns=ns)
        results = camb.get_results(pars)  # recompute with rescaled As
    # else: we used user As directly

    # Interpolator that returns k in h/Mpc and P in (Mpc/h)^3 (matches your code)
    pk_interp = camb.get_matter_power_interpolator(
        pars,
        nonlinear=nonlinear,
        hubble_units=True,   # P in (Mpc/h)^3
        k_hunit=True,        # k in h/Mpc
        return_z_k=False
    )

    def Pdelta_callable(K):
        # CAMB interpolator wants plain array of k; supports broadcasting
        K = np.asarray(K)
        out = pk_interp.P(z, K)
        # force DC to zero for safety
        if out.shape == K.shape:
            out = np.where(K == 0.0, 0.0, out)
        elif np.isscalar(out):
            out = 0.0 if np.isscalar(K) and K == 0.0 else out
        return out

    return Pdelta_callable, pk_interp