import numpy as np
from Box import Box
def kgrid_rfft3d(box: Box):
    n, L = box.n, box.L
    kx = 2*np.pi * np.fft.fftfreq(n, d=L/n)
    ky = 2*np.pi * np.fft.fftfreq(n, d=L/n)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=L/n)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    return KX, KY, KZ, K, K2


def rfft_multiplicity_last_axis(n):
    w = np.full(n//2+1, 2.0, dtype=float)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w

def spectral_d_dz(field, dx):
    """Periodic spectral derivative along z (uses rFFT along the last axis)."""
    n = field.shape[2]
    kz = 2*np.pi * np.fft.rfftfreq(n, d=dx)       # [h/Mpc]
    KZ = kz[None, None, :]                        # broadcast over x,y
    fk = np.fft.rfftn(field)
    dfdz = np.fft.irfftn(1j * KZ * fk, s=field.shape)
    return dfdz

def shell_power_rfft(field: np.ndarray, L: float, N: int,
                     convention: str = "sampler",
                     return_counts: bool = True):
    """
    Shell-averaged power spectrum P(k) in continuous units [(Mpc/h)^3]
    using NumPy rFFT on a periodic n^3 grid.

    Parameters
    ----------
    field : (n,n,n) array
        Real-space scalar field (e.g., δ or reconstructed δ̂).
    L : float
        Box side length [Mpc/h].
    N : int
        Number of k-bins (shells).
    convention : {"sampler","weighted"}, default "sampler"
        - "sampler": matches your current pipeline (multiplicity in numerator,
          **unweighted** average per shell). Use this if your sampler used 1/w_k.
        - "weighted": multiplicity in numerator **and** denominator (sum of weights).
          Yields unbiased white-noise flat level σ_n^2 * (L/n)^3.

    Returns
    -------
    k_centers : (N,) array
        Bin-center wavenumbers [h/Mpc].
    Pk : (N,) array
        Shell-averaged power [(Mpc/h)^3].
    counts : (N,) array (only if return_counts=True)
        Denominator used in the average (raw counts for "sampler",
        sum of weights for "weighted").
    """
    n = field.shape[0]
    assert field.shape == (n, n, n), "field must be cubic (n,n,n)"
    V = L**3
    Npts = n**3

    # rFFT of field
    Fk = np.fft.rfftn(field)

    # k-grid (h/Mpc)
    kx = 2*np.pi * np.fft.fftfreq(n, d=L/n)
    ky = 2*np.pi * np.fft.fftfreq(n, d=L/n)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=L/n)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Km = np.sqrt(KX**2 + KY**2 + KZ**2)

    # multiplicity weights broadcast to full cube
    wlast = rfft_multiplicity_last_axis(n)[None, None, :]
    wfull = np.broadcast_to(wlast, Km.shape)

    # per-mode power in continuous units (unweighted)
    Pmodes = (V / Npts**2) * (Fk * Fk.conj()).real

    # choose averaging convention
    if convention == "sampler":
        # multiplicity in numerator, raw counts in denominator
        num_arr = wfull * Pmodes
        den_arr = np.ones_like(wfull)
    elif convention == "weighted":
        # multiplicity in both numerator and denominator
        num_arr = wfull * Pmodes
        den_arr = wfull
    else:
        raise ValueError("convention must be 'sampler' or 'weighted'")

    # binning
    edges = np.linspace(0.0, Km.max(), N + 1)
    bins = np.digitize(Km.ravel(), edges) - 1
    valid = (bins >= 0) & (bins < N)

    Psum = np.bincount(bins[valid], weights=num_arr.ravel()[valid], minlength=N)
    Wsum = np.bincount(bins[valid], weights=den_arr.ravel()[valid], minlength=N)

    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = np.where(Wsum > 0, Psum / Wsum, np.nan)

    k_centers = 0.5 * (edges[:-1] + edges[1:])
    if return_counts:
        return k_centers, Pk, Wsum
    return k_centers, Pk


def shell_power_rfft_stencil(field: np.ndarray, L: float, N: int,
                             convention: str = "sampler",
                             return_counts: bool = True):
    """
    Shell-averaged power spectrum P(tilde{k}) using the *lattice* wavenumber
    appropriate for 7-point stencil operators.

    Parameters
    ----------
    field : (n,n,n) array
        Real-space scalar field on a periodic grid.
    L : float
        Box side length [Mpc/h].
    N : int
        Number of k-bins (shells).
    convention : {"sampler","weighted"}, default "sampler"
        Same meaning as in the continuum version.
    return_counts : bool
        If True, also return the bin denominators.

    Returns
    -------
    ktilde_centers : (N,) array
        Bin-center lattice wavenumbers \tilde{k} [h/Mpc].
    Pk : (N,) array
        Shell-averaged power [(Mpc/h)^3].
    counts : (N,) array (only if return_counts=True)
        Denominator used in the average.
    """
    n = field.shape[0]
    assert field.shape == (n, n, n), "field must be cubic (n,n,n)"
    V = L**3
    Npts = n**3
    dx = L / n

    # FFT
    Fk = np.fft.rfftn(field)

    # Continuum k for grid construction
    kx = 2*np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2*np.pi * np.fft.rfftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    # Lattice k^2 symbol: ktilde^2 = (2/dx^2) * sum_i (1 - cos(k_i dx))
    KT2  = (2.0 / dx**2) * (3.0 - np.cos(KX*dx) - np.cos(KY*dx) - np.cos(KZ*dx))
    Ktilde = np.sqrt(KT2)

    # rFFT multiplicity weights on last axis
    wlast = rfft_multiplicity_last_axis(n)[None, None, :]
    wfull = np.broadcast_to(wlast, Ktilde.shape)

    # Per-mode power in continuous units (unweighted)
    Pmodes = (V / Npts**2) * (Fk * Fk.conj()).real

    # Averaging convention
    if convention == "sampler":
        num_arr = wfull * Pmodes
        den_arr = np.ones_like(wfull)
    elif convention == "weighted":
        num_arr = wfull * Pmodes
        den_arr = wfull
    else:
        raise ValueError("convention must be 'sampler' or 'weighted'")

    # Bin on Ktilde
    edges = np.linspace(0.0, Ktilde.max(), N + 1)
    bins = np.digitize(Ktilde.ravel(), edges) - 1
    valid = (bins >= 0) & (bins < N)

    Psum = np.bincount(bins[valid], weights=num_arr.ravel()[valid], minlength=N)
    Wsum = np.bincount(bins[valid], weights=den_arr.ravel()[valid], minlength=N)

    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = np.where(Wsum > 0, Psum / Wsum, np.nan)

    ktilde_centers = 0.5 * (edges[:-1] + edges[1:])
    if return_counts:
        return ktilde_centers, Pk, Wsum
    return ktilde_centers, Pk


def gaussian_smooth_fft(field, box, R_smooth):
    """
    Isotropic Gaussian smoothing via FFT: exp[-0.5 * k^2 * R^2].
    field: (n,n,n) real array
    R_smooth: smoothing radius in Mpc/h (same units as box.L)
    """
    n = box.n
    assert field.shape == (n, n, n), "field shape must match box.n"
    if R_smooth is None or R_smooth <= 0:
        return field.copy()

    # k-grid in h/Mpc (since d = L/n is in Mpc/h)
    k1d = 2 * np.pi * np.fft.fftfreq(n, d=box.L / n)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    k2 = kx*kx + ky*ky + kz*kz

    Wk = np.exp(-0.5 * k2 * (R_smooth**2))  # dimensionless
    F  = np.fft.fftn(field)
    F *= Wk
    smoothed = np.fft.ifftn(F).real
    return smoothed

def make_triangular_rays_mask(n, rays=6, opening_deg=30.0,
                              thickness_frac=0.25, inner_hole_frac=0.02,
                              soft_edge_frac=0.05):
    """
    Build a 3D mask with 'rays' (triangular wedges) emanating from the center
    in the x–y plane, extruded along z across a central slab.

    Parameters
    ----------
    n : int
        Grid size (box is n x n x n).
    rays : int
        Number of rays (6 by default, spaced every 60 degrees).
    opening_deg : float
        Full opening at the *outer radius* per ray (triangle tip at center).
        Controls how wide the triangles get at the edges.
    thickness_frac : float in (0,1]
        Fraction of the box thickness along z that is "observed". E.g. 0.25
        gives a slab |z| <= 0.125 n voxels.
    inner_hole_frac : float
        Exclude a tiny central disk (fraction of box radius) to avoid a fat
        vertex; set small (e.g. 0.02).
    soft_edge_frac : float
        Soft transition width (fraction of box radius) to apodize edges
        (helps reduce ringing). 0 → hard mask.

    Returns
    -------
    M : (n,n,n) float32
        Mask weights in [0,1]. 1 = observed, 0 = masked.
    """

    # voxel-centered coordinates
    ax = np.arange(n) - 0.5*(n-1)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')

    # sky-plane polar coords
    R = np.hypot(X, Y)
    theta = np.arctan2(Y, X)                     # [-pi, pi]
    Rmax = R.max() + 1e-9                        # avoid /0

    # central slab in z
    half_thick = 0.5 * thickness_frac * n
    slab = (np.abs(Z) <= half_thick)

    # ray orientations
    phi0 = np.arange(rays) * (2*np.pi / rays)    # centers of rays

    # wrapped angular distance to the nearest ray center
    # (|angle difference| in [-pi,pi], take min over rays)
    dtheta = np.min(np.abs(np.angle(np.exp(1j*(theta[..., None] - phi0[None, None, :])))),
                    axis=-1)

    # target half-opening grows linearly with radius → triangles
    alpha_edge = 0.5 * np.deg2rad(opening_deg)   # half-opening at outer edge
    half_opening = alpha_edge * (R / Rmax)       # 0 at center → alpha_edge at edge

    # soft edge width in angle units (also grows with radius)
    soft = soft_edge_frac * (R / Rmax) * np.pi   # ~fraction of pi scaled by radius

    # core condition for being inside a ray (with soft transition)
    # hard mask: inside if dtheta <= half_opening
    # soft mask: smoothstep across [half_opening, half_opening+soft]
    t = (half_opening - dtheta) / (soft + 1e-12)   # >0 inside; <0 outside
    if soft_edge_frac > 0:
        # smoothstep 0→1
        u = np.clip(0.5 * (t + 1.0), 0.0, 1.0)
        M_xy = u*u*(3 - 2*u)   # 3u^2 - 2u^3
    else:
        M_xy = (dtheta <= half_opening).astype(float)

    # remove tiny central disk so rays are truly triangular
    M_xy *= (R >= inner_hole_frac * Rmax)

    # extrude through the central slab
    M = (M_xy * slab).astype(np.float32)
    return M

def Pphi_from_Pdelta(K, Pdelta, a, H, f):
    # P_phi = (a H f)^2 / k^4 * P_delta, with safe k=0
    K2 = np.maximum(K**2, 1e-30)
    return (a*H*f)**2 * Pdelta / (K2**2)


def make_cone_mask_with_stripes(
    box,
    los_dir="z",
    cone_half_angle_deg=5,
    observer_offset_L=5.0,
    stripes_count=10,
    stripe_width_deg=10.0,
    stripes_start_deg=0.0,
    radial_period=None,          # [Mpc/h], optional radial “picket fence”
    radial_duty=0.1,             # fraction of period to blank if radial_period given
):
    """
    Build a 3D mask with a cone (apex at an observer outside the box) and
    optional small “stripe” blind spots inside the cone.

    Parameters
    ----------
    box : Box
        Must provide n (grid size) and L (box size [Mpc/h]).
    los_dir : {"x","y","z"} or array-like length 3
        Line-of-sight direction (from observer toward the box).
    cone_half_angle_deg : float
        Half-opening angle of the cone [deg].
    observer_offset_L : float
        Observer distance from box center in units of L (placed along -LOS).
    stripes_count : int
        Number of equally spaced azimuthal stripes (thin angular blind spots).
        0 disables angular stripes.
    stripe_width_deg : float
        Angular width of each azimuthal stripe [deg] (measured in azimuth φ).
    stripes_start_deg : float
        Phase shift of stripe pattern in azimuth [deg].
    radial_period : float or None
        If set (in Mpc/h), applies radial blind spots along the LOS every
        `radial_period`, removing a fraction `radial_duty` of each period.
    radial_duty : float in (0,1)
        Fraction of each radial period to blank (only used if radial_period is not None).

    Returns
    -------
    M : (n,n,n) float array
        Mask with 1.0 inside the cone minus stripes, 0.0 elsewhere.
    """
    n, L = box.n, box.L
    dx = L / n

    # --- LOS direction unit vector ---
    if isinstance(los_dir, str):
        d = dict(x=np.array([1,0,0.], float),
                 y=np.array([0,1,0.], float),
                 z=np.array([0,0,1.], float))[los_dir.lower()]
    else:
        d = np.asarray(los_dir, float)
    d = d / np.linalg.norm(d)

    # --- Box coords & observer position ---
    # grid coordinates in [0, L)
    x = (np.arange(n) + 0.5) * dx
    y = (np.arange(n) + 0.5) * dx
    z = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    center = np.array([L/2, L/2, L/2], float)
    obs = center - observer_offset_L * L * d  # observer sits "behind" the box

    # vectors from observer to each voxel
    RX = X - obs[0]
    RY = Y - obs[1]
    RZ = Z - obs[2]
    R2 = RX*RX + RY*RY + RZ*RZ
    R  = np.sqrt(R2) + 1e-300

    # --- Cone selection ---
    cos_th = (RX*d[0] + RY*d[1] + RZ*d[2]) / R
    cos_th0 = np.cos(np.deg2rad(cone_half_angle_deg))

    # need points in front of the observer (dot > 0) and within half-angle
    in_front = cos_th > 0.0
    inside_cone = in_front & (cos_th >= cos_th0)

    M = np.zeros((n, n, n), dtype=float)
    M[inside_cone] = 1.0

    # --- Build an orthonormal basis (u,v,los) to measure azimuth φ ---
    # pick any vector not parallel to d
    a = np.array([0.0, 0.0, 1.0])
    if np.allclose(np.abs(np.dot(a, d)), 1.0):
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)

    # components in the plane perpendicular to LOS
    Ru = RX*u[0] + RY*u[1] + RZ*u[2]
    Rv = RX*v[0] + RY*v[1] + RZ*v[2]

    # azimuth angle around LOS in [0, 2π)
    phi = np.mod(np.arctan2(Rv, Ru), 2*np.pi)

    # --- Angular (azimuthal) stripes (DESI-like blind spots) ---
    if stripes_count and stripes_count > 0:
        period = 2*np.pi / float(stripes_count)
        width  = np.deg2rad(stripe_width_deg)
        phase  = np.deg2rad(stripes_start_deg)
        # distance to nearest stripe center in periodic sense
        # Use modulo distance: we blank angles within [0, width) of each multiple of 'period'
        phi_mod = np.mod(phi - phase, period)
        angular_hole = (phi_mod < width)
        # apply only within the cone
        M[inside_cone & angular_hole] = 0.0

    # --- Optional radial stripes along LOS ---
    if radial_period is not None and radial_period > 0.0 and 0.0 < radial_duty < 1.0:
        R_par = (RX*d[0] + RY*d[1] + RZ*d[2])  # LOS distance from observer [Mpc/h]
        rp = float(radial_period)
        duty_len = radial_duty * rp
        R_mod = np.mod(R_par, rp)
        radial_hole = (R_mod < duty_len)
        M[inside_cone & radial_hole] = 0.0

    return M.astype(float)

# === 3D diagnostic render of a binary mask using voxels ===
# === 3D diagnostic render of a binary mask using voxels ===
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D registered)

def plot_mask_3d_voxels(M, box=None, stride=None, max_voxels=200_000, alpha=0.18, title=None):
    """
    Render a 3D binary mask as translucent voxels.

    Parameters
    ----------
    M : (n,n,n) array-like
        Mask; treat values >0.5 as 'on'.
    box : Box or None
        If provided, uses box.L for axis labels in [Mpc/h].
    stride : int or None
        Downsampling step (e.g., 2 keeps every 2nd voxel). If None, chosen
        automatically to keep ~max_voxels active cells.
    max_voxels : int
        Target cap for the total number of voxels in the downsampled grid ((n/stride)^3).
    alpha : float in [0,1]
        Opacity of 'on' voxels.
    title : str or None
        Optional plot title.
    """
    M = np.asarray(M)
    assert M.ndim == 3 and M.shape[0] == M.shape[1] == M.shape[2], "M must be (n,n,n)"
    n = M.shape[0]
    on = M > 0.5

    # choose stride automatically to keep grid manageable
    if stride is None:
        s = 1
        while ((n // s) ** 3) > max_voxels and s < n:
            s += 1
        stride = s

    Ms = on[::stride, ::stride, ::stride]
    ns = Ms.shape[0]

    # RGBA facecolors: blue, translucent where mask==1; empty elsewhere
    fc = np.zeros(Ms.shape + (4,), dtype=float)
    fc[Ms] = (0.1, 0.3, 0.9, alpha)  # (R,G,B,alpha)

    # LaTeX look (optional)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.linewidth": 1.5,
        "figure.constrained_layout.use": True,
    })
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # draw voxels
    ax.voxels(Ms, facecolors=fc, edgecolor="none")

    # aspect & labeling
    ax.set_box_aspect((1, 1, 1))
    if box is not None:
        L = float(box.L)
        # tick labels in physical units (approx; indices map linearly)
        ticks = np.linspace(0, ns, 5)
        labels = [rf"{t*L/ns:.0f}" for t in ticks]
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
        ax.set_xticklabels(labels); ax.set_yticklabels(labels); ax.set_zticklabels(labels)
        ax.set_xlabel(r"$x\,[h^{-1}\,\mathrm{Mpc}]$", labelpad=8)
        ax.set_ylabel(r"$y\,[h^{-1}\,\mathrm{Mpc}]$", labelpad=8)
        ax.set_zlabel(r"$z\,[h^{-1}\,\mathrm{Mpc}]$", labelpad=8)
    else:
        ax.set_xlabel("x (vox)"); ax.set_ylabel("y (vox)"); ax.set_zlabel("z (vox)")

    if title:
        ax.set_title(title, pad=12)

    # a little padding around the cube
    ax.set_xlim(0, ns); ax.set_ylim(0, ns); ax.set_zlim(0, ns)

    # cleaner grid
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane._axinfo["grid"]["linewidth"] = 0.3
        pane._axinfo["tick"]["inward_factor"] = 0.0
        pane._axinfo["tick"]["outward_factor"] = 0.0

    plt.show()


import numpy as np

def make_cone_mask_with_subcones(
    box,
    los_dir="z",                 # "x","y","z" or 3-vector
    cone_half_angle_deg=0.7,     # big cone half-angle
    observer_offset_L=5.0,       # observer distance in units of L, along -LOS

    # sub-cones
    subcone_count=5,            # number of small blind cones
    subcone_half_angle_deg=0.1,  # each small cone half-angle (deg)
    subcone_layout="random",     # "random" (inside main cone) or "ring"

    # ring-only params (ignored when layout="random")
    ring_polar_deg=2.0,          # polar tilt of ring (deg) from the LOS axis
    ring_phase_deg=0.0,          # rotate the ring around the LOS (deg)

    # optional central (on-axis) blind cone
    center_subcone_deg=None,     # e.g. 0.8 to carve a central small cone; None to disable

    # RNG control for random layout
    seed=None,

    return_debug=False,          # return directions etc. for plotting arrows
):
    """
    Returns M (n,n,n) with 1 inside the big cone MINUS the union of sub-cones (blind spots),
    and 0 elsewhere. Sub-cones all originate from the same 'observer' point.
    """
    n, L = box.n, box.L
    dx = L / n

    # --- LOS unit vector ---
    if isinstance(los_dir, str):
        d = dict(x=np.array([1,0,0.], float),
                 y=np.array([0,1,0.], float),
                 z=np.array([0,0,1.], float))[los_dir.lower()]
    else:
        d = np.asarray(los_dir, float)
    d = d / np.linalg.norm(d)

    # --- grid & observer ---
    x = (np.arange(n) + 0.5) * dx
    y = (np.arange(n) + 0.5) * dx
    z = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    center = np.array([L/2, L/2, L/2], float)
    obs = center - observer_offset_L * L * d  # far away along -LOS

    RX, RY, RZ = X - obs[0], Y - obs[1], Z - obs[2]
    R2 = RX*RX + RY*RY + RZ*RZ
    R  = np.sqrt(R2) + 1e-300

    # main cone
    cos_th0 = np.cos(np.deg2rad(cone_half_angle_deg))
    cos_th  = (RX*d[0] + RY*d[1] + RZ*d[2]) / R
    in_front = cos_th > 0.0
    inside_cone = in_front & (cos_th >= cos_th0)

    M = np.zeros((n,n,n), dtype=float)
    M[inside_cone] = 1.0

    # --- basis (u, v, d) ---
    a = np.array([0.0, 0.0, 1.0])
    if np.allclose(np.abs(np.dot(a, d)), 1.0):
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)

    sub_dirs = []

    # central sub-cone (on-axis)
    if center_subcone_deg is not None and center_subcone_deg > 0.0:
        sub_dirs.append(d.copy())
        cos_sub_c = np.cos(np.deg2rad(center_subcone_deg))
    else:
        cos_sub_c = None

    # build sub-cone axes
    if subcone_count and subcone_count > 0:
        if subcone_layout.lower() == "ring":
            # fixed polar angle ring around d
            alpha = np.deg2rad(ring_polar_deg)
            phi0  = np.deg2rad(ring_phase_deg)
            for j in range(subcone_count):
                phi = phi0 + 2*np.pi * j / subcone_count
                axis = (np.cos(alpha) * d
                        + np.sin(alpha) * (np.cos(phi) * u + np.sin(phi) * v))
                sub_dirs.append(axis / np.linalg.norm(axis))
        elif subcone_layout.lower() == "random":
            # uniformly sample directions *within the main cone* (spherical-cap uniform)
            rng = np.random.default_rng(seed)
            # sample cos(theta) uniformly on [cos_th0, 1], phi uniformly on [0, 2π)
            cos_t = rng.uniform(cos_th0, 1.0, size=subcone_count)
            sin_t = np.sqrt(1.0 - cos_t**2)
            phi   = rng.uniform(0.0, 2*np.pi, size=subcone_count)
            for ct, st, ph in zip(cos_t, sin_t, phi):
                axis = ct * d + st * (np.cos(ph)*u + np.sin(ph)*v)
                sub_dirs.append(axis / np.linalg.norm(axis))
        else:
            raise ValueError("subcone_layout must be 'random' or 'ring'.")

    cos_sub = np.cos(np.deg2rad(subcone_half_angle_deg))

    # --- carve out sub-cones (blind spots) ---
    RXh, RYh, RZh = RX / R, RY / R, RZ / R
    carved = np.zeros_like(M, dtype=bool)

    # central
    if cos_sub_c is not None:
        dot_c = RXh*d[0] + RYh*d[1] + RZh*d[2]
        carved |= (inside_cone & (dot_c >= cos_sub_c))

    # union of all sub-cones
    for axis in sub_dirs:
        dot = RXh*axis[0] + RYh*axis[1] + RZh*axis[2]
        carved |= (inside_cone & (dot >= cos_sub))

    M[carved] = 0.0

    # diagnostics
    corners = center + 0.5*L*np.array([[-1,-1,-1],[+1,-1,-1],[-1,+1,-1],[+1,+1,-1],
                                       [-1,-1,+1],[+1,-1,+1],[-1,+1,+1],[+1,+1,+1]], float)
    angs = []
    for c in corners:
        r = c - obs
        cc = np.dot(r, d) / (np.linalg.norm(r) + 1e-300)
        angs.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))
    phi_box = float(np.max(angs))

    print(f"[cone] box apparent half-angle ≈ {phi_box:.2f}°; main cone = {cone_half_angle_deg:.2f}°")
    print(f"[mask] coverage (mean) after carving = {M.mean():.4f}")

    # ... inside make_cone_mask_with_subcones(...)

    if return_debug:
        debug = {
            "los": d,                               # NEW: main direction
            "sub_dirs": np.array(sub_dirs),         # axes of all sub-cones
            "observer": obs,
            "subcone_half_angle_deg": float(subcone_half_angle_deg),  # NEW
            "cone_half_angle_deg": float(cone_half_angle_deg)
        }
        return M.astype(float), debug

    return M.astype(float)



# ---------- 2D orthogonal slices ----------
def plot_mask_slices(mask, box, ijk=None, *, cmap="gray",
                     edge_style="contour",   # "contour" | "upsample" | "bilinear"
                     sigma_px=0.75,          # contour smoothing (in pixels)
                     upsample=4):            # upsample factor for "upsample"
    """
    Show x/y/z slices of a binary/float mask in [0,1] with smoother-looking edges.
    edge_style:
      - "contour": discrete image + smoothed 0.5-level contour overlay (recommended)
      - "upsample": upsample slice then show with bilinear interpolation
      - "bilinear": show slice directly with bilinear interpolation
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import ListedColormap, BoundaryNorm

    try:
        from scipy.ndimage import gaussian_filter, zoom
    except Exception:
        gaussian_filter = None
        zoom = None

    n, L = box.n, box.L
    if ijk is None:
        ijk = (n//2, n//2, n//2)
    i, j, k = ijk
    extent = [0, L, 0, L]

    # --- LaTeX plotting style ---
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.linewidth": 2,
        "axes.labelweight": "bold",
        "figure.constrained_layout.use": True,
    })
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=400, constrained_layout=True)

    # two soft greys for 0 and 1 (discrete)
    cmap_mask = ListedColormap([(0.93,0.93,0.93), (0.40,0.40,0.40)])
    norm = BoundaryNorm([0, 0.5, 1], ncolors=2)

    slices = [
        (mask[i, :, :].T, r"$x\,-\,\mathrm{slice}$"),
        (mask[:, j, :].T, r"$y\,-\,\mathrm{slice}$"),
        (mask[:, :, k].T, r"$z\,-\,\mathrm{slice}$"),
    ]

    for ax, (sl, title) in zip(axs, slices):
        if edge_style == "contour":
            # base discrete image (crisp body)
            ax.imshow(sl, origin="lower", extent=extent,
                      cmap=cmap_mask, norm=norm, interpolation="nearest")
            # overlay smooth boundary (needs scipy)
            if gaussian_filter is not None:
                slf = gaussian_filter(sl.astype(float), sigma=sigma_px)
                cs = ax.contour(slf, levels=[0.5], colors='k',
                                linewidths=0.6, alpha=0.85, antialiased=True,
                                origin="lower", extent=extent)
            else:
                # fallback: thin contour on raw field
                cs = ax.contour(sl.astype(float), levels=[0.5], colors='k',
                                linewidths=0.6, alpha=0.85,
                                origin="lower", extent=extent)

        elif edge_style == "upsample":
            if zoom is None:
                # fallback to bilinear if scipy not available
                ax.imshow(sl, origin="lower", extent=extent, cmap="gray",
                          vmin=0, vmax=1, interpolation="bilinear")
            else:
                sl_hr = zoom(sl.astype(float), upsample, order=1, prefilter=False)
                ax.imshow(sl_hr, origin="lower", extent=extent, cmap="gray",
                          vmin=0, vmax=1, interpolation="bilinear")

        elif edge_style == "bilinear":
            ax.imshow(sl.astype(float), origin="lower", extent=extent, cmap="gray",
                      vmin=0, vmax=1, interpolation="bilinear")
        else:
            raise ValueError("edge_style must be 'contour', 'upsample', or 'bilinear'.")

        ax.set_title(title)

        ax.set_aspect("equal")

        # Ticks styling
        def bold_formatter_int(x, pos): return r'${:.0f}$'.format(x)
        ax.tick_params(axis='both', which='both', direction='in', labelsize=10.5)
        ax.tick_params(axis='both', which='major', length=4, width=0.6)
        ax.tick_params(axis='both', which='minor', length=2, width=0.6)
        ax.tick_params(top=True, bottom=True, left=True, right=True)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(bold_formatter_int))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(bold_formatter_int))
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    axs[0].set_xlabel(r'$y\,[h^{-1}\,\mathrm{Mpc}]$', fontsize=10.5)
    axs[0].set_ylabel(r'$z\,[h^{-1}\,\mathrm{Mpc}]$', fontsize=10.5)
    axs[1].set_xlabel(r'$x\,[h^{-1}\,\mathrm{Mpc}]$', fontsize=10.5)
    axs[1].set_ylabel(r'$z\,[h^{-1}\,\mathrm{Mpc}]$', fontsize=10.5)
    axs[2].set_xlabel(r'$x\,[h^{-1}\,\mathrm{Mpc}]$', fontsize=10.5)
    axs[2].set_ylabel(r'$y\,[h^{-1}\,\mathrm{Mpc}]$', fontsize=10.5)
    plt.savefig('mask.png', dpi=700, bbox_inches="tight")
    plt.show()




# ---------- 3D point cloud view ----------
def plot_mask_3d(mask, box, dbg=None, *, thresh=0.5, max_points=120_000, seed=0):
    """
    3D scatter of voxels where mask > thresh. Optionally overlays the observer,
    the LOS, and the sub-cone axes from dbg={'observer','los','sub_dirs'}.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n, L = box.n, box.L
    dx = L / n

    # voxels passing threshold
    idx = np.argwhere(mask > float(thresh))
    if idx.size == 0:
        print("No voxels exceed threshold; try lowering 'thresh'.")
        return

    # random decimation for speed
    N = idx.shape[0]
    if N > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(N, size=max_points, replace=False)
        idx = idx[keep]

    X = (idx[:, 0] + 0.5) * dx
    Y = (idx[:, 1] + 0.5) * dx
    Z = (idx[:, 2] + 0.5) * dx

    fig = plt.figure(figsize=(7, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=1, alpha=0.15)  # light to show volume

    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [Mpc/h]"); ax.set_ylabel("y [Mpc/h]"); ax.set_zlabel("z [Mpc/h]")
    ax.set_title("Mask > threshold (3D scatter)")

    # Overlay observer + axes if provided
    if isinstance(dbg, dict):
        obs = dbg.get("observer", None)
        los = dbg.get("los", None)
        subs = dbg.get("sub_dirs", None)

        ray_len = 1.3 * L
        if obs is not None:
            ax.scatter([obs[0]], [obs[1]], [obs[2]], s=60, marker="*", depthshade=False)
        if (obs is not None) and (los is not None):
            P = np.vstack([obs, obs + ray_len * np.asarray(los)])
            ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=2)
        if (obs is not None) and (subs is not None):
            for a in np.atleast_2d(subs):
                a = np.asarray(a, float)
                P = np.vstack([obs, obs + ray_len * a / (np.linalg.norm(a) + 1e-30)])
                ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=1)

    plt.show()


import numpy as np

def subcone_footprints_on_slice(debug, box, *,
                                slice_axis='z', slice_index=None,
                                use_ellipse=False,  # False: return circles; True: ellipses
                                max_aspect=8.0):   # clamp extreme ellipses if axis ~ in-plane
    """
    From debug info returned by make_cone_mask_with_subcones(return_debug=True),
    compute the footprints of sub-cones on a slice.

    Returns:
      - if use_ellipse=False:
          list of dicts: {"x": x0, "y": y0, "r": r_proj}
        (all in box coordinates, same units as your extent [0,L])

      - if use_ellipse=True:
          list of dicts: {"x": x0, "y": y0, "a": a, "b": b, "theta_deg": angle}
        where (a,b) are semi-axes in data units, and theta is rotation (deg)
        measured from +x toward +y.
    """
    n, L = box.n, box.L
    dx = L / n

    los = np.asarray(debug["los"], float)
    sub_dirs = np.asarray(debug["sub_dirs"], float)
    obs = np.asarray(debug["observer"], float)
    theta_sub = np.deg2rad(float(debug["subcone_half_angle_deg"]))

    # pick plane
    if slice_index is None:
        slice_index = n // 2
    if slice_axis.lower() == 'z':
        n_p = np.array([0,0,1.0])
        c   = (slice_index + 0.5) * dx
        axes_order = ('x','y')  # we’ll return x,y
    elif slice_axis.lower() == 'y':
        n_p = np.array([0,1.0,0])
        c   = (slice_index + 0.5) * dx
        axes_order = ('x','z')
    elif slice_axis.lower() == 'x':
        n_p = np.array([1.0,0,0])
        c   = (slice_index + 0.5) * dx
        axes_order = ('y','z')
    else:
        raise ValueError("slice_axis must be 'x','y', or 'z'.")

    footprints = []
    for a in sub_dirs:
        a = a / np.linalg.norm(a)
        denom = float(np.dot(a, n_p))
        if np.isclose(denom, 0.0, atol=1e-10):
            continue  # axis nearly parallel to plane -> skip

        # Intersection with plane n_p · x = c
        t = (c - np.dot(obs, n_p)) / denom
        if t <= 0:         # behind the observer or pointing away
            continue
        P = obs + t * a    # intersection point (x,y,z)

        # inside box?
        if np.any(P < 0.0) or np.any(P > L):
            continue

        # cone's physical radius at distance t along axis:
        rho = t * np.tan(theta_sub)

        if not use_ellipse:
            # Simple circle approximation (good if axis near normal to plane)
            r_proj = rho / max(abs(denom), 1e-6)  # mild correction for obliquity
            if slice_axis == 'z':
                footprints.append({"x": P[0], "y": P[1], "r": r_proj})
            elif slice_axis == 'y':
                footprints.append({"x": P[0], "y": P[2], "r": r_proj})
            else:  # 'x'
                footprints.append({"x": P[1], "y": P[2], "r": r_proj})
        else:
            # Ellipse on oblique plane
            # Semi-axes: minor ≈ rho, major ≈ rho/|a·n_p|, oriented by a_proj in the plane.
            a_dot = abs(denom)
            b = rho                     # minor
            a_sem = min(rho / max(a_dot, 1e-6), max_aspect * rho)  # major, clamped

            # orientation: projection of axis into the plane
            a_proj = a - denom * n_p
            if np.linalg.norm(a_proj) < 1e-12:
                theta = 0.0
            else:
                a_proj /= np.linalg.norm(a_proj)
                # angle in the 2D plane coordinates
                if slice_axis == 'z':
                    theta = np.degrees(np.arctan2(a_proj[1], a_proj[0]))
                    x0, y0 = P[0], P[1]
                elif slice_axis == 'y':
                    theta = np.degrees(np.arctan2(a_proj[2], a_proj[0]))
                    x0, y0 = P[0], P[2]
                else: # 'x'
                    theta = np.degrees(np.arctan2(a_proj[2], a_proj[1]))
                    x0, y0 = P[1], P[2]

            footprints.append({"x": x0, "y": y0, "a": a_sem, "b": b, "theta_deg": theta})

    return footprints
