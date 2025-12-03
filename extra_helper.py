import numpy as np

def _rng(shape, seed):
    rng = np.random.default_rng(seed)
    # zero-mean to avoid DC pollution in FFT-based ops
    x = rng.normal(0.0, 1.0, size=shape)
    x -= x.mean()
    return x

def _inner(a, b):
    # Real inner product; guard complex intermediates
    return float(np.vdot(a, b).real)

def dot_test_pair(L, LT, shape, trials=5, seed=0):
    """
    Tests <L x, y> ?= <x, LT y>. Returns list of relative errors per trial.
    """
    errs = []
    for t in range(trials):
        x = _rng(shape, seed + 13*t)
        y = _rng(shape, seed + 17*t)
        lhs = _inner(L(x), y)
        rhs = _inner(x, LT(y))
        denom = max(abs(lhs), abs(rhs), 1e-30)
        errs.append(abs(lhs - rhs) / denom)
    return np.array(errs)

def dot_test_LWL(L, LT, W, shape, trials=5, seed=0):
    """
    Tests <W L x, y> ?= <x, LT (W y)> — this is what you need inside A.
    """
    errs = []
    for t in range(trials):
        x = _rng(shape, seed + 23*t)
        y = _rng(shape, seed + 29*t)
        lhs = _inner(W * L(x), y)       # < W L x, y >
        rhs = _inner(x, LT(W * y))      # < x, L^T (W y) >
        denom = max(abs(lhs), abs(rhs), 1e-30)
        errs.append(abs(lhs - rhs) / denom)
    return np.array(errs)

def symmetry_test_A(apply_A, shape, trials=5, seed=0):
    """
    Tests <A x, y> ?= <x, A y> for the full normal operator A.
    """
    errs = []
    for t in range(trials):
        x = _rng(shape, seed + 31*t)
        y = _rng(shape, seed + 37*t)
        lhs = _inner(apply_A(x), y)
        rhs = _inner(x, apply_A(y))
        denom = max(abs(lhs), abs(rhs), 1e-30)
        errs.append(abs(lhs - rhs) / denom)
    return np.array(errs)

def summarize(name, errs, tol_good=1e-10, tol_ok=1e-8):
    m = float(np.median(errs))
    M = float(np.max(errs))
    tag = "OK" if m < tol_ok and M < 10*tol_ok else ("GREAT" if m < tol_good else "BAD")
    print(f"[{name}] median={m:.2e}, max={M:.2e}  -> {tag}")


import numpy as np

# --------------------------
# 1) Apodization utilities
# --------------------------
def tukey_1d(n, alpha=0.3):
    """
    1D Tukey window (alpha in [0,1]).
    alpha=0   -> rectangular (no taper)
    alpha=1   -> Hann over full length
    """
    i = np.arange(n, dtype=float)
    if alpha <= 0.0:
        return np.ones(n, float)
    if alpha >= 1.0:
        return 0.5*(1 - np.cos(2*np.pi*i/(n-1)))

    w = np.ones(n, float)
    # left taper: 0 <= i < alpha*(n-1)/2
    L = int(np.floor(alpha*(n-1)/2.0))
    if L > 0:
        il = np.arange(L+1, dtype=float)
        w[:L+1] = 0.5*(1 + np.cos(np.pi*(2*il/(alpha*(n-1)) - 1)))

    # right taper: (n-1)*(1 - alpha/2) < i <= n-1
    R0 = int(np.ceil((n-1)*(1 - alpha/2.0)))
    if R0 < n:
        ir = np.arange(n - R0, dtype=float)
        w[R0:] = 0.5*(1 + np.cos(np.pi*(2*ir/(alpha*(n-1)) + 1)))
    return w

def tukey3d(n, alpha=0.3):
    """Separable 3D Tukey window."""
    w1 = tukey_1d(n, alpha=alpha)
    W  = w1[:,None,None] * w1[None,:,None] * w1[None,None,:]
    return W

# ---------------------------------------
# 2) Generic dot-test with apodized inner
# ---------------------------------------
def dot_test_apodized(L, LT, shape, *, rng=None, nsamples=6, window=None, Wx=None, desc=""):
    """
    Tests <Lx, y> == <x, LT y> under:
      (a) standard inner product
      (b) apodized inner product with window w(x)
      (c) (optional) data-weighted inner product with Wx(x)

    Parameters
    ----------
    L, LT   : callables mapping (n,n,n)->(n,n,n)
    shape   : (n,n,n)
    rng     : numpy Generator
    nsamples: int
    window  : (n,n,n) apodization weights in [0,1] (e.g. tukey3d(n,0.3))
    Wx      : (n,n,n) nonnegative weights (e.g. 1/sigma^2 * mask)
    """
    rng = np.random.default_rng() if rng is None else rng
    n = shape[0]
    assert shape == (n,n,n)

    def ip(u, v, w=None):
        if w is None:
            return float(np.sum(u*v))
        return float(np.sum(w*u*v))

    eps = 1e-30
    errs_plain = []
    errs_apod  = []
    errs_W     = []

    for _ in range(nsamples):
        x = rng.normal(0,1,size=shape).astype(np.float64, copy=False)
        y = rng.normal(0,1,size=shape).astype(np.float64, copy=False)

        Lx   = L(x).astype(np.float64, copy=False)
        LTy  = LT(y).astype(np.float64, copy=False)

        # (a) plain inner product
        a = ip(Lx, y)
        b = ip(x, LTy)
        err_plain = abs(a - b) / (abs(a) + abs(b) + eps)
        errs_plain.append(err_plain)

        # (b) apodized inner product <u,v>_w = sum w u v
        if window is not None:
            aw = ip(Lx, y, window)
            bw = ip(x, LTy, window)
            err_apod = abs(aw - bw) / (abs(aw) + abs(bw) + eps)
            errs_apod.append(err_apod)

        # (c) weighted inner product with Wx (data/noise weights):
        if Wx is not None:
            # NOTE: This is the correct weighted-duality: <Wx Lx, y> = <x, LT (Wx y)>
            aW = ip(Wx*Lx, y)
            bW = ip(x, LT(Wx*y))
            err_W = abs(aW - bW) / (abs(aW) + abs(bW) + eps)
            errs_W.append(err_W)

    def summarize(name, arr):
        if len(arr)==0: return
        arr = np.array(arr)
        print(f"[{desc:<12} {name}] median={np.median(arr):.3e}, max={np.max(arr):.3e}")

    summarize("plain",   errs_plain)
    summarize("apod",    errs_apod)
    summarize("weighted",errs_W)
