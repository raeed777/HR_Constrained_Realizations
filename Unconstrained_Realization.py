class Operators:
    def __init__(self, box: Box, cosmo: Cosmology, Pdelta_callable, use_lattice_in_fft=True,
                 # new: observer setup for radial RSD
                 radial_observer_offset_L=5.0,  # observer at center - offset*L * los_dir
                 radial_los_dir="z"):          # or a 3-vector
        a, H, f = cosmo.a, cosmo.H, cosmo.f
        self.box = box
        self.a, self.H, self.f = a, H, f
        self.KX, self.KY, self.KZ, self.K, self.K2 = kgrid_rfft3d(box)

        # Prior spectra
        self.Pdelta = Pdelta_callable(self.K)
        self.Pphi   = Pphi_from_Pdelta(self.K, self.Pdelta, a, H, f)

        # FFT prior precision
        dx3 = box.dx**3
        self.Sphi_inv_k_spec = np.zeros_like(self.Pphi)
        np.divide(dx3, self.Pphi, out=self.Sphi_inv_k_spec, where=(self.Pphi>0))
        self.Sphi_inv_k_spec[0,0,0] = 0.0

        # lattice symbols
        dx = box.dx
        cosx = np.cos(self.KX * dx)
        cosy = np.cos(self.KY * dx)
        cosz = np.cos(self.KZ * dx)
        self.KT2  = (2.0 / dx**2) * (3.0 - cosx - cosy - cosz)
        self.KTZ2 = (2.0 / dx**2) * (1.0 - cosz)

        self._dx = dx
        self._laplace_coeff = 1.0 / (dx*dx)

        # φ -> δ (real; spec vs lat)
        self.Lk_real_spec = -(self.K2)  / (a*H*f)
        self.Lk_real_lat  = -(self.KT2) / (a*H*f)

        # φ -> δ_s (PP along z; spec vs lat)
        self.Lk_pp_spec = -(self.K2 + f*(self.KZ**2)) / (a*H*f)
        self.Lk_pp_lat  = -(self.KT2 + f*self.KTZ2)   / (a*H*f)

        self.Lk_real = self.Lk_real_lat if use_lattice_in_fft else self.Lk_real_spec
        self.Lk_pp   = self.Lk_pp_lat   if use_lattice_in_fft else self.Lk_pp_spec

        self.Lk_real[0,0,0] = 0.0
        self.Lk_pp[0,0,0]   = 0.0

        # stencil-consistent prior for preconditioning
        KT2_safe = np.maximum(self.KT2, 1e-30)
        self.Pphi_lat = (a*H*f)**2 * self.Pdelta / (KT2_safe**2)
        self.Sphi_inv_k_sten = np.zeros_like(self.Pphi_lat)
        np.divide(dx3, self.Pphi_lat, out=self.Sphi_inv_k_sten, where=(self.Pphi_lat>0))
        self.Sphi_inv_k_sten[0,0,0] = 0.0

        # legacy PP symbol (kept for convenience)
        if use_lattice_in_fft:
            self.Lk = -(self.KT2 + f*self.KTZ2) / max(a*H*f, 1e-30)
        else:
            self.Lk = -(self.K2 + f*(self.KZ**2)) / max(a*H*f, 1e-30)

        # ---------- NEW: radial geometry precompute ----------
        # line-of-sight base direction
        if isinstance(radial_los_dir, str):
            d = dict(x=np.array([1,0,0.], float),
                     y=np.array([0,1,0.], float),
                     z=np.array([0,0,1.], float))[radial_los_dir.lower()]
        else:
            d = np.asarray(radial_los_dir, float)
        d = d / np.linalg.norm(d)

        n = box.n; L = box.L
        # grid coords (cell-centered)
        ax = (np.arange(n) + 0.5) * (L/n)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        center = np.array([L/2, L/2, L/2], float)
        observer = center - radial_observer_offset_L * L * d

        RX, RY, RZ = X - observer[0], Y - observer[1], Z - observer[2]
        R = np.sqrt(RX*RX + RY*RY + RZ*RZ) + 1e-300
        nx, ny, nz = RX/R, RY/R, RZ/R

        # store unit-LOS and 1/R for later use (geometry term if needed)
        self._nhat = (nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32))
        self._invR = (1.0/R).astype(np.float32)

    # ----- prior terms -----
    def apply_Sphi_inv_fft(self, x):
        Xk = np.fft.rfftn(x)
        Yk = self.Sphi_inv_k_spec * Xk
        Yk[0,0,0] = 0.0
        return np.fft.irfftn(Yk, s=x.shape)

    def apply_Sphi_inv_stencil(self, x):
        Xk = np.fft.rfftn(x)
        Yk = self.Sphi_inv_k_sten * Xk
        Yk[0,0,0] = 0.0
        return np.fft.irfftn(Yk, s=x.shape)

    # ----- existing PP operators -----
    def apply_L_rsd_pp_fft(self, phi):
        Phik = np.fft.rfftn(phi)
        return np.fft.irfftn(self.Lk_pp * Phik, s=phi.shape)

    def apply_L_stencil(self, x):
        dx = self._dx
        def d2(u, axis): return (np.roll(u,+1,axis)+np.roll(u,-1,axis)-2.0*u)/(dx*dx)
        lap = d2(x,0)+d2(x,1)+d2(x,2)
        return -lap / (self.a*self.H*self.f)

    # ========= NEW: radial RSD apply (hybrid spectral) =========
    def apply_L_rsd_radial_fft(self, phi, include_geom=False):
        """
        δ_s = -(∇²φ + f ∂_n^2 φ)/(a H f),     ∂_n^2 φ = n_i n_j ∂_i ∂_j φ
        Compute ∂_i∂_j via FFT (Hessian), contract with n_i n_j(x) in real space.
        If include_geom=True, add the geometric 2 v_r / r term to ∂_n v_r before linearization;
        normally you keep it False for linear-RSD form used here.
        """
        a, H, f = self.a, self.H, self.f
        n = phi.shape[0]
        KX, KY, KZ, K2 = self.KX, self.KY, self.KZ, self.K2

        Phik = np.fft.rfftn(phi, s=phi.shape)

        # Laplacian: ∇²φ = -k^2 φ̂ -> real space
        lap = np.fft.irfftn(-K2 * Phik, s=phi.shape).real

        # Hessian components: ∂i∂j φ -> IFFT of (-k_i k_j) φ̂
        Hxx = np.fft.irfftn(-(KX*KX) * Phik, s=phi.shape).real
        Hyy = np.fft.irfftn(-(KY*KY) * Phik, s=phi.shape).real
        Hzz = np.fft.irfftn(-(KZ*KZ) * Phik, s=phi.shape).real
        Hxy = np.fft.irfftn(-(KX*KY) * Phik, s=phi.shape).real
        Hxz = np.fft.irfftn(-(KX*KZ) * Phik, s=phi.shape).real
        Hyz = np.fft.irfftn(-(KY*KZ) * Phik, s=phi.shape).real

        nx, ny, nz = self._nhat
        # ∂_n^2 φ = n_i n_j H_ij
        dnn = (nx*nx)*Hxx + (ny*ny)*Hyy + (nz*nz)*Hzz + 2*(nx*ny)*Hxy + 2*(nx*nz)*Hxz + 2*(ny*nz)*Hyz

        # linear radial RSD mapping
        delta_s = -(lap + f * dnn) / (a*H*f + 1e-30)
        return delta_s

    # ========= NEW: radial RSD apply (real-space stencil) =========
    def apply_L_rsd_radial_stencil(self, phi):
        """
        Same form as above but using 7-point second derivatives and mixed terms
        via central differences for ∂i∂j.
        """
        a, H, f = self.a, self.H, self.f
        dx = self._dx

        def d2(u, axis):
            return (np.roll(u,+1,axis)+np.roll(u,-1,axis)-2.0*u)/(dx*dx)

        # mixed ∂i∂j via standard centered scheme:
        def d2_mixed(u, ax1, ax2):
            # ∂_{ax1}∂_{ax2} u with central differences
            return ( np.roll(np.roll(u,-1,ax1),-1,ax2)
                   - np.roll(np.roll(u,+1,ax1),-1,ax2)
                   - np.roll(np.roll(u,-1,ax1),+1,ax2)
                   + np.roll(np.roll(u,+1,ax1),+1,ax2) ) / (4*dx*dx)

        lap = d2(phi,0) + d2(phi,1) + d2(phi,2)
        Hxx, Hyy, Hzz = d2(phi,0), d2(phi,1), d2(phi,2)
        Hxy = d2_mixed(phi,0,1)
        Hxz = d2_mixed(phi,0,2)
        Hyz = d2_mixed(phi,1,2)

        nx, ny, nz = self._nhat
        dnn = (nx*nx)*Hxx + (ny*ny)*Hyy + (nz*nz)*Hzz + 2*(nx*ny)*Hxy + 2*(nx*nz)*Hxz + 2*(ny*nz)*Hyz

        delta_s = -(lap + f * dnn) / (a*H*f + 1e-30)
        return delta_s
