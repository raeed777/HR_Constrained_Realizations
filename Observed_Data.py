# Observed_Data.py
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict
import numpy as np
from Data import Data  # your truth container


Array = np.ndarray

@dataclass
class ObservedData(Data):
    """
    Base class for observed fields produced from a truth Data object
    by applying bias, mask, and noise. Results live in:
      - d: observed field (masked, noisy)
      - sigma: per-voxel noise std
      - W: weights = mask / sigma^2
    """
    mask: Optional[Array] = field(default=None, repr=False)     # 0/1 or fractional weights in [0,1]
    noise: Optional[Array] = field(default=None, repr=False)   # noise
    sigma_noise: float = 0.0
    b_bias: float = 1.0
    dataset: Literal["real", "pp"] = "real"                     # "real" or plane-parallel "pp"

    # Outputs
    d: Optional[Array] = field(default=None, repr=False)

    def __post_init__(self):
        n = self.box.n
        if self.mask is None:
            self.mask = np.ones((n, n, n), dtype=float)
        else:
            assert self.mask.shape == (n, n, n), "mask must be (n,n,n)"
            self.mask = self.mask.astype(float)

    def generate_artificial_noise(self, sigma_frac=0.2, jitter_frac=0.0):
        if self.dataset == "real":
            obs_field = self.delta_r
        else:
            obs_field = self.delta_s_z
        rng = np.random.default_rng(42)
        n = obs_field.shape[0]

        base = float(obs_field.std()) if obs_field.std() > 0 else 1.0
        sigma0 = sigma_frac * base
        if jitter_frac > 0:
            lo, hi = (1.0 - jitter_frac)*sigma0, (1.0 + jitter_frac)*sigma0
            sigma_x = rng.uniform(lo, hi, size=obs_field.shape).astype(np.float64)
        else:
            sigma_x = np.full_like(obs_field, sigma0, dtype=np.float64)
        self.sigma_noise = sigma_x
        self.noise = rng.normal(0.0, 1.0, size=obs_field.shape) * sigma_x
        
    def add_noise_and_mask(self):
        if self.dataset == "real":
            obs_field = self.delta_r
        else:
            obs_field = self.delta_s_z

        M = np.ones_like(field, dtype=np.float64) if self.mask is None else self.mask.astype(np.float64)

        self.d = M * (self.b_bias * obs_field + self.noise)     # outside mask this is 0 (and solver will set W=0 there)

    def generate_mock_fields(self, *args, **kwargs):
        super().generate_mock_fields(*args, **kwargs)
