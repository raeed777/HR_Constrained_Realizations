from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Box:
    n: int
    L: float # Mpc/h
    @property
    def N(self): return self.n**3
    @property
    def V(self): return self.L**3
    @property
    def dx(self) -> float:
        return self.L / self.n