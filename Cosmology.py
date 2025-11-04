from dataclasses import dataclass


@dataclass(frozen=True)
class Cosmology:
    a: float
    H: float
    f: float
    Om: float
    h: float
    ns: float
    A: float