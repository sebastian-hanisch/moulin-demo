"""Vehikel B "Spediteurs-Kooperation": n Spediteure mit je einem Kunden-Stopp in einem 100 x 100 km großen Gebiet, gemeinsames Depot in der Mitte. Ein Spediteur allein fährt
Depot - Stopp - Depot; eine Koalition fährt eine gemeinsame Rundtour (Held-Karp, exakt)."""

from dataclasses import dataclass

import numpy as np

import mo_constants as C


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray        # (n + 1, 2): Zeile 0 = Depot, Zeile i + 1 = Stopp von Spediteur i
    seed: int
    layout: str

    @property
    def n(self):
        return len(self.xy) - 1

    def dist(self):
        d = self.xy[:, None, :] - self.xy[None, :, :]
        return np.sqrt((d ** 2).sum(axis=2))


def generate(n, layout="uniform", seed=0):
    rng = np.random.default_rng(seed)
    if layout == "uniform":
        pts = rng.random((n, 2)) * C.AREA
    elif layout == "clustered":
        centres = 15.0 + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 30.0)
        which = rng.integers(0, C.N_CLUSTERS, size=n)
        pts = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n, 2)), 0.0, C.AREA)
    else:
        raise ValueError(layout)
    depot = np.array([[C.AREA / 2, C.AREA / 2]])
    return Instance(np.vstack([depot, pts]), int(seed), layout)
