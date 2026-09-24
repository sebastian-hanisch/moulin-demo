"""Auswertung: die drei Mechanismen auf einer Instanz mit privaten Zahlungsbereitschaften, Lügen-Suche und vier Experimente
(Voraussetzungen der Moulin-Konstruktion, Vergleich der Mechanismen, Wirkung des Folk-Faktors, gezielte Lügensuche im Shapley-Mechanismus)."""

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import mo_constants as C
import mo_game as G
import mo_mechanisms as M
import mo_scenario as S
import mo_shares as H


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    layout: str = "uniform"
    seed: int = C.DEFAULT_SEED
    theta: float = C.DEFAULT_THETA
    factor: float = C.DEFAULT_FACTOR


@lru_cache(maxsize=256)
def instance(n, layout, seed):
    return S.generate(n, layout, seed)


@lru_cache(maxsize=64)
def game(n, layout, seed):
    inst = instance(n, layout, seed)
    dist = inst.dist()
    c, parent, dp = G.tsp_values(dist)
    Xs = H.shapley_table(c, n)
    Xf, mst = H.folk_table(dist, n)
    return inst, dist, c, parent, dp, Xs, Xf, mst


def make_values(c, n, theta, seed):
    """Private Werte: v_i = theta_i * Kosten der Alleinfahrt von i, theta_i ~ U(0, theta) (eigener Zufallsstrom, unabhängig von der Lage der Stopps)."""
    rng = np.random.default_rng([seed, 4242])
    return theta * rng.random(n) * np.array([c[1 << i] for i in range(n)])


def cm_violation(X, n, tol=1e-9):
    """Größte Verletzung der Kreuzmonotonie: (Spediteur i, Menge S, dazukommender Spediteur j, Anteil in S, Anteil in S u j) oder None."""
    best = None
    for s in range(1, 1 << n):
        for j in range(n):
            if (s >> j) & 1:
                continue
            t = s | (1 << j)
            for i in H.members(s, n):
                d = X[t, i] - X[s, i]
                if d > tol and (best is None or d > best[5]):
                    best = (i, s, j, float(X[s, i]), float(X[t, i]), float(d))
    return None if best is None else best[:5]


@dataclass
class Analysis:
    settings: Settings
    inst: object
    dist: np.ndarray
    c: np.ndarray
    parent: np.ndarray
    dp: np.ndarray
    Xs: np.ndarray
    Xf: np.ndarray
    mst: np.ndarray
    v: np.ndarray
    outcomes: dict
    eff_set: int
    eff_welfare: float
    submodular: bool
    shapley_cm: bool
    shapley_violation: object

    @property
    def n(self):
        return self.inst.n

    @property
    def alone(self):
        return G.stand_alone(self.c, self.n)

    def run(self, mech, bids):
        n = self.n
        if mech == "moulin_folk":
            return M.moulin(bids, self.Xf, n, self.settings.factor)
        if mech == "moulin_shapley":
            return M.moulin(bids, self.Xs, n, 1.0)
        return M.vcg(bids, self.c, n)

    def candidates(self, mech, i):
        n = self.n
        if mech == "moulin_folk":
            return M.moulin_candidates(self.Xf, n, i, self.settings.factor)
        if mech == "moulin_shapley":
            return M.moulin_candidates(self.Xs, n, i, 1.0)
        return M.vcg_candidates(self.v, self.c, n, i)

    def welfare(self, mech):
        return M.welfare_of(self.v, self.outcomes[mech].served, self.c)

    def revenue(self, mech):
        return float(self.outcomes[mech].pay.sum())

    def cost(self, mech):
        return float(self.c[self.outcomes[mech].served])

    def utility_curve(self, mech, i, grid):
        """Nutzen von Spediteur i für jedes Gebot der Gitterpunkte (alle anderen bieten ehrlich)."""
        out = []
        for b in grid:
            bids = self.v.copy()
            bids[i] = b
            out.append(M.utility(self.v, self.run(mech, bids), i))
        return out

    def deviations(self, mech):
        """Je Spediteur (Gewinn, bestes falsches Gebot) der besten Einzel-Lüge."""
        return [M.best_deviation(lambda b: self.run(mech, b), self.v, i, self.candidates(mech, i)) for i in range(self.n)]


def vcg_breakdown(an):
    """Je bedientem Spediteur: Wohlfahrt mit allen Geboten, ohne ihn, sein Beitrag und seine Zahlung (Clarke-Pivot)."""
    n = an.n
    out = an.outcomes["vcg"]
    W = M.welfare_table(an.v, an.c, n)
    masks = np.arange(1 << n)
    rows = []
    for i in H.members(out.served, n):
        without = float(W[((masks >> i) & 1) == 0].max())
        rows.append({"carrier": i, "with": an.eff_welfare, "without": without, "contribution": an.eff_welfare - without, "value": float(an.v[i]), "pay": float(out.pay[i])})
    return rows


def pair_search(an, mech):
    """Erste Zweier-Absprache, bei der keiner der beiden schlechter und einer besser fährt (Moulin: alle Schwellen, exakt; VCG: Gitter aus Vielfachen des Werts plus Schwelle, also nur eine untere Grenze).
    Rückgabe (Fund oder None, Zahl der geprüften Paare)."""
    n = an.n
    scanned = 0
    for i, j in itertools.combinations(range(n), 2):
        scanned += 1
        found, bids, gains = M.pair_deviation(lambda b: an.run(mech, b), an.v, i, j, an.candidates(mech, i), an.candidates(mech, j))
        if found:
            return {"i": i, "j": j, "bids": bids, "gains": gains}, scanned
    return None, scanned


@lru_cache(maxsize=32)
def analyse(settings):
    inst, dist, c, parent, dp, Xs, Xf, mst = game(settings.n, settings.layout, settings.seed)
    n = inst.n
    v = make_values(c, n, settings.theta, settings.seed)
    an = Analysis(settings, inst, dist, c, parent, dp, Xs, Xf, mst, v, {}, 0, 0.0, H.is_submodular(c, n), *_cm(Xs, n))
    an.outcomes = {m: an.run(m, v) for m in C.MECHANISMS}
    an.eff_set, an.eff_welfare = M.efficient_set(v, c, n)
    return an


def _cm(Xs, n):
    ok = H.is_cross_monotone(Xs, n)[0]
    return ok, (None if ok else cm_violation(Xs, n))


# --- Experiment 1: Voraussetzungen ---------------------------------------------------------------------------------------------------------------


def prereq_experiment(ns=None, layouts=None, seeds=None):
    ns = C.PREREQ_NS if ns is None else ns
    layouts = C.LAYOUTS if layouts is None else layouts
    seeds = C.PREREQ_SEEDS if seeds is None else seeds
    rows = []
    for n in ns:
        for layout in layouts:
            sub = cm_s = cm_f = 0
            budget_gap, over = 0.0, []
            for s in seeds:
                _, dist, c, _, _, Xs, Xf, mst = game(n, layout, s)
                sub += H.is_submodular(c, n)
                cm_s += H.is_cross_monotone(Xs, n)[0]
                cm_f += H.is_cross_monotone(Xf, n)[0]
                budget_gap = max(budget_gap, float(np.abs(Xf.sum(axis=1)[1:] - mst[1:]).max()))
                over.extend(2 * mst[m] / c[m] for m in range(1, 1 << n) if bin(m).count("1") >= 2)
            k = len(seeds)
            rows.append({"n": n, "layout": layout, "n_inst": k, "submodular": sub / k, "shapley_cm": cm_s / k, "folk_cm": cm_f / k, "folk_budget_gap": budget_gap, "overcharge": float(np.mean(over)),
                         "overcharge_max": float(np.max(over))})
    return rows


# --- Experiment 2: Vergleich der Mechanismen -----------------------------------------------------------------------------------------------------


def _profiles(n, layout, seed, theta, k):
    inst, dist, c, parent, dp, Xs, Xf, mst = game(n, layout, seed)
    return [make_values(c, n, theta, seed * 10 + r) for r in range(k)], (c, Xs, Xf)


def compare_experiment(n=None, layout="uniform", seeds=None, profiles=None, theta=None, factor=None, pairs=True):
    n = C.COMPARE_N if n is None else n
    seeds = C.COMPARE_SEEDS if seeds is None else seeds
    profiles = C.COMPARE_PROFILES if profiles is None else profiles
    theta = C.DEFAULT_THETA if theta is None else theta
    factor = C.DEFAULT_FACTOR if factor is None else factor
    acc = {m: {"welfare": 0.0, "rev": 0.0, "cost": 0.0, "served": 0, "deficit": 0, "recovery_fail": 0, "ind": 0, "pair": 0, "runs": 0} for m in C.MECHANISMS}
    eff_w = eff_served = total = 0
    non_cm = 0
    for seed in seeds:
        vs, (c, Xs, Xf) = _profiles(n, layout, seed, theta, profiles)
        non_cm += not H.is_cross_monotone(Xs, n)[0]
        for v in vs:
            total += 1
            Se, We = M.efficient_set(v, c, n)
            eff_w += We
            eff_served += bin(Se).count("1")
            runs = {"moulin_folk": lambda b: M.moulin(b, Xf, n, factor), "moulin_shapley": lambda b: M.moulin(b, Xs, n, 1.0), "vcg": lambda b: M.vcg(b, c, n)}
            for m, run in runs.items():
                out = run(v)
                a = acc[m]
                a["welfare"] += M.welfare_of(v, out.served, c)
                a["served"] += bin(out.served).count("1")
                if out.served:
                    a["rev"] += out.pay.sum()
                    a["cost"] += c[out.served]
                    a["runs"] += 1
                    a["deficit"] += out.pay.sum() < c[out.served] - 1e-6
                cands = [M.moulin_candidates(Xf, n, i, factor) for i in range(n)] if m == "moulin_folk" else [M.moulin_candidates(Xs, n, i, 1.0) for i in range(n)] if m == "moulin_shapley" \
                    else [M.vcg_candidates(v, c, n, i) for i in range(n)]
                a["ind"] += any(M.best_deviation(run, v, i, cands[i])[0] > 1e-6 for i in range(n))
                if pairs:
                    a["pair"] += any(M.pair_deviation(run, v, i, j, cands[i], cands[j])[0] for i, j in itertools.combinations(range(n), 2))
    out = {"n": n, "n_profiles": total, "non_cm_instances": non_cm, "n_inst": len(seeds), "eff_welfare": eff_w, "eff_served": eff_served / total}
    for m, a in acc.items():
        out[m] = {"welfare_share": a["welfare"] / eff_w if eff_w > 0 else 1.0, "budget": a["rev"] / a["cost"] if a["cost"] > 0 else float("nan"), "served": a["served"] / total,
                  "deficit": a["deficit"] / max(a["runs"], 1), "ind": a["ind"] / total, "pair": a["pair"] / total}
    return out


# --- Experiment 3: Folk-Faktor -------------------------------------------------------------------------------------------------------------------


def factor_experiment(n=None, layout="uniform", seeds=None, levels=None, theta=None, profiles=3):
    n = C.FACTOR_N if n is None else n
    seeds = C.FACTOR_SEEDS if seeds is None else seeds
    levels = C.FACTOR_LEVELS if levels is None else levels
    theta = C.DEFAULT_THETA if theta is None else theta
    rows = []
    for f in levels:
        w = ew = 0.0
        rev = cost = 0.0
        served = deficit = runs = total = 0
        for seed in seeds:
            vs, (c, Xs, Xf) = _profiles(n, layout, seed, theta, profiles)
            for v in vs:
                total += 1
                _, We = M.efficient_set(v, c, n)
                ew += We
                out = M.moulin(v, Xf, n, f)
                w += M.welfare_of(v, out.served, c)
                served += bin(out.served).count("1")
                if out.served:
                    runs += 1
                    rev += out.pay.sum()
                    cost += c[out.served]
                    deficit += out.pay.sum() < c[out.served] - 1e-6
        rows.append({"factor": f, "welfare_share": w / ew if ew > 0 else 1.0, "budget": rev / cost if cost > 0 else float("nan"), "served": served / total, "deficit": deficit / max(runs, 1), "runs": runs})
    return rows


# --- Experiment 4: gezielte Lügensuche im Shapley-Mechanismus ------------------------------------------------------------------------------------


def manipulation_search(n=5, layout="uniform", seeds=None, profiles=20, rng_seed=1, factor=None):
    """Nur Instanzen, in denen der Shapley-Wert NICHT kreuzmonoton ist (dort gilt die Moulin-Garantie nicht), je `profiles` zufällige Wert-Profile (Werte = Alleinfahrt-Kosten x U(0,1) x Niveau aus {1, 1,5, 2,5}).
    Gezählt: Profile, in denen irgendein Spediteur allein durch ein falsches Gebot besser fährt - im Shapley-Mechanismus und (Kontrolle) im Folk-Mechanismus mit denselben Profilen."""
    seeds = range(60) if seeds is None else seeds
    factor = C.DEFAULT_FACTOR if factor is None else factor
    rng = np.random.default_rng(rng_seed)
    out = {"n": n, "instances": 0, "non_cm_instances": 0, "profiles": 0, "shapley_hits": 0, "folk_hits": 0, "example": None}
    for seed in seeds:
        _, dist, c, _, _, Xs, Xf, _ = game(n, layout, 8000 + seed)
        out["instances"] += 1
        if H.is_cross_monotone(Xs, n)[0]:
            continue
        out["non_cm_instances"] += 1
        run_s = lambda b: M.moulin(b, Xs, n, 1.0)
        run_f = lambda b: M.moulin(b, Xf, n, factor)
        cs = [M.moulin_candidates(Xs, n, i, 1.0) for i in range(n)]
        cf = [M.moulin_candidates(Xf, n, i, factor) for i in range(n)]
        alone = np.array([c[1 << i] for i in range(n)])
        for p in range(profiles):
            v = alone * rng.random(n) * rng.choice([1.0, 1.5, 2.5])
            out["profiles"] += 1
            for i in range(n):
                g, b = M.best_deviation(run_s, v, i, cs[i])
                if g > 1e-6:
                    out["shapley_hits"] += 1
                    if out["example"] is None:
                        out["example"] = {"seed": 8000 + seed, "profile": p, "carrier": i, "gain": g, "bid": b, "value": float(v[i])}
                    break
            out["folk_hits"] += any(M.best_deviation(run_f, v, i, cf[i])[0] > 1e-6 for i in range(n))
    return out
