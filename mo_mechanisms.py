"""Zwei Mechanismen, die Zahlungsbereitschaften abfragen: der Moulin-Mechanismus (Moulin/Shenker 2001) mit beliebiger Anteilstabelle und der Grenzkosten-Mechanismus (VCG/Clarke-Pivot).

Mechanismus = Regel, die aus den gemeldeten Geboten b_i bestimmt, wer bedient wird und wer wie viel zahlt. Ehrlich melden (b_i = v_i, den wahren Wert) soll sich für jeden lohnen ("strategiesicher").
Nutzen eines Spediteurs: v_i - Zahlung, wenn bedient, sonst 0."""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import mo_constants as C
from mo_shares import members


@dataclass
class Outcome:
    served: int                       # Bitmaske der bedienten Spediteure
    pay: np.ndarray                   # Zahlung je Spediteur (0, wenn nicht bedient)
    rounds: list = field(default_factory=list)      # Moulin: je Runde (Menge S, Anteile, Ausgestiegene)


def moulin(bids, X, n, factor=1.0):
    """Moulin-Mechanismus: beginne mit allen; wer weniger bietet, als sein Anteil in der aktuellen Menge kostet, steigt (gleichzeitig mit allen anderen Betroffenen) aus; wiederhole, bis niemand mehr aussteigt.
    Wer bleibt, zahlt seinen Anteil in der Endmenge."""
    S = (1 << n) - 1
    rounds = []
    while True:
        shares = X[S] * factor
        out = [i for i in members(S, n) if bids[i] < shares[i] - C.EPS]
        rounds.append((S, shares, out))
        if not out:
            break
        for i in out:
            S &= ~(1 << i)
    pay = np.zeros(n)
    for i in members(S, n):
        pay[i] = rounds[-1][1][i]
    return Outcome(S, pay, rounds)


@lru_cache(maxsize=None)
def _membership(n):
    masks = np.arange(1 << n)
    return np.array([(masks >> i) & 1 for i in range(n)], dtype=float).T          # (2^n, n)


def _popcount(n):
    return _membership(n).sum(axis=1)


def welfare_table(v, c, n):
    """W[S] = Summe der Werte von S minus Tourkosten c(S)."""
    return _membership(n) @ np.asarray(v, dtype=float) - c


def efficient_set(v, c, n):
    """Wohlfahrtsmaximale Menge (bei Gleichstand die größte) und ihre Wohlfahrt."""
    W = welfare_table(v, c, n)
    best = W.max()
    cand = np.flatnonzero(W >= best - C.EPS)
    pop = _popcount(n)[cand]
    return int(cand[np.argmax(pop * 1e6 + cand)]), float(best)


def vcg(bids, c, n):
    """Grenzkosten-Mechanismus: die für die Gebote wohlfahrtsmaximale Menge wird bedient; wer bedient wird, zahlt sein Gebot minus seinen Beitrag zur Wohlfahrt (Clarke-Pivot)."""
    S, best = efficient_set(bids, c, n)
    W = welfare_table(bids, c, n)
    masks = np.arange(1 << n)
    pay = np.zeros(n)
    for i in members(S, n):
        without = W[((masks >> i) & 1) == 0].max()
        pay[i] = bids[i] - (best - without)
    return Outcome(S, pay)


def utility(v, outcome, i):
    return float(v[i] - outcome.pay[i]) if (outcome.served >> i) & 1 else 0.0


def welfare_of(v, served, c):
    return float(sum(v[i] for i in members(served, len(v))) - c[served])


# --- Lügen suchen -------------------------------------------------------------------------------------------------------------------------------


def moulin_candidates(X, n, i, factor=1.0):
    """Die Ausgänge hängen nur davon ab, welche Schwellen (Anteile von i in einer Menge S) das Gebot erreicht: Gebote genau auf den Schwellen (und 0) decken jedes mögliche Verhalten ab."""
    vals = {0.0}
    for S in range(1, 1 << n):
        if (S >> i) & 1:
            vals.add(round(float(X[S, i] * factor), 10))
    return sorted(vals)


def vcg_candidates(v, c, n, i):
    """Gitter aus Vielfachen des wahren Werts plus die Schwelle, ab der i bedient wird (mit den ehrlichen Geboten der anderen)."""
    lo = v.copy()
    lo[i] = 0.0
    S0, _ = efficient_set(lo, c, n)
    W = welfare_table(lo, c, n)
    masks = np.arange(1 << n)
    with_i = W[((masks >> i) & 1) == 1].max()
    without_i = W[((masks >> i) & 1) == 0].max()
    tau = max(0.0, without_i - with_i)
    base = float(v[i])
    return sorted({0.0, tau, max(tau - 0.01, 0.0), tau + 0.01, 0.5 * base, base, 1.5 * base, 2 * base, 3 * base, 5 * base})


def best_deviation(run, v, i, candidates, tol=1e-6):
    """Größter Gewinn, den Spediteur i allein durch ein falsches Gebot erzielen kann, gegenüber ehrlichem Bieten (mit den ehrlichen Geboten der anderen). Rückgabe (Gewinn >= 0, bestes Gebot)."""
    truth = utility(v, run(v), i)
    best, arg = 0.0, float(v[i])
    for b in candidates:
        bids = v.copy()
        bids[i] = b
        gain = utility(v, run(bids), i) - truth
        if gain > best + tol:
            best, arg = gain, float(b)
    return best, arg


def pair_deviation(run, v, i, j, cand_i, cand_j, tol=1e-6):
    """Gibt es Gebote (b_i, b_j), bei denen keiner der beiden schlechter und mindestens einer besser fährt (Koalitionsabweichung)? Rückgabe (gefunden, (b_i, b_j), (Gewinn_i, Gewinn_j))."""
    ti, tj = utility(v, run(v), i), utility(v, run(v), j)
    for bi in cand_i:
        for bj in cand_j:
            bids = v.copy()
            bids[i], bids[j] = bi, bj
            out = run(bids)
            gi, gj = utility(v, out, i) - ti, utility(v, out, j) - tj
            if gi >= -tol and gj >= -tol and (gi > tol or gj > tol):
                return True, (float(bi), float(bj)), (gi, gj)
    return False, None, None
