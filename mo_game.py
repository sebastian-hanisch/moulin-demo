"""Kostenspiel der Spediteurs-Kooperation. Koalitionswert c(S) = Länge der kürzesten Rundtour ab Depot durch alle Stopps der Koalition S (Traveling-Salesman-Spiel, Potters/Curiel/Tijs 1992).
Alle 2^n Werte kommen aus einer einzigen Held-Karp-Rechnung: dp[S][j] = kürzester Weg vom Depot durch genau die Stopps in S mit Ende bei j."""

import math

import numpy as np

import mo_constants as C


def tsp_values(dist):
    """(c, parent, dp): c[mask] = Länge der kürzesten Rundtour für die Stopps in `mask` (Bit i = Stopp i, Depot ist Knoten 0 von `dist`); parent[mask, j] = vorletzter Stopp (-1 = Depot)."""
    n = len(dist) - 1
    full = 1 << n
    INF = float("inf")
    dp = np.full((full, n), INF)
    parent = np.full((full, n), -2, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = dist[0, j + 1]
        parent[1 << j, j] = -1
    for mask in range(1, full):
        for j in range(n):
            if not (mask >> j) & 1 or dp[mask, j] == INF:
                continue
            base = dp[mask, j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nxt = mask | (1 << k)
                cand = base + dist[j + 1, k + 1]
                if cand < dp[nxt, k]:
                    dp[nxt, k] = cand
                    parent[nxt, k] = j
    c = np.zeros(full)
    for mask in range(1, full):
        c[mask] = min(dp[mask, j] + dist[j + 1, 0] for j in range(n) if (mask >> j) & 1)
    return c, parent, dp


def tour(dist, parent, dp, mask):
    """Optimale Rundtour der Koalition `mask` als Stopp-Reihenfolge (Indizes 0..n-1, ohne Depot)."""
    n = len(dist) - 1
    j = min((k for k in range(n) if (mask >> k) & 1), key=lambda k: dp[mask, k] + dist[k + 1, 0])
    order = []
    m = mask
    while j != -1:
        order.append(j)
        pj = int(parent[m, j])
        m ^= 1 << j
        j = pj
    return order[::-1]


def popcount_array(n):
    full = 1 << n
    pc = np.zeros(full, dtype=np.int64)
    for i in range(n):
        pc += (np.arange(full) >> i) & 1
    return pc


def shapley(c, n):
    """Shapley-Wert des Kostenspiels c (Array über alle Masken): phi_i = Summe über S ohne i von |S|! (n-|S|-1)! / n! * (c(S + i) - c(S))."""
    full = 1 << n
    pc = popcount_array(n)
    masks = np.arange(full)
    weight = np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) if s < n else 0.0 for s in range(n + 1)])
    phi = np.zeros(n)
    for i in range(n):
        without = masks[(masks >> i) & 1 == 0]
        phi[i] = float((weight[pc[without]] * (c[without | (1 << i)] - c[without])).sum())
    return phi


def stand_alone(c, n):
    return np.array([c[1 << i] for i in range(n)])


def _coalition_sums(x, n):
    full = 1 << n
    sums = np.zeros(full)
    masks = np.arange(full)
    for i in range(n):
        sums += ((masks >> i) & 1) * x[i]
    return sums


def core_excess(x, c, n):
    """Größte Überschreitung: max_S (Summe_{i in S} x_i - c(S)) über alle echten, nichtleeren Koalitionen (ohne die große Koalition; <= 0: keine Gruppe von Spediteuren würde allein besser fahren).
    Rückgabe: (Überschreitung, Maske der schlimmsten Koalition)."""
    ex = _coalition_sums(x, n) - c
    ex[0] = -np.inf
    ex[(1 << n) - 1] = -np.inf
    k = int(np.argmax(ex))
    return float(ex[k]), k


def blocking_count(x, c, n, tol=1e-9):
    """Zahl der Koalitionen, die allein günstiger führen als mit der Aufteilung x."""
    ex = _coalition_sums(x, n) - c
    ex[0] = -np.inf
    ex[(1 << n) - 1] = -np.inf
    return int((ex > tol).sum())


def in_core(x, c, n, tol=1e-9):
    """Effizient (Summe = c(N)) und keine Koalition zahlt mehr als allein."""
    return abs(x.sum() - c[(1 << n) - 1]) <= 1e-7 and core_excess(x, c, n)[0] <= tol


def proportional(c, n):
    """Aufteilung der Gesamtkosten proportional zu den Kosten der Alleinfahrt."""
    a = stand_alone(c, n)
    return a / a.sum() * c[(1 << n) - 1]


def equal_savings(c, n):
    """Jeder Spediteur bekommt gleich viel Ersparnis: Alleinkosten minus (Gesamtersparnis / n)."""
    a = stand_alone(c, n)
    return a - (a.sum() - c[(1 << n) - 1]) / n


def shapley_sampled(c, n, samples, seed=0):
    """Schätzung des Shapley-Werts: Mittel der Grenzkosten über `samples` zufällige Reihenfolgen (Castro/Gómez/Tejada 2009)."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    for _ in range(samples):
        order = rng.permutation(n)
        mask = 0
        for i in order:
            total[i] += c[mask | (1 << int(i))] - c[mask]
            mask |= 1 << int(i)
    return total / samples


def marginal_costs(c, order):
    """Grenzkosten der Spediteure, wenn sie in der Reihenfolge `order` (Liste von Indizes) zur Kooperation hinzukommen: Zuwachs der Tourlänge."""
    mask = 0
    out = {}
    for i in order:
        out[int(i)] = float(c[mask | (1 << int(i))] - c[mask])
        mask |= 1 << int(i)
    return out


def coalition_members(mask, n):
    return [i for i in range(n) if (mask >> i) & 1]
