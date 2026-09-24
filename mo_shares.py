"""Kostenanteile für jede Teilmenge S der Spediteure (Zeile S der Tabelle, Spalte i = Anteil von Spediteur i, wenn genau S bedient wird).

- Shapley-Anteile des Tourenspiels c(S) (kürzeste Rundtour): fair, aber nur kreuzmonoton, wenn die Kosten submodular sind - im Tourenspiel meist nicht der Fall.
- Folk-Anteile des Spannbaums: Kosten von S = Länge des kürzesten Spannbaums über Depot und S (Kruskal), aufgeteilt nach der Folk-Regel. Die Anteile sind kreuzmonoton, weil ein weiterer Spediteur
  Cluster nur vergrößert oder früher ans Depot anschließt (siehe folk_shares). Die Summe der Anteile ist die Spannbaumlänge; das Doppelte davon deckt jede Rundtour (Tour-Verdopplung)."""

import math

import numpy as np


def members(mask, n):
    return [i for i in range(n) if (mask >> i) & 1]


def shapley_table(c, n):
    """Shapley-Wert des Teilspiels auf S, für jede Teilmenge S (Zeile S; Nicht-Mitglieder 0)."""
    full = 1 << n
    X = np.zeros((full, n))
    fact = [math.factorial(k) for k in range(n + 2)]
    for s in range(1, full):
        k = bin(s).count("1")
        for i in range(n):
            if not (s >> i) & 1:
                continue
            rest = s & ~(1 << i)
            t = rest
            tot = 0.0
            while True:
                tk = bin(t).count("1")
                tot += fact[tk] * fact[k - tk - 1] / fact[k] * (c[t | (1 << i)] - c[t])
                if t == 0:
                    break
                t = (t - 1) & rest
            X[s, i] = tot
    return X


def _kruskal(dist, mask, n):
    """Kruskal über Depot (Knoten 0) und die Spediteure in `mask`; liefert (Folk-Anteile je Spediteur, Spannbaumlänge).
    Folk-Regel als Zeitintegral: über den Schwellenwert t steigend gibt es Cluster von Stopps, die durch Kanten <= t verbunden sind; jeder Cluster ohne Depot verteilt seine
    Zuwachsrate 1 gleichmäßig auf seine Spediteure (Shapley-Wert des Irreduziblen Spiels, Bergantiños/Vidal-Puerto)."""
    mem = members(mask, n)
    nodes = [0] + [i + 1 for i in mem]
    edges = sorted((dist[a, b], a, b) for ai, a in enumerate(nodes) for b in nodes[ai + 1:])
    parent = {a: a for a in nodes}
    size = {a: (0 if a == 0 else 1) for a in nodes}          # Zahl der Spediteure im Cluster
    has_root = {a: a == 0 for a in nodes}
    share = {a: 0.0 for a in nodes[1:]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev, total = 0.0, 0.0
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        for x in nodes[1:]:
            rx = find(x)
            if not has_root[rx]:
                share[x] += (w - prev) / size[rx]
        prev = w
        total += w
        parent[rb] = ra
        size[ra] += size[rb]
        has_root[ra] = has_root[ra] or has_root[rb]
    out = np.zeros(n)
    for i in mem:
        out[i] = share[i + 1]
    return out, total


def folk_shares(dist, mask, n):
    return _kruskal(dist, mask, n)[0]


def mst_cost(dist, mask, n):
    return _kruskal(dist, mask, n)[1]


def folk_table(dist, n):
    """(X, mst): X[S] = Folk-Anteile, mst[S] = Spannbaumlänge über Depot und S."""
    full = 1 << n
    X = np.zeros((full, n))
    mst = np.zeros(full)
    for s in range(1, full):
        X[s], mst[s] = _kruskal(dist, s, n)
    return X, mst


def is_cross_monotone(X, n, tol=1e-9):
    """Kein Anteil steigt, wenn ein weiterer Spediteur dazukommt: X[S u {j}][i] <= X[S][i] für alle i in S, j nicht in S. Rückgabe (ok, größte Verletzung)."""
    worst = 0.0
    for s in range(1, 1 << n):
        for j in range(n):
            if (s >> j) & 1:
                continue
            t = s | (1 << j)
            for i in range(n):
                if (s >> i) & 1:
                    worst = max(worst, X[t, i] - X[s, i])
    return worst <= tol, worst


def is_submodular(c, n, tol=1e-9):
    """c(S u i) - c(S) >= c(S u i u j) - c(S u j) für alle S, i, j (abnehmende Grenzkosten)."""
    for s in range(1 << n):
        for i in range(n):
            if (s >> i) & 1:
                continue
            for j in range(i + 1, n):
                if (s >> j) & 1:
                    continue
                if c[s | 1 << i] - c[s] < c[s | 1 << i | 1 << j] - c[s | 1 << j] - tol:
                    return False
    return True
