"""Unabhängige Orakel: (1) Folk-Anteile als Shapley-Wert des irreduziblen Spiels (Bergantiños/Vidal-Puerto: Kosten einer Gruppe T = Spannbaum über Depot und T mit den Minimax-Abständen
der Gruppe S, Prim statt Kruskal, Permutationen in exakter Bruchrechnung), (2) Moulin-Ausgang gegen eine mengenbasierte Neuschreibung und gegen die Vereinigung aller stabilen Mengen
(kreuzmonotone Anteile), (3) Grenzkosten-Mechanismus gegen Vollaufzählung aller Mengen."""

import itertools
import math
import random
from fractions import Fraction

import numpy as np

import mo_game as G
import mo_mechanisms as M
import mo_scenario as S
import mo_shares as H


def shapley_by_permutations(cost, players):
    total = {p: Fraction(0) for p in players}
    for order in itertools.permutations(players):
        seen = frozenset()
        for p in order:
            total[p] += Fraction(cost(seen | {p})) - Fraction(cost(seen))
            seen = seen | {p}
    return {p: float(t / math.factorial(len(players))) for p, t in total.items()}


def prim_length(nodes, w):
    nodes = list(nodes)
    inside, total = {nodes[0]}, 0.0
    while len(inside) < len(nodes):
        cost, nxt = min((w(a, b), b) for a in inside for b in nodes if b not in inside)
        total += cost
        inside.add(nxt)
    return total


def folk_by_irreducible_game(d, members):
    nodes = [0] + [i + 1 for i in members]
    B = {(a, b): (0.0 if a == b else d[a, b]) for a in nodes for b in nodes}
    for k in nodes:
        for a in nodes:
            for b in nodes:
                B[a, b] = min(B[a, b], max(B[a, k], B[k, b]))
    return shapley_by_permutations(lambda T: prim_length([0] + [i + 1 for i in T], lambda a, b: B[a, b]) if T else 0.0, members)


def test_folk_shares_equal_the_shapley_value_of_the_irreducible_game_for_every_group():
    rng = random.Random(3)
    for _ in range(14):
        n = rng.randint(2, 5)
        d = S.generate(n, rng.choice(("uniform", "clustered")), rng.randrange(10 ** 6)).dist()
        X, mst = H.folk_table(d, n)
        for s in range(1, 1 << n):
            members = [i for i in range(n) if (s >> i) & 1]
            ref = folk_by_irreducible_game(d, members)
            for i in members:
                assert abs(ref[i] - X[s, i]) < 1e-8
            assert abs(X[s].sum() - prim_length([0] + [i + 1 for i in members], lambda a, b: d[a, b])) < 1e-8


def test_folk_shares_with_duplicate_stops_and_stops_at_the_depot():
    rng = random.Random(9)
    for _ in range(20):
        n = rng.randint(2, 4)
        xy = np.array([[50, 50]] + [[rng.choice((50, 60, 70)), rng.choice((50, 60, 80))] for _ in range(n)], dtype=float)
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(2))
        X, _ = H.folk_table(d, n)
        for s in range(1, 1 << n):
            members = [i for i in range(n) if (s >> i) & 1]
            ref = folk_by_irreducible_game(d, members)
            assert all(abs(ref[i] - X[s, i]) < 1e-8 for i in members)


def moulin_reference(bids, X, n, factor):
    current = set(range(n))
    while True:
        s = sum(1 << i for i in current)
        leaving = {i for i in current if bids[i] < X[s, i] * factor - 1e-9}
        if not leaving:
            return s, [X[s, i] * factor if i in current else 0.0 for i in range(n)]
        current -= leaving


def vcg_reference(bids, c, n):
    welfare = {s: sum(bids[i] for i in range(n) if (s >> i) & 1) - c[s] for s in range(1 << n)}
    best = max(welfare.values())
    served = max((s for s in welfare if welfare[s] >= best - 1e-9), key=lambda s: (bin(s).count("1"), s))
    pay = [bids[i] - (best - max(w for s, w in welfare.items() if not (s >> i) & 1)) if (served >> i) & 1 else 0.0 for i in range(n)]
    return served, pay, best


def test_moulin_and_vcg_agree_with_enumeration_on_random_profiles():
    rng = random.Random(11)
    for t in range(120):
        n = rng.randint(2, 5)
        d = S.generate(n, rng.choice(("uniform", "clustered")), rng.randrange(10 ** 6)).dist()
        c = G.tsp_values(d)[0]
        Xs, (Xf, _) = H.shapley_table(c, n), H.folk_table(d, n)
        alone = [c[1 << i] for i in range(n)]
        v = np.array([rng.choice((0.4, 1.0, 1.5, 3.0)) * rng.random() * a for a in alone])
        if t % 5 == 0:
            v = np.array([Xf[(1 << n) - 1, i] * 1.5 if rng.random() < 0.5 else 0.0 for i in range(n)])          # Gebote genau auf Schwellen und Nullen
        for X, factor in ((Xf, rng.choice((1.0, 2.0))), (Xs, 1.0)):
            out = M.moulin(v, X, n, factor)
            served, pay = moulin_reference(v, X, n, factor)
            assert out.served == served and np.allclose(out.pay, pay, atol=1e-9)
        out = M.moulin(v, Xf, n, 2.0)
        stable = [s for s in range(1, 1 << n) if all(v[i] >= Xf[s, i] * 2.0 - 1e-9 for i in range(n) if (s >> i) & 1)]
        union = 0
        for s in stable:
            union |= s
        assert out.served == union                                                    # kreuzmonoton: Endmenge = Vereinigung aller stabilen Mengen
        out = M.vcg(v, c, n)
        served, pay, best = vcg_reference(v, c, n)
        assert out.served == served and np.allclose(out.pay, pay, atol=1e-8)
