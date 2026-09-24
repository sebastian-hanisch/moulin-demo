"""Kostenanteile: Folk-Regel per Handrechnung und gegen einen unabhängigen Prim, Kreuzmonotonie, Shapley-Tabelle gegen die Einzelfunktion, Submodularität per Handrechnung."""

import itertools

import numpy as np
import pytest

import mo_game as G
import mo_scenario as S
import mo_shares as H


def prim_cost(dist, mask, n):
    nodes = [0] + [i + 1 for i in range(n) if (mask >> i) & 1]
    in_tree, total = {0}, 0.0
    while len(in_tree) < len(nodes):
        w, v = min((dist[u, v], v) for u in in_tree for v in nodes if v not in in_tree)
        in_tree.add(v)
        total += w
    return total


def test_folk_by_hand_for_two_agents():
    """Depot 0, Spediteur 1 (Abstand 3 zum Depot), Spediteur 2 (5 zum Depot, 4 zu 1): Kanten 3 und 4 im Spannbaum; bis t = 3 wächst jeder Cluster {1}, {2} mit Rate 1 (je 3),
    danach ist 1 am Depot, 2 wächst allein bis t = 4 (+1): Anteile 3 und 4, zusammen der Spannbaum 7."""
    dist = np.array([[0, 3, 5], [3, 0, 4], [5, 4, 0]], dtype=float)
    x, total = H._kruskal(dist, 0b11, 2)
    assert x == pytest.approx([3.0, 4.0]) and total == pytest.approx(7.0)
    assert H.folk_shares(dist, 0b01, 2) == pytest.approx([3.0, 0.0]) and H.folk_shares(dist, 0b10, 2) == pytest.approx([0.0, 5.0])


def test_folk_by_hand_two_agents_far_from_the_depot_but_close_together():
    """Beide 10 vom Depot, 2 voneinander: bis t = 2 wächst jeder für sich (je 2); dann ein Cluster {1,2} bis t = 10, Rate 1 geteilt durch 2: je +4. Anteile 6 und 6, Summe 12 = Spannbaum (10 + 2)."""
    dist = np.array([[0, 10, 10], [10, 0, 2], [10, 2, 0]], dtype=float)
    assert H.folk_shares(dist, 0b11, 2) == pytest.approx([6.0, 6.0])
    assert H.mst_cost(dist, 0b11, 2) == pytest.approx(12.0)


@pytest.mark.parametrize("n,layout,seed", [(5, "uniform", 1), (6, "clustered", 2), (7, "uniform", 3)])
def test_folk_sums_to_the_spanning_tree_and_matches_an_independent_prim(n, layout, seed):
    d = S.generate(n, layout, seed).dist()
    X, mst = H.folk_table(d, n)
    for mask in range(1, 1 << n):
        assert X[mask].sum() == pytest.approx(prim_cost(d, mask, n), abs=1e-7)
        assert mst[mask] == pytest.approx(prim_cost(d, mask, n), abs=1e-7)
        for i in range(n):
            if not (mask >> i) & 1:
                assert X[mask, i] == 0.0
            else:
                assert 0 < X[mask, i] <= d[0, i + 1] + 1e-9                     # nie mehr als der direkte Weg zum Depot


def test_folk_of_a_single_carrier_is_the_distance_to_the_depot():
    d = S.generate(6, "uniform", 4).dist()
    for i in range(6):
        assert H.folk_shares(d, 1 << i, 6)[i] == pytest.approx(d[0, i + 1])


@pytest.mark.parametrize("n,layout,seed", [(5, "uniform", 5), (6, "uniform", 6), (6, "clustered", 7), (7, "clustered", 8)])
def test_folk_is_cross_monotone(n, layout, seed):
    d = S.generate(n, layout, seed).dist()
    ok, worst = H.is_cross_monotone(H.folk_table(d, n)[0], n)
    assert ok and worst <= 1e-9


def test_folk_cross_monotonicity_against_an_explicit_loop():
    n = 5
    d = S.generate(n, "uniform", 9).dist()
    X = H.folk_table(d, n)[0]
    for s in range(1, 1 << n):
        for t in range(s, 1 << n):
            if s & t == s:                                                          # S Teilmenge von T
                for i in range(n):
                    if (s >> i) & 1:
                        assert X[t, i] <= X[s, i] + 1e-9


def test_spanning_tree_bounds_the_tour_from_both_sides():
    """MST(S u Depot) <= Rundtour(S) <= 2 MST(S u Depot) (Tour-Verdopplung mit Abkürzungen)."""
    n = 7
    for seed in range(3):
        d = S.generate(n, "uniform", seed).dist()
        c = G.tsp_values(d)[0]
        mst = H.folk_table(d, n)[1]
        for mask in range(1, 1 << n):
            assert mst[mask] <= c[mask] + 1e-9 <= 2 * mst[mask] + 2e-9


def test_shapley_table_agrees_with_the_permutation_definition_on_every_subgame():
    n = 5
    c = G.tsp_values(S.generate(n, "uniform", 3).dist())[0]
    X = H.shapley_table(c, n)
    for mask in range(1, 1 << n):
        mem = H.members(mask, n)
        total = np.zeros(n)
        perms = list(itertools.permutations(mem))
        for order in perms:
            cur = 0
            for i in order:
                total[i] += c[cur | (1 << i)] - c[cur]
                cur |= 1 << i
        assert X[mask] == pytest.approx(total / len(perms), abs=1e-9)
        assert X[mask].sum() == pytest.approx(c[mask])
    assert X[(1 << n) - 1] == pytest.approx(G.shapley(c, n))


def test_submodularity_and_cross_monotonicity_by_hand():
    """c(S) = Wurzel aus |S| ist submodular (konkav in der Größe); Shapley = c(S)/|S| je Mitglied und fällt mit der Gruppengröße: kreuzmonoton."""
    n = 4
    c = np.array([np.sqrt(bin(m).count("1")) for m in range(1 << n)])
    assert H.is_submodular(c, n)
    X = H.shapley_table(c, n)
    assert X[0b0111, 0] == pytest.approx(np.sqrt(3) / 3)
    assert H.is_cross_monotone(X, n)[0]


def test_non_submodular_and_non_cross_monotone_by_hand():
    """Drei Spieler: 1 und 2 allein je 10, gemeinsam 11 (starke Ersparnis); 3 allein 10, mit 1 oder 2 je 20 (kein Nutzen), alle drei 22. Die Grenzkosten von 1 steigen mit der Gruppe:
    c(1 | {2}) = 1, aber c(1 | {2,3}) = 22 - 20 = 2: nicht submodular."""
    c = np.zeros(8)
    c[0b001], c[0b010], c[0b100] = 10.0, 10.0, 10.0
    c[0b011], c[0b101], c[0b110], c[0b111] = 11.0, 20.0, 20.0, 22.0
    assert not H.is_submodular(c, 3)
    X = H.shapley_table(c, 3)
    assert X[0b111].sum() == pytest.approx(22.0)
    assert X[0b011] == pytest.approx([5.5, 5.5, 0.0])
    # Spediteur 1 über die 6 Reihenfolgen: 10 + 10 + 1 + 2 + 10 + 2 = 35 -> 35/6; Spediteur 3: 22 - 2 * 35/6
    assert X[0b111] == pytest.approx([35 / 6, 35 / 6, 22 - 70 / 6])
    assert X[0b111, 0] > X[0b011, 0]                                              # Spediteur 1 zahlt mit Spediteur 3 mehr als ohne
    assert not H.is_cross_monotone(X, 3)[0]


def test_members():
    assert H.members(0b1011, 4) == [0, 1, 3] and H.members(0, 4) == []
