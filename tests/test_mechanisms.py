"""Moulin-Mechanismus und Grenzkosten-Mechanismus: Handrechnungen, Eigenschaften (Freiwilligkeit, größte stabile Menge, Reihenfolge-Unabhängigkeit), Strategiesicherheit durch Vollprüfung."""

import itertools

import numpy as np
import pytest

import mo_evaluation as E
import mo_game as G
import mo_mechanisms as M
import mo_scenario as S
import mo_shares as H


def table_two(a01, a10, both):
    """Anteilstabelle für zwei Spediteure: allein 0: a01, allein 1: a10, beide: both (Liste)."""
    X = np.zeros((4, 2))
    X[1] = [a01, 0.0]
    X[2] = [0.0, a10]
    X[3] = both
    return X


def test_moulin_by_hand_for_two_carriers():
    X = table_two(10.0, 10.0, [6.0, 6.0])
    out = M.moulin(np.array([7.0, 7.0]), X, 2)
    assert out.served == 0b11 and out.pay == pytest.approx([6.0, 6.0]) and len(out.rounds) == 1
    out = M.moulin(np.array([5.0, 7.0]), X, 2)                     # 0 steigt aus (5 < 6); allein kostet 1 dann 10 > 7
    assert out.served == 0 and out.pay == pytest.approx([0.0, 0.0]) and [r[2] for r in out.rounds] == [[0], [1], []]
    out = M.moulin(np.array([12.0, 5.0]), X, 2)                    # 1 steigt aus; 0 zahlt allein 10 <= 12
    assert out.served == 0b01 and out.pay == pytest.approx([10.0, 0.0])
    out = M.moulin(np.array([6.0, 6.0]), X, 2)                     # Gebot genau auf dem Anteil bleibt
    assert out.served == 0b11


def test_moulin_factor_scales_the_shares():
    X = table_two(10.0, 10.0, [6.0, 6.0])
    out = M.moulin(np.array([7.0, 7.0]), X, 2, factor=0.5)
    assert out.pay == pytest.approx([3.0, 3.0])
    out = M.moulin(np.array([7.0, 7.0]), X, 2, factor=1.5)         # Anteil 9 > 7: beide steigen gleichzeitig aus
    assert out.served == 0


def test_vcg_by_hand_for_two_carriers():
    """c(1) = 4, c(2) = 6, c(12) = 7; Gebote 5 und 4: W({1}) = 1, W({2}) = -2, W({12}) = 2 -> beide fahren; ohne 1 ist das Maximum 0, ohne 2 ist es 1:
    p_1 = 5 - (2 - 0) = 3, p_2 = 4 - (2 - 1) = 3, zusammen 6 < 7: Defizit 1."""
    c = np.array([0.0, 4.0, 6.0, 7.0])
    out = M.vcg(np.array([5.0, 4.0]), c, 2)
    assert out.served == 0b11 and out.pay == pytest.approx([3.0, 3.0]) and out.pay.sum() < c[3]
    assert M.welfare_of(np.array([5.0, 4.0]), out.served, c) == pytest.approx(2.0)
    out = M.vcg(np.array([2.0, 2.0]), c, 2)                        # W({12}) = -3, W({1}) = -2: niemand fährt
    assert out.served == 0 and out.pay == pytest.approx([0.0, 0.0])


def test_efficient_set_matches_brute_force_and_prefers_the_larger_set_on_ties():
    c = G.tsp_values(S.generate(5, "uniform", 3).dist())[0]
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.random(5) * 120
        best, best_w = 0, 0.0
        for mask in range(1 << 5):
            w = sum(v[i] for i in range(5) if (mask >> i) & 1) - c[mask]
            if w > best_w + 1e-12:
                best, best_w = mask, w
        sett, w = M.efficient_set(v, c, 5)
        assert w == pytest.approx(best_w) and M.welfare_of(v, sett, c) == pytest.approx(best_w) and sett == best
    tie = np.array([0.0, 5.0, 5.0, 10.0])                             # c: allein 5, zusammen 10: Wohlfahrt bei v = (5, 5): 0, 0, 0 (alle gleich)
    assert M.efficient_set(np.array([5.0, 5.0]), tie, 2)[0] == 0b11


@pytest.fixture(scope="module")
def game5():
    inst, dist, c, parent, dp, Xs, Xf, mst = E.game(5, "uniform", 4)
    return c, Xs, Xf


def test_moulin_with_cross_monotone_shares_gives_the_largest_stable_group(game5):
    """Für kreuzmonotone Anteile ist die Endmenge die größte Menge S, in der jeder Bieter mindestens seinen Anteil bietet (Vollaufzählung)."""
    c, Xs, Xf = game5
    rng = np.random.default_rng(2)
    for _ in range(60):
        b = rng.random(5) * 90
        out = M.moulin(b, Xf, 5, 1.7)
        stable = [S_ for S_ in range(1, 1 << 5) if all(b[i] >= Xf[S_, i] * 1.7 - 1e-9 for i in H.members(S_, 5))]
        union = 0
        for S_ in stable:
            union |= S_
        assert out.served == union                                      # die Vereinigung stabiler Mengen ist stabil und die Endmenge
        for S_ in stable:
            assert S_ & ~out.served == 0


def test_moulin_result_does_not_depend_on_the_removal_order_for_cross_monotone_shares(game5):
    c, Xs, Xf = game5
    rng = np.random.default_rng(3)

    def sequential(b, order):
        S_ = (1 << 5) - 1
        changed = True
        while changed:
            changed = False
            for i in order:
                if (S_ >> i) & 1 and b[i] < Xf[S_, i] - 1e-9:
                    S_ &= ~(1 << i)
                    changed = True
        return S_

    for _ in range(40):
        b = rng.random(5) * 90
        simult = M.moulin(b, Xf, 5).served
        for order in itertools.islice(itertools.permutations(range(5)), 0, 120, 17):
            assert sequential(b, order) == simult


def test_moulin_is_voluntary_and_never_pays_out(game5):
    c, Xs, Xf = game5
    rng = np.random.default_rng(4)
    for X, f in ((Xf, 2.0), (Xs, 1.0)):
        for _ in range(60):
            b = rng.random(5) * 100
            out = M.moulin(b, X, 5, f)
            for i in range(5):
                if (out.served >> i) & 1:
                    assert 0 <= out.pay[i] <= b[i] + 1e-9
                else:
                    assert out.pay[i] == 0.0


def test_vcg_is_individually_rational_and_efficient_on_the_honest_profile(game5):
    c, Xs, Xf = game5
    rng = np.random.default_rng(5)
    for _ in range(40):
        v = rng.random(5) * 110
        out = M.vcg(v, c, 5)
        Se, We = M.efficient_set(v, c, 5)
        assert out.served == Se and M.welfare_of(v, out.served, c) == pytest.approx(We)
        assert all(M.utility(v, out, i) >= -1e-9 for i in range(5))


def test_vcg_is_strategyproof_for_individuals(game5):
    c, Xs, Xf = game5
    rng = np.random.default_rng(6)
    run = lambda b: M.vcg(b, c, 5)
    for _ in range(25):
        v = rng.random(5) * 110
        for i in range(5):
            grid = list(M.vcg_candidates(v, c, 5, i)) + list(rng.random(15) * 250)
            assert M.best_deviation(run, v, i, grid)[0] == 0.0


def test_moulin_with_cross_monotone_shares_admits_no_lie_alone_or_in_pairs(game5):
    """Vollprüfung aller Schwellen-Gebote für jeden Einzelnen und jedes Paar (Gruppen-Strategiesicherheit, Moulin 1999)."""
    c, Xs, Xf = game5
    rng = np.random.default_rng(7)
    run = lambda b: M.moulin(b, Xf, 5, 2.0)
    cands = [M.moulin_candidates(Xf, 5, i, 2.0) for i in range(5)]
    for _ in range(12):
        v = rng.random(5) * 100
        assert all(M.best_deviation(run, v, i, cands[i])[0] == 0.0 for i in range(5))
        for i, j in itertools.combinations(range(5), 2):
            assert not M.pair_deviation(run, v, i, j, cands[i], cands[j])[0]


def test_candidates_cover_every_possible_behavior_of_the_mechanism(game5):
    """Zufällige Gebote eines Spediteurs führen zum selben Ergebnis wie das nächstkleinere Schwellen-Gebot."""
    c, Xs, Xf = game5
    rng = np.random.default_rng(8)
    cands = np.array(M.moulin_candidates(Xs, 5, 1, 1.0))
    for _ in range(60):
        v = rng.random(5) * 100
        b = float(rng.random() * 120)
        low = float(cands[cands <= b + 1e-12].max())
        v1, v2 = v.copy(), v.copy()
        v1[1], v2[1] = b, low
        o1, o2 = M.moulin(v1, Xs, 5), M.moulin(v2, Xs, 5)
        assert o1.served == o2.served and o1.pay == pytest.approx(o2.pay)


def test_shapley_mechanism_has_a_profitable_lie_on_the_preset_instance():
    a = E.analyse(E.Settings(5, "uniform", 150, 1.6, 2.0))
    gain, bid = M.best_deviation(lambda b: a.run("moulin_shapley", b), a.v, 2, a.candidates("moulin_shapley", 2))
    assert gain == pytest.approx(4.15, abs=0.005) and bid == pytest.approx(39.51, abs=0.005)
    assert M.utility(a.v, a.outcomes["moulin_shapley"], 2) == 0.0
    honest_shares = a.Xs[0b01111, 2], a.Xs[0b11111, 2]
    assert honest_shares[0] == pytest.approx(33.85, abs=0.005) and honest_shares[1] == pytest.approx(39.51, abs=0.005)      # nicht kreuzmonoton
    liar = a.v.copy()
    liar[2] = bid
    out = M.moulin(liar, a.Xs, 5)
    assert out.served == 0b01111 and a.v[2] - out.pay[2] == pytest.approx(gain)


def test_vcg_coalition_can_gain_on_the_preset_instance():
    a = E.analyse(E.Settings(6, "uniform", 35, 2.0, 2.0))
    found, scanned = E.pair_search(a, "vcg")
    assert found is not None
    i, j = found["i"], found["j"]
    out = a.run("vcg", np.where(np.arange(6) == i, found["bids"][0], np.where(np.arange(6) == j, found["bids"][1], a.v)))
    g_i = M.utility(a.v, out, i) - M.utility(a.v, a.outcomes["vcg"], i)
    g_j = M.utility(a.v, out, j) - M.utility(a.v, a.outcomes["vcg"], j)
    assert g_i >= -1e-6 and g_j >= -1e-6 and max(g_i, g_j) > 1e-6
