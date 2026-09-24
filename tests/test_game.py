"""Kostenspiel: Held-Karp gegen Brute-Force, Shapley per Handrechnung und gegen Permutationen, die vier Axiome, Kern-Prüfung gegen eine Schleife, Vergleichs-Aufteilungen, Stichprobe."""

import itertools
import math

import numpy as np
import pytest

import mo_game as G
import mo_scenario as S


def brute_force_tour(dist, stops):
    if not stops:
        return 0.0
    best = float("inf")
    for perm in itertools.permutations(stops):
        length = dist[0, perm[0] + 1] + sum(dist[perm[i] + 1, perm[i + 1] + 1] for i in range(len(perm) - 1)) + dist[perm[-1] + 1, 0]
        best = min(best, length)
    return best


def shapley_by_permutations(c, n):
    total = np.zeros(n)
    for order in itertools.permutations(range(n)):
        mask = 0
        for i in order:
            total[i] += c[mask | (1 << i)] - c[mask]
            mask |= 1 << i
    return total / math.factorial(n)


def test_held_karp_agrees_with_brute_force_for_every_coalition():
    for seed in range(4):
        inst = S.generate(6, "uniform", seed)
        d = inst.dist()
        c, parent, dp = G.tsp_values(d)
        for mask in range(1 << 6):
            stops = [i for i in range(6) if (mask >> i) & 1]
            assert c[mask] == pytest.approx(brute_force_tour(d, stops))


def test_single_stop_costs_twice_the_distance_and_empty_coalition_is_free():
    inst = S.generate(5, "uniform", 2)
    d = inst.dist()
    c, _, _ = G.tsp_values(d)
    assert c[0] == 0.0 and all(c[1 << i] == pytest.approx(2 * d[0, i + 1]) for i in range(5))


def test_reconstructed_tour_has_the_optimal_length():
    inst = S.generate(7, "clustered", 5)
    d = inst.dist()
    c, parent, dp = G.tsp_values(d)
    for mask in (0b1111111, 0b1010101, 0b0000110):
        order = G.tour(d, parent, dp, mask)
        assert sorted(order) == [i for i in range(7) if (mask >> i) & 1]
        length = d[0, order[0] + 1] + sum(d[order[i] + 1, order[i + 1] + 1] for i in range(len(order) - 1)) + d[order[-1] + 1, 0]
        assert length == pytest.approx(c[mask])


def test_cooperation_never_costs_more_than_going_apart():
    """Dreiecksungleichung -> c(S u T) <= c(S) + c(T) für disjunkte S, T (Superadditivität der Ersparnis)."""
    inst = S.generate(8, "uniform", 9)
    c, _, _ = G.tsp_values(inst.dist())
    full = 1 << 8
    for s_mask in range(1, full, 7):
        for t_mask in range(1, full, 11):
            if s_mask & t_mask == 0:
                assert c[s_mask | t_mask] <= c[s_mask] + c[t_mask] + 1e-9


def test_shapley_by_hand_for_a_three_player_game():
    """c(1)=4, c(2)=6, c(3)=8, c(12)=9, c(13)=10, c(23)=11, c(123)=12. Grenzkosten je Reihenfolge (Spieler 1, 2, 3):
    123: 4, 5, 3   132: 4, 2, 6   213: 3, 6, 3   231: 1, 6, 5   312: 2, 2, 8   321: 1, 3, 8. Summen 15, 24, 33; durch 6 geteilt: 2,5 / 4,0 / 5,5 (zusammen 12)."""
    c = np.zeros(8)
    c[0b001], c[0b010], c[0b100] = 4.0, 6.0, 8.0
    c[0b011], c[0b101], c[0b110], c[0b111] = 9.0, 10.0, 11.0, 12.0
    assert G.shapley(c, 3) == pytest.approx([2.5, 4.0, 5.5])
    assert G.shapley(c, 3) == pytest.approx(shapley_by_permutations(c, 3))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_shapley_formula_agrees_with_the_permutation_definition(n):
    inst = S.generate(n, "uniform", 10 + n)
    c, _, _ = G.tsp_values(inst.dist())
    assert G.shapley(c, n) == pytest.approx(shapley_by_permutations(c, n), abs=1e-9)


def test_axiom_efficiency():
    for seed in range(5):
        inst = S.generate(7, "uniform", seed)
        c, _, _ = G.tsp_values(inst.dist())
        assert G.shapley(c, 7).sum() == pytest.approx(c[-1])


def test_axiom_symmetry_two_carriers_at_the_same_place_pay_the_same():
    inst = S.generate(5, "uniform", 3)
    xy = inst.xy.copy()
    xy[2] = xy[1]                                                 # Stopp 2 = Stopp 1
    dist = S.Instance(xy, 0, "uniform").dist()
    c, _, _ = G.tsp_values(dist)
    phi = G.shapley(c, 5)
    assert phi[0] == pytest.approx(phi[1])


def test_axiom_null_player_a_stop_at_the_depot_pays_nothing():
    inst = S.generate(5, "uniform", 4)
    xy = inst.xy.copy()
    xy[3] = xy[0]                                                 # Stopp 3 liegt im Depot: kostet nie etwas
    dist = S.Instance(xy, 0, "uniform").dist()
    c, _, _ = G.tsp_values(dist)
    assert G.shapley(c, 5)[2] == pytest.approx(0.0, abs=1e-9)
    assert c[1 << 2] == pytest.approx(0.0, abs=1e-9)


def test_axiom_additivity():
    n = 6
    c1, _, _ = G.tsp_values(S.generate(n, "uniform", 1).dist())
    c2, _, _ = G.tsp_values(S.generate(n, "clustered", 2).dist())
    assert G.shapley(c1 + c2, n) == pytest.approx(G.shapley(c1, n) + G.shapley(c2, n))
    assert G.shapley(3.0 * c1, n) == pytest.approx(3.0 * G.shapley(c1, n))


def test_core_excess_agrees_with_a_loop_over_all_coalitions():
    inst = S.generate(6, "uniform", 8)
    c, _, _ = G.tsp_values(inst.dist())
    x = G.proportional(c, 6)
    worst, worst_mask = -1e18, None
    for mask in range(1, (1 << 6) - 1):
        ex = sum(x[i] for i in range(6) if (mask >> i) & 1) - c[mask]
        if ex > worst:
            worst, worst_mask = ex, mask
    ex, mask = G.core_excess(x, c, 6)
    assert ex == pytest.approx(worst) and mask == worst_mask
    assert G.blocking_count(x, c, 6) == sum(1 for m in range(1, 63) if sum(x[i] for i in range(6) if (m >> i) & 1) - c[m] > 1e-9)


def test_all_three_allocations_are_efficient():
    inst = S.generate(6, "uniform", 6)
    c, _, _ = G.tsp_values(inst.dist())
    for x in (G.shapley(c, 6), G.proportional(c, 6), G.equal_savings(c, 6)):
        assert x.sum() == pytest.approx(c[-1])


def test_in_core_by_hand():
    """Gleiche Kosten für jeden allein (10) und für jedes Paar (15), für alle drei 18: die Aufteilung (6, 6, 6) ist im Kern, (9, 9, 0) nicht (Paar 1,2 zahlt 18 > 15)."""
    c = np.zeros(8)
    c[[1, 2, 4]] = 10.0
    c[[3, 5, 6]] = 15.0
    c[7] = 18.0
    assert G.in_core(np.array([6.0, 6.0, 6.0]), c, 3)
    assert not G.in_core(np.array([9.0, 9.0, 0.0]), c, 3)
    assert G.core_excess(np.array([9.0, 9.0, 0.0]), c, 3)[0] == pytest.approx(3.0)
    assert not G.in_core(np.array([5.0, 5.0, 5.0]), c, 3)             # nicht effizient


def test_proportional_and_equal_savings_by_hand():
    c = np.zeros(8)
    c[[1, 2, 4]] = [10.0, 20.0, 30.0]
    c[7] = 45.0
    prop = G.proportional(c, 3)
    assert prop == pytest.approx([7.5, 15.0, 22.5])
    eq = G.equal_savings(c, 3)
    assert eq == pytest.approx([10 - 5, 20 - 5, 30 - 5]) and eq.sum() == pytest.approx(45.0)


def test_sampling_is_reproducible_and_converges_to_the_exact_value():
    inst = S.generate(6, "uniform", 5)
    c, _, _ = G.tsp_values(inst.dist())
    exact = G.shapley(c, 6)
    assert np.array_equal(G.shapley_sampled(c, 6, 50, seed=1), G.shapley_sampled(c, 6, 50, seed=1))
    assert G.shapley_sampled(c, 6, 50, seed=1).sum() == pytest.approx(c[-1])            # jede Reihenfolge ist effizient
    err_small = np.abs(G.shapley_sampled(c, 6, 20, seed=2) - exact).sum()
    err_large = np.abs(G.shapley_sampled(c, 6, 20000, seed=2) - exact).sum()
    assert err_large < err_small and err_large < 0.02 * exact.sum()


def test_sampling_over_all_permutations_is_exactly_the_shapley_value():
    """Mittel über alle n! Reihenfolgen = Shapley-Wert (Definition); für n = 4 in Handarbeit aufsummiert."""
    inst = S.generate(4, "uniform", 3)
    c, _, _ = G.tsp_values(inst.dist())
    assert shapley_by_permutations(c, 4) == pytest.approx(G.shapley(c, 4))


def test_marginal_costs_by_hand():
    c = np.zeros(8)
    c[[1, 2, 4]] = [4.0, 6.0, 8.0]
    c[[3, 5, 6]] = [9.0, 10.0, 11.0]
    c[7] = 12.0
    assert G.marginal_costs(c, [0, 1, 2]) == {0: 4.0, 1: 5.0, 2: 3.0}
    assert G.marginal_costs(c, [2, 1, 0]) == {2: 8.0, 1: 3.0, 0: 1.0}
    assert G.coalition_members(0b101, 3) == [0, 2]
