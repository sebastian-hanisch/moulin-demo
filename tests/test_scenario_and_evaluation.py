"""Vehikel, Wert-Erzeugung und Auswertung (Analyse, vier Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import mo_constants as C
import mo_evaluation as E
import mo_game as G
import mo_mechanisms as M
import mo_scenario as S
import mo_shares as H


def test_generate_is_reproducible_and_shaped():
    a, b = S.generate(8, "uniform", 5), S.generate(8, "uniform", 5)
    assert np.array_equal(a.xy, b.xy) and a.xy.shape == (9, 2) and a.n == 8
    assert np.allclose(a.xy[0], [C.AREA / 2, C.AREA / 2]) and (a.xy >= 0).all() and (a.xy <= C.AREA).all()
    with pytest.raises(ValueError):
        S.generate(4, "bogus", 0)


def test_values_are_reproducible_scale_with_theta_and_stay_independent_of_the_stops():
    inst, dist, c, *_ = E.game(6, "uniform", 3)
    v1, v2 = E.make_values(c, 6, 1.0, 3), E.make_values(c, 6, 1.0, 3)
    assert np.array_equal(v1, v2) and np.allclose(E.make_values(c, 6, 2.0, 3), 2 * v1)
    alone = np.array([c[1 << i] for i in range(6)])
    assert (v1 >= 0).all() and (v1 <= alone).all()                                   # theta_i <= theta = 1
    assert not np.allclose(E.make_values(c, 6, 1.0, 4), v1)


def test_analyse_is_consistent():
    a = E.analyse(E.Settings(n=6))
    assert a.n == 6 and set(a.outcomes) == set(C.MECHANISMS) and a.v.shape == (6,)
    Se, We = M.efficient_set(a.v, a.c, 6)
    assert a.eff_set == Se and a.eff_welfare == pytest.approx(We) and a.outcomes["vcg"].served == Se
    for m in C.MECHANISMS:
        out = a.outcomes[m]
        assert a.welfare(m) == pytest.approx(M.welfare_of(a.v, out.served, a.c)) and a.welfare(m) <= a.eff_welfare + 1e-9
        assert a.revenue(m) == pytest.approx(out.pay.sum()) and a.cost(m) == pytest.approx(a.c[out.served])
    assert a.submodular == H.is_submodular(a.c, 6) and a.shapley_cm == H.is_cross_monotone(a.Xs, 6)[0]
    assert (a.shapley_violation is None) == a.shapley_cm


def test_moulin_shapley_recovers_exactly_and_moulin_folk_never_undercharges_at_factor_two():
    for seed in range(8):
        a = E.analyse(E.Settings(6, "uniform", seed, 1.5, 2.0))
        for m in ("moulin_shapley", "moulin_folk"):
            out = a.outcomes[m]
            if out.served:
                assert a.revenue(m) >= a.cost(m) - 1e-7
        if a.outcomes["moulin_shapley"].served:
            assert a.revenue("moulin_shapley") == pytest.approx(a.cost("moulin_shapley"), abs=1e-7)


def test_large_n_runs():
    a = E.analyse(E.Settings(n=C.N_MAX))
    assert a.n == C.N_MAX and a.eff_welfare >= 0


def test_cm_violation_returns_a_real_violation():
    a = E.analyse(E.Settings(5, "uniform", 150, 1.6, 2.0))
    i, s, j, before, after = a.shapley_violation
    assert after > before and (s >> i) & 1 and not (s >> j) & 1
    assert before == pytest.approx(a.Xs[s, i]) and after == pytest.approx(a.Xs[s | (1 << j), i])


def test_vcg_breakdown_adds_up():
    a = E.analyse(E.Settings(6, "uniform", 35, 2.0, 2.0))
    rows = E.vcg_breakdown(a)
    assert [r["carrier"] for r in rows] == H.members(a.outcomes["vcg"].served, 6)
    for r in rows:
        assert r["pay"] == pytest.approx(r["value"] - r["contribution"]) and r["with"] == pytest.approx(a.eff_welfare)
        assert r["pay"] == pytest.approx(a.outcomes["vcg"].pay[r["carrier"]])


def test_utility_curve_matches_direct_evaluation():
    a = E.analyse(E.Settings(5, "uniform", 150, 1.6, 2.0))
    grid = [0.0, 20.0, 38.0, 39.6, 80.0]
    u = a.utility_curve("moulin_shapley", 2, grid)
    for b, x in zip(grid, u):
        bids = a.v.copy()
        bids[2] = b
        assert x == pytest.approx(M.utility(a.v, a.run("moulin_shapley", bids), 2))
    assert u[2] == 0.0 and u[3] > 0


def test_prereq_experiment_shape_and_invariants():
    rows = E.prereq_experiment(ns=(4, 5), seeds=range(950000, 950006))
    assert [(r["n"], r["layout"]) for r in rows] == [(4, "uniform"), (4, "clustered"), (5, "uniform"), (5, "clustered")]
    for r in rows:
        assert r["folk_cm"] == 1.0 and r["folk_budget_gap"] < 1e-7 and 0 <= r["shapley_cm"] <= 1 and r["submodular"] <= r["shapley_cm"] + 1.0 and 1.0 <= r["overcharge"] <= r["overcharge_max"] <= 2.0 + 1e-9


def test_compare_experiment_shape_and_invariants():
    r = E.compare_experiment(n=5, seeds=range(960000, 960004), profiles=2)
    assert r["n_profiles"] == 8 and set(C.MECHANISMS) <= set(r)
    assert r["vcg"]["welfare_share"] == pytest.approx(1.0) and r["vcg"]["ind"] == 0.0
    assert r["moulin_folk"]["ind"] == 0.0 and r["moulin_folk"]["pair"] == 0.0 and r["moulin_folk"]["deficit"] == 0.0
    for m in C.MECHANISMS:
        assert 0 <= r[m]["welfare_share"] <= 1.0 + 1e-9 and 0 <= r[m]["served"] <= 5


def test_factor_experiment_monotone_trends():
    rows = E.factor_experiment(n=5, seeds=range(970000, 970008), levels=(1.0, 1.5, 2.0), profiles=2)
    assert [r["factor"] for r in rows] == [1.0, 1.5, 2.0]
    assert rows[-1]["deficit"] == 0.0 and rows[0]["deficit"] >= rows[-1]["deficit"]
    assert rows[0]["served"] >= rows[-1]["served"] and rows[0]["budget"] <= rows[-1]["budget"]


def test_manipulation_search_is_reproducible_and_folk_is_clean():
    a = E.manipulation_search(n=5, seeds=range(30), profiles=5)
    b = E.manipulation_search(n=5, seeds=range(30), profiles=5)
    assert a == b and a["folk_hits"] == 0 and a["profiles"] == 5 * a["non_cm_instances"] <= 150
