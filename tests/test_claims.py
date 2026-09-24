"""Jede Zahl aus PRESET_HELP und README: Presets auf die gezeigte Rundung (deterministische Rechnung), Mehr-Instanz-Befunde mit großzügigen Bändern
(CI installiert die neueste numpy/scipy - keine Aussage hängt an einem einzelnen chaotischen Lauf)."""

import numpy as np
import pytest

import mo_constants as C
import mo_evaluation as E
import mo_presets as P
import mo_shares as H


def analyse_preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["n"], p["layout"], p["seed"], p["theta"], p["factor"]))


def served(a, m):
    return [i + 1 for i in H.members(a.outcomes[m].served, a.n)]


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_standard():
    a = analyse_preset("Standardfall (7 Spediteure)")
    assert a.eff_welfare == pytest.approx(131.8, abs=0.05) and H.members(a.eff_set, 7) == [0, 1, 3, 4]
    assert served(a, "vcg") == [1, 2, 4, 5] and a.revenue("vcg") == pytest.approx(96.7, abs=0.05) and a.cost("vcg") == pytest.approx(174.4, abs=0.05)
    assert served(a, "moulin_shapley") == [1, 4, 5] and a.welfare("moulin_shapley") / a.eff_welfare == pytest.approx(0.89, abs=0.005)
    assert a.revenue("moulin_shapley") == pytest.approx(a.cost("moulin_shapley")) and a.cost("moulin_shapley") == pytest.approx(149.9, abs=0.05)
    assert served(a, "moulin_folk") == [4, 5] and a.welfare("moulin_folk") / a.eff_welfare == pytest.approx(0.81, abs=0.005)
    assert a.revenue("moulin_folk") == pytest.approx(153.7, abs=0.05) and a.cost("moulin_folk") == pytest.approx(127.8, abs=0.05)
    assert a.revenue("vcg") - a.cost("vcg") == pytest.approx(-77.7, abs=0.15)


def test_preset_liar_wins():
    a = analyse_preset("Ein Lügner gewinnt (Shapley)")
    assert a.v[2] == pytest.approx(38.0, abs=0.05) and not a.shapley_cm
    i, s, j, before, after = a.shapley_violation
    assert (i, s, j) == (2, 0b01111, 4) and before == pytest.approx(33.85, abs=0.005) and after == pytest.approx(39.51, abs=0.005)
    gain, bid = a.deviations("moulin_shapley")[2]
    assert gain == pytest.approx(4.15, abs=0.005) and bid == pytest.approx(39.51, abs=0.005)
    assert max(g for g, _ in a.deviations("moulin_folk")) == 0.0 and max(g for g, _ in a.deviations("vcg")) == 0.0


def test_preset_vcg_deficit():
    a = analyse_preset("Grenzkosten im Defizit")
    assert a.eff_welfare == pytest.approx(233.9, abs=0.05) and served(a, "vcg") == [1, 2, 4, 5]
    assert a.revenue("vcg") == pytest.approx(81.3, abs=0.05) and a.cost("vcg") == pytest.approx(174.4, abs=0.05) and a.revenue("vcg") / a.cost("vcg") == pytest.approx(0.47, abs=0.005)
    assert max(g for g, _ in a.deviations("vcg")) == 0.0
    found, _ = E.pair_search(a, "vcg")
    assert found is not None


def test_preset_clustered():
    a = analyse_preset("Zwei Ballungszentren")
    assert a.revenue("vcg") == pytest.approx(9.1, abs=0.05) and a.cost("vcg") == pytest.approx(50.7, abs=0.05) and a.revenue("vcg") / a.cost("vcg") == pytest.approx(0.18, abs=0.005)
    assert a.eff_welfare == pytest.approx(84.4, abs=0.05) and a.welfare("moulin_shapley") == pytest.approx(a.eff_welfare)
    assert a.welfare("moulin_folk") == pytest.approx(63.7, abs=0.05) and a.welfare("moulin_folk") / a.eff_welfare == pytest.approx(0.75, abs=0.005)


def test_preset_low_willingness():
    a = analyse_preset("Geringe Zahlungsbereitschaft")
    assert a.eff_welfare == 0.0 and all(a.outcomes[m].served == 0 for m in C.MECHANISMS)


def test_preset_factor_one():
    a = analyse_preset("Folk-Faktor 1,0 (Unterdeckung)")
    assert served(a, "moulin_folk") == [1, 2, 4, 5] and a.welfare("moulin_folk") == pytest.approx(a.eff_welfare) and a.eff_welfare == pytest.approx(131.8, abs=0.05)
    assert a.revenue("moulin_folk") == pytest.approx(106.7, abs=0.05) and a.cost("moulin_folk") == pytest.approx(174.4, abs=0.05) and a.revenue("moulin_folk") / a.cost("moulin_folk") == pytest.approx(0.61, abs=0.005)


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prereq():
    return {(r["n"], r["layout"]): r for r in E.prereq_experiment()}


def test_tour_costs_are_rarely_submodular_and_shapley_rarely_cross_monotone(prereq):
    assert 0.15 < prereq[(5, "uniform")]["submodular"] < 0.40 and prereq[(8, "uniform")]["submodular"] <= 0.05           # gemessen 27 % -> 0 %
    assert 0.30 < prereq[(5, "uniform")]["shapley_cm"] < 0.60 and prereq[(8, "uniform")]["shapley_cm"] <= 0.10           # gemessen 43 % -> 0 %
    assert prereq[(5, "clustered")]["shapley_cm"] >= 0.75 and 0.10 < prereq[(8, "clustered")]["shapley_cm"] < 0.40       # gemessen 90 % -> 23 %
    assert 0.15 < prereq[(5, "clustered")]["submodular"] < 0.50 and prereq[(8, "clustered")]["submodular"] <= 0.05        # gemessen 30 % -> 0 %
    for n in C.PREREQ_NS:
        assert prereq[(n, "clustered")]["shapley_cm"] > prereq[(n, "uniform")]["shapley_cm"]                              # bei Ballungszentren häufiger
    assert all(r["shapley_cm"] < 1.0 for r in prereq.values() if r["n"] >= 6 and r["layout"] == "uniform")


def test_folk_is_always_cross_monotone_and_sums_to_the_spanning_tree(prereq):
    assert all(r["folk_cm"] == 1.0 and r["folk_budget_gap"] < 1e-6 for r in prereq.values())


def test_doubled_spanning_tree_overcharges_moderately(prereq):
    means = [r["overcharge"] for r in prereq.values()]
    assert 1.10 < min(means) and max(means) < 1.26                                          # gemessen 13 % bis 23 %
    assert 1.4 < max(r["overcharge_max"] for r in prereq.values()) <= 1.70                  # gemessen 64 % im Extremfall


@pytest.fixture(scope="module")
def compare():
    return E.compare_experiment()


def test_mechanism_comparison(compare):
    f, s, v = compare["moulin_folk"], compare["moulin_shapley"], compare["vcg"]
    assert compare["n_profiles"] == 120 and compare["non_cm_instances"] >= 24                # gemessen 27 von 30
    assert v["welfare_share"] == pytest.approx(1.0) and 0.50 < v["budget"] < 0.68 and v["deficit"] > 0.9 and v["ind"] == 0.0     # gemessen 59 %, 99 %
    assert 0.6 < v["pair"] < 0.95                                                           # gemessen 78 % (Gitter: untere Grenze)
    assert 0.70 < f["welfare_share"] < 0.86 and 1.10 < f["budget"] < 1.25 and f["deficit"] == 0.0 and f["ind"] == 0.0 and f["pair"] == 0.0        # gemessen 79 %, 118 %
    assert 0.88 < s["welfare_share"] < 0.97 and s["budget"] == pytest.approx(1.0) and s["ind"] <= 0.05 and s["pair"] <= 0.05                   # gemessen 93 %, 100 %
    assert f["welfare_share"] < s["welfare_share"] < v["welfare_share"]


@pytest.fixture(scope="module")
def factor_rows():
    return E.factor_experiment()


def test_factor_trade_off(factor_rows):
    lo, hi = factor_rows[0], factor_rows[-1]
    assert lo["factor"] == 1.0 and hi["factor"] == 2.0
    assert lo["deficit"] > 0.95 and 0.55 < lo["budget"] < 0.72 and lo["welfare_share"] > 0.88                   # gemessen 100 %, 64 %, 94 %
    assert hi["deficit"] == 0.0 and 1.10 < hi["budget"] < 1.30 and 0.70 < hi["welfare_share"] < 0.86         # gemessen 0 %, 120 %, 78 %
    d = [r["deficit"] for r in factor_rows]
    w = [r["welfare_share"] for r in factor_rows]
    assert all(d[i] >= d[i + 1] - 1e-9 for i in range(len(d) - 1))                              # Unterdeckung fällt mit dem Faktor
    assert w[0] > w[-1] + 0.10 and all(w[i] >= w[i + 1] - 1e-9 for i in range(1, len(w) - 1))   # Wohlfahrt fällt (von 1,0 auf 1,2 leicht steigend: 93,9 -> 94,0 %)


def test_targeted_lie_search():
    r = E.manipulation_search(n=5, seeds=range(60), profiles=20)
    assert r["profiles"] == 600 and r["non_cm_instances"] == 30 and r["folk_hits"] == 0                # gemessen: 1 von 600 im Shapley-Mechanismus
    assert 1 <= r["shapley_hits"] <= 6 and r["shapley_hits"] / r["profiles"] < 0.02
    assert r["example"]["gain"] > 0
