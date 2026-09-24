"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Lügen-Widget, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mo_constants as C
import mo_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_shows_the_three_mechanisms():
    at = _run()
    _ok(at)
    assert at.dataframe and any("Der wohlfahrtsmaximale Mechanismus (VCG) bedient" in i.value for i in at.info)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]
    assert at.get("plotly_chart")


def test_liar_preset_shows_the_profitable_lie_after_the_best_lie_button():
    at = _run()
    next(b for b in at.button if b.key == "preset_Ein Lügner gewinnt (Shapley)").click().run()
    _ok(at)
    best = next(b for b in at.button if b.key == "best_lie")
    assert "39,51" in best.label
    best.click().run()
    _ok(at)
    assert any("Die Lüge lohnt sich: Nutzen 4,15 km statt 0,00 km" in w.value for w in at.warning)


def test_no_best_lie_button_when_no_lie_pays():
    at = _run(mech_select="vcg")
    _ok(at)
    assert not any(b.key == "best_lie" for b in at.button)
    assert any("keine lohnende Einzel-Lüge" in c.value for c in at.caption)


def test_liar_choice_survives_shrinking_n():
    at = _run(n_slider=9, liar_select=9)
    _ok(at)
    at.slider(key="n_slider").set_value(4).run()
    _ok(at)
    assert at.session_state["liar_select"] == 4


def test_pair_search_button_runs_and_is_disabled_for_large_n():
    at = _run(n_slider=5, mech_select="moulin_folk")
    next(b for b in at.button if b.key == "pair_start").click().run()
    _ok(at)
    assert any("Keine Absprache zu zweit gefunden" in s.value for s in at.success)
    at = _run(n_slider=8)
    _ok(at)
    assert any("höchstens 7 Spediteure" in i.value for i in at.info) and not any(b.key == "pair_start" for b in at.button)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["theta"] = "1.57"
    at.query_params["factor"] = "0.2"
    at.query_params["mech"] = "vcg"
    at.query_params["layout"] = "clustered"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["theta_slider"] == 1.6 and at.session_state["factor_slider"] == C.FACTOR_MIN
    assert at.session_state["mech_select"] == "vcg" and at.session_state["layout_select"] == "clustered"


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(layout_select="clustered"), dict(theta_slider=C.THETA_MIN), dict(theta_slider=C.THETA_MAX), dict(factor_slider=C.FACTOR_MIN),
                                dict(n_slider=4, layout_select="clustered", seed_input=0, mech_select="moulin_shapley")])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


@pytest.mark.parametrize("mech", C.MECHANISMS)
def test_every_mechanism_view_runs(mech):
    at = _run(mech_select=mech)
    _ok(at)
    assert any(C.MECHANISM_LABELS[mech] in m.value for m in at.markdown)


def test_prereq_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "PREREQ_NS", (4, 5))
    monkeypatch.setattr(C, "PREREQ_SEEDS", tuple(range(950000, 950004)))
    at = _run()
    next(b for b in at.button if b.key == "prereq_start").click().run()
    _ok(at)
    assert at.session_state["prereq_on"] and any("Die Tourkosten sind fast nie submodular" in w.value for w in at.warning)


def test_compare_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "COMPARE_N", 5)
    monkeypatch.setattr(C, "COMPARE_SEEDS", tuple(range(960000, 960003)))
    monkeypatch.setattr(C, "COMPARE_PROFILES", 2)
    at = _run()
    next(b for b in at.button if b.key == "compare_start").click().run()
    _ok(at)
    assert at.session_state["compare_on"] and any("Das ist das Dilemma von Moulin/Shenker" in w.value for w in at.warning)


def test_factor_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "FACTOR_N", 5)
    monkeypatch.setattr(C, "FACTOR_SEEDS", tuple(range(970000, 970004)))
    monkeypatch.setattr(C, "FACTOR_LEVELS", (1.0, 2.0))
    at = _run()
    next(b for b in at.button if b.key == "factor_start").click().run()
    _ok(at)
    assert at.session_state["factor_on"] and any("Der Faktor 2 ist nötig" in w.value for w in at.warning)


def test_search_experiment_runs_on_demand():
    at = _run()
    next(b for b in at.button if b.key == "search_start").click().run()
    _ok(at)
    assert at.session_state["search_on"] and len(at.metric) >= 3 and any("Lohnende Lüge im Folk-Mechanismus" == m.label for m in at.metric)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "de(" not in el.value.replace("Kosten de(", "")
