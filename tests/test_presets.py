"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import mo_constants as C
import mo_evaluation as E
import mo_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for p in P.PRESETS.values():
        assert C.N_MIN <= p["n"] <= C.N_MAX and p["layout"] in C.LAYOUTS and 0 <= p["seed"] <= C.SEED_MAX and p["mech"] in C.MECHANISMS
        assert C.THETA_MIN <= p["theta"] <= C.THETA_MAX and C.FACTOR_MIN <= p["factor"] <= C.FACTOR_MAX and 1 <= p["liar"] <= p["n"]
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall (7 Spediteure)"]
    assert E.Settings(p["n"], p["layout"], p["seed"], p["theta"], p["factor"]) == E.Settings()
    assert p["mech"] == P.SETTING_SPECS["mech_select"].default


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and set(P.STEPS) == {"n_slider", "theta_slider", "factor_slider"}
    assert P.bounds("theta_slider") == (C.THETA_MIN, C.THETA_MAX) and P.bounds("factor_slider") == (C.FACTOR_MIN, C.FACTOR_MAX)


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_float_presets_sit_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in (("theta", "theta_slider"), ("factor", "factor_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9
