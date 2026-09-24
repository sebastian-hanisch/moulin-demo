"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. kn_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import mo_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "layout_select": SettingSpec("layout", _choice(C.LAYOUTS), "uniform"),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "theta_slider": SettingSpec("theta", float, C.DEFAULT_THETA, C.THETA_MIN, C.THETA_MAX),
    "factor_slider": SettingSpec("factor", float, C.DEFAULT_FACTOR, C.FACTOR_MIN, C.FACTOR_MAX),
    "mech_select": SettingSpec("mech", _choice(C.MECHANISMS), "moulin_folk"),
    "liar_select": SettingSpec("liar", int, 1, 1, C.N_MAX),
}
PRESET_KEYS = {"n": "n_slider", "layout": "layout_select", "seed": "seed_input", "theta": "theta_slider", "factor": "factor_slider", "mech": "mech_select", "liar": "liar_select"}
STEPS = {"n_slider": C.N_STEP, "theta_slider": C.THETA_STEP, "factor_slider": C.FACTOR_STEP}

PRESETS = {
    "Standardfall (7 Spediteure)": {"n": 7, "layout": "uniform", "seed": 35, "theta": 1.5, "factor": 2.0, "mech": "moulin_folk", "liar": 1},
    "Ein Lügner gewinnt (Shapley)": {"n": 5, "layout": "uniform", "seed": 150, "theta": 1.6, "factor": 2.0, "mech": "moulin_shapley", "liar": 3},
    "Grenzkosten im Defizit": {"n": 6, "layout": "uniform", "seed": 35, "theta": 2.0, "factor": 2.0, "mech": "vcg", "liar": 1},
    "Zwei Ballungszentren": {"n": 7, "layout": "clustered", "seed": 35, "theta": 1.5, "factor": 2.0, "mech": "moulin_folk", "liar": 1},
    "Geringe Zahlungsbereitschaft": {"n": 7, "layout": "uniform", "seed": 35, "theta": 0.8, "factor": 2.0, "mech": "moulin_folk", "liar": 1},
    "Folk-Faktor 1,0 (Unterdeckung)": {"n": 7, "layout": "uniform", "seed": 35, "theta": 1.5, "factor": 1.0, "mech": "moulin_folk", "liar": 1},
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 1)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall (7 Spediteure)": "Alle bieten ehrlich. Das Wohlfahrtsmaximum (131,8 km) erreicht nur der Grenzkosten-Mechanismus (Spediteure 1, 2, 4, 5), der aber nur 96,7 von 174,4 km Tourkosten einnimmt. Moulin mit Shapley-Anteilen fährt mit 1, 4, 5 (89 % des Maximums, Kosten genau gedeckt), mit Spannbaum-Anteilen nur mit 4, 5 (81 %, 153,7 km eingenommen für 127,8 km Tour).",
    "Ein Lügner gewinnt (Shapley)": "5 Spediteure, Mechanismus Moulin mit Shapley-Anteilen: Spediteur 3 (Wert 38,0 km) fährt ehrlich nicht mit. Bietet er mindestens 39,51 km, bleibt er in der ersten Runde und gewinnt 4,15 km, weil Spediteur 5 dann aussteigt und sein Anteil von 39,51 auf 33,85 km fällt - die Shapley-Anteile sind nicht kreuzmonoton. Der Knopf \"Beste Lüge einsetzen\" zeigt es.",
    "Grenzkosten im Defizit": "6 Spediteure, Grenzkosten-Mechanismus: wohlfahrtsmaximal (233,9 km, Spediteure 1, 2, 4, 5), aber nur 81,3 von 174,4 km Tourkosten eingenommen (47 %). Für Einzelne lohnt sich keine Lüge; die Suche nach Zweier-Absprachen findet eine.",
    "Zwei Ballungszentren": "Stopps in zwei Ballungszentren, 7 Spediteure: der Grenzkosten-Mechanismus nimmt nur 9,1 von 50,7 km Tourkosten ein (18 %); Moulin mit Shapley-Anteilen erreicht hier das Maximum (84,4 km), mit Spannbaum-Anteilen 75 % (63,7 km).",
    "Geringe Zahlungsbereitschaft": "Niveau 0,8: die Werte reichen für keine Gruppe - das Wohlfahrtsmaximum ist 0 und kein Mechanismus fährt. Strategiesicherheit kostet hier nichts, weil es nichts zu gewinnen gibt.",
    "Folk-Faktor 1,0 (Unterdeckung)": "Faktor 1 auf die Spannbaum-Anteile: Moulin fährt mit den Spediteuren 1, 2, 4, 5 (Wohlfahrtsmaximum 131,8 km), nimmt aber nur 106,7 von 174,4 km Tourkosten ein (61 %) - der Spannbaum ist kürzer als die Tour, erst Faktor 2 deckt jede Tour.",
}
