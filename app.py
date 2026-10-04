"""Moulin-Mechanismus - Kostenteilung, bei der Lügen sich nicht lohnt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Linie "Spieltheorie & Mechanism Design" der "Konzepte"-Reihe (Nachfolger von kern-demo, kooperativer Ast): Spediteure kennen ihre Zahlungsbereitschaft nur selbst. Wie teilt man die Kosten
einer gemeinsamen Tour so, dass ehrliches Bieten sich lohnt?

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import mo_constants as C
import mo_evaluation as E
import mo_mechanisms as M
import mo_shares as H
from mo_evaluation import Settings, analyse, compare_experiment, factor_experiment, manipulation_search, pair_search, prereq_experiment, vcg_breakdown
from mo_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from mo_visualization import build_compare, build_compare_experiment, build_factor, build_liar, build_outcome, build_prereq

st.set_page_config(page_title="Moulin-Kostenteilung – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def group_label(mask, n):
    mem = H.members(mask, n)
    if not mem:
        return "niemand"
    return ("Spediteure " if len(mem) > 1 else "Spediteur ") + ", ".join(str(i + 1) for i in mem)


@st.cache_data(show_spinner=False)
def _prereq(ns, seeds):
    return prereq_experiment(ns=ns, seeds=seeds)


@st.cache_data(show_spinner=False)
def _compare(n, seeds, profiles, theta, factor):
    return compare_experiment(n=n, seeds=seeds, profiles=profiles, theta=theta, factor=factor)


@st.cache_data(show_spinner=False)
def _factor(n, seeds, levels, theta):
    return factor_experiment(n=n, seeds=seeds, levels=levels, theta=theta)


@st.cache_data(show_spinner=False)
def _search(n, seeds, profiles, factor):
    return manipulation_search(n=n, seeds=seeds, profiles=profiles, factor=factor)


st.title("🗳️ Moulin-Mechanismus – Kosten teilen, wenn niemand ehrlich sein muss")
st.markdown(
    """
Bisher war in dieser Linie bekannt, was jeder Spediteur wert ist. Hier weiß das nur er selbst: seine **Zahlungsbereitschaft** $v_i$ - was ihm die gemeinsame Tour höchstens wert ist - ist privat, und er kann
bei der Anmeldung lügen. Ein **Mechanismus** legt fest, wer mitfährt und wer wie viel zahlt, allein aus den genannten Geboten. Der **Moulin-Mechanismus** (Moulin/Shenker 2001) macht ehrliches Bieten - auch in
Gruppen - zur besten Strategie, verlangt dafür aber ein Kostenteilungsverfahren mit einer Zusatzeigenschaft, der **Kreuzmonotonie**. Die Demo zeigt, wann das gelingt: mit dem Shapley-Wert der Vorgänger-Stücke
gelingt es in diesem Tourenspiel fast nie - eine Alternative über den Spannbaum gelingt immer, kostet aber Wohlfahrt.
"""
)
st.caption(
    "Neuntes Stück der Linie \"Spieltheorie & Mechanism Design\" der \"Konzepte\"-Reihe, Nachfolger von **kern-demo** (dasselbe Vehikel: Spediteure teilen die Tour; jetzt mit privaten Werten). "
    "Wohlfahrt = Summe der Werte der Mitfahrenden minus Tourlänge."
)

with st.expander("So funktioniert der Moulin-Mechanismus", expanded=True):
    st.markdown(
        """
1. **Gebote.** Jeder Spediteur nennt ein Gebot $b_i$. Ehrlich heißt $b_i = v_i$.
2. **Anteile.** Ein Kostenteilungsverfahren nennt für jede mögliche Gruppe $S$ den Anteil $\\xi_i(S)$, den Spediteur $i$ zahlen würde, wenn genau $S$ mitfährt.
3. **Aussteigen.** Start mit allen. Wer weniger bietet, als sein Anteil in der aktuellen Gruppe kostet, steigt aus; danach gelten die Anteile der kleineren Gruppe. Wiederholen, bis niemand mehr aussteigt.
   Wer bleibt, zahlt seinen Anteil.
4. **Kreuzmonotonie** heißt: $\\xi_i(S)$ steigt nie, wenn die Gruppe wächst. Dann kann Lügen nicht helfen: Zu hoch bieten hält einen nur in einer Gruppe, die zu teuer ist; zu niedrig bieten wirft einen früher
   hinaus - und auch ein Zusammenschluss mehrerer Lügner ändert daran nichts (Moulin 1999; Moulin/Shenker 2001).
5. **Wann gilt das für Shapley?** Moulin/Shenker: bei **submodularen** Kosten (abnehmende Grenzkosten) sind die Shapley-Anteile kreuzmonoton und decken die Kosten genau. Das Tourenspiel ist meist
   **nicht** submodular (Experiment unten) - dann greift die Garantie nicht.
6. **Alternative.** Der **Grenzkosten-Mechanismus** (VCG/Clarke) ist wohlfahrtsmaximal und für Einzelne ehrlich, macht aber Verlust. Und die **Folk-Anteile des Spannbaums** sind immer kreuzmonoton -
   dafür rechnet man mit dem Spannbaum statt mit der Tour (ein Faktor 2 deckt jede Tour).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_carriers = st.slider("Spediteure", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Spediteure mit je einem Stopp. Bis 9 sind alle Gruppen exakt durchgerechnet (2^n Touren).")
    layout = st.selectbox("Lage der Stopps", C.LAYOUTS, key="layout_select", format_func=lambda k: C.LAYOUT_LABELS[k], help="Gleichmäßig im Gebiet verteilt oder um zwei Ballungszentren gruppiert.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1, help="Legt die Lage der Stopps und - mit dem Niveau unten - die privaten Werte fest.")
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    theta = st.slider("Zahlungsbereitschaft (Niveau)", *bounds("theta_slider"), key="theta_slider", step=C.THETA_STEP,
                      help="Wert v_i = Zufallszahl zwischen 0 und diesem Niveau, mal die Kosten der Alleinfahrt von i. Bei 1,0 wollte im Mittel jeder die Hälfte seiner Alleinfahrt-Kosten zahlen.")
    factor = st.slider("Faktor auf die Folk-Anteile", *bounds("factor_slider"), key="factor_slider", step=C.FACTOR_STEP,
                       help="Der Moulin-Mechanismus mit Spannbaum-Anteilen multipliziert die Folk-Anteile mit diesem Faktor. 2,0 deckt jede Tour (Tour-Verdopplung); weniger ist billiger, aber nicht garantiert kostendeckend.")
    mech = st.selectbox("Mechanismus im Detail", C.MECHANISMS, key="mech_select", format_func=lambda k: C.MECHANISM_LABELS[k], help="Welcher Mechanismus in den Abschnitten unten Schritt für Schritt gezeigt wird.")

sync_query_params({"n_slider": int(n_carriers), "layout_select": layout, "seed_input": int(seed), "theta_slider": round(float(theta), 1), "factor_slider": round(float(factor), 1), "mech_select": mech})

settings = Settings(int(n_carriers), layout, int(seed), round(float(theta), 1), round(float(factor), 1))
with st.spinner("Rechne alle Gruppen durch..."):
    a = analyse(settings)
n = a.n
mech_label = C.MECHANISM_LABELS[mech]

# --- Das Vehikel --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wer wie viel bereit ist zu zahlen")
st.dataframe(
    {"Spediteur": [f"Spediteur {i + 1}" for i in range(n)], "Alleinfahrt (km)": [de(x) for x in a.alone], "Wert v_i (km)": [de(x) for x in a.v], "Wert / Alleinfahrt": [pct(a.v[i] / a.alone[i]) for i in range(n)],
     "Allein lohnend?": ["ja" if a.v[i] >= a.alone[i] else "nein" for i in range(n)]},
    hide_index=True,
)
alone_ok = int(sum(a.v[i] >= a.alone[i] for i in range(n)))
st.caption(
    f"Werte in km Tourlänge (Nutzen und Kosten in derselben Einheit). Allein ist die Tour nur für {alone_ok} von {n} Spediteuren lohnend; gemeinsam kostet die Tour aller {n} {de(a.c[-1])} km "
    f"(alle allein: {de(a.alone.sum())} km). Das Wohlfahrtsmaximum liegt bei {de(a.eff_welfare)} km, erreicht durch die Gruppe: {group_label(a.eff_set, n)}."
)

st.markdown("---")

# --- Drei Mechanismen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Drei Mechanismen, alle bieten ehrlich")
dev = {m: a.deviations(m) for m in C.MECHANISMS}
rows_cmp = []
for m in C.MECHANISMS:
    o = a.outcomes[m]
    rows_cmp.append({"Mechanismus": C.MECHANISM_LABELS[m], "Mitfahrer": group_label(o.served, n), "Wohlfahrt (km)": de(a.welfare(m)),
                     "Anteil am Maximum": pct(a.welfare(m) / a.eff_welfare) if a.eff_welfare > 1e-9 else "–", "Einnahmen (km)": de(a.revenue(m)), "Kosten der Tour (km)": de(a.cost(m)),
                     "Einnahmen − Kosten (km)": de(a.revenue(m) - a.cost(m)), "Einzel-Lüge lohnt": "ja" if max(g for g, _ in dev[m]) > 1e-6 else "nein"})
st.dataframe(rows_cmp, hide_index=True)
st.plotly_chart(build_compare(a), width="stretch", key="compare_chart")
mf, ms_, mv = a.outcomes["moulin_folk"], a.outcomes["moulin_shapley"], a.outcomes["vcg"]
notes = []
if a.eff_welfare > 1e-9:
    notes.append(f"Der wohlfahrtsmaximale Mechanismus (VCG) bedient {group_label(mv.served, n)}, zahlt aber nur {de(a.revenue('vcg'))} km der {de(a.cost('vcg'))} km Tour ein - ein Defizit von {de(a.cost('vcg') - a.revenue('vcg'))} km.")
    notes.append(f"Moulin mit Shapley-Anteilen deckt {'die Kosten genau' if abs(a.revenue('moulin_shapley') - a.cost('moulin_shapley')) < 1e-6 else 'die Kosten'} und erreicht {pct(a.welfare('moulin_shapley') / a.eff_welfare)} des Maximums; "
                 f"mit Spannbaum-Anteilen (Faktor {de(settings.factor)}) sind es {pct(a.welfare('moulin_folk') / a.eff_welfare)}, bei Einnahmen von {de(a.revenue('moulin_folk'))} km für {de(a.cost('moulin_folk'))} km Tour.")
else:
    notes.append("Bei diesen Werten lohnt sich die Tour für niemanden: das Wohlfahrtsmaximum ist 0 und kein Mechanismus fährt.")
st.info(" ".join(notes))

st.markdown("---")

# --- Der gewählte Mechanismus Schritt für Schritt -------------------------------------------------------------------------------------------------

st.markdown(f"## 🎯 {mech_label} – Schritt für Schritt")
out = a.outcomes[mech]
st.plotly_chart(build_outcome(a, mech), width="stretch", key="outcome_chart")
if mech == "vcg":
    st.caption("Der Grenzkosten-Mechanismus wählt die wohlfahrtsmaximale Gruppe; jeder Mitfahrer zahlt sein Gebot minus den Beitrag, den er zur Wohlfahrt leistet (Wohlfahrt mit ihm minus Wohlfahrt ohne ihn).")
    bd = vcg_breakdown(a)
    if bd:
        st.dataframe({"Spediteur": [f"Spediteur {r['carrier'] + 1}" for r in bd], "Wert (km)": [de(r["value"]) for r in bd], "Wohlfahrt mit allen (km)": [de(r["with"]) for r in bd],
                      "Wohlfahrt ohne ihn (km)": [de(r["without"]) for r in bd], "Beitrag (km)": [de(r["contribution"]) for r in bd], "Zahlung = Wert − Beitrag (km)": [de(r["pay"]) for r in bd]}, hide_index=True)
    else:
        st.info("Niemand fährt mit: es gibt keine Zahlungen.")
else:
    st.caption("Moulin-Mechanismus, Runde für Runde: wer weniger bietet, als sein Anteil in der aktuellen Gruppe kostet, steigt aus.")
    rounds = out.rounds
    st.dataframe({"Runde": [k + 1 for k in range(len(rounds))], "Gruppe": [group_label(r[0], n) for r in rounds],
                  "Anteile in dieser Gruppe (km)": [", ".join(f"{i + 1}: {de(r[1][i])}" for i in H.members(r[0], n)) or "–" for r in rounds],
                  "Steigen aus": [", ".join(str(i + 1) for i in r[2]) or "niemand" for r in rounds]}, hide_index=True)
    if out.served:
        st.caption(f"Endgruppe: {group_label(out.served, n)}; jeder zahlt seinen Anteil in dieser Gruppe (zusammen {de(a.revenue(mech))} km für eine Tour von {de(a.cost(mech))} km).")
    else:
        st.caption("Am Ende bleibt niemand übrig: kein Anteil war für die Verbleibenden bezahlbar.")

st.markdown("---")

# --- Lügen -------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Lohnt sich eine Lüge?")
st.caption(f"Wählen Sie einen Spediteur und ein anderes Gebot als sein wahres (alle anderen bieten ehrlich). Der Nutzen ist Wert minus Zahlung, wenn er mitfährt, sonst 0. Mechanismus: {mech_label}.")
if st.session_state.get("liar_select", 1) > n:
    st.session_state["liar_select"] = n
liar = st.selectbox("Spediteur, der lügt", list(range(1, n + 1)), key="liar_select", format_func=lambda i: f"Spediteur {i}")
li = int(liar) - 1
truth = float(a.v[li])
bmax = float(max(1.6 * a.alone[li], 2.0 * truth))
bid_key = f"liar_bid_{li}_{n}_{layout}_{int(seed)}_{settings.theta}_{settings.factor}_{mech}"
best_gain, best_bid = a.deviations(mech)[li]


def _set_bid(key, value):
    st.session_state[key] = min(value, bmax)


if bid_key not in st.session_state:
    st.session_state[bid_key] = round(truth, 2)
bid = st.number_input("Gebot (km)", min_value=0.0, max_value=bmax, step=0.5, key=bid_key, help="Das wahre Gebot ist der Wert des Spediteurs.")
if best_gain > 1e-6:
    st.button(f"💡 Beste Lüge einsetzen (Gebot {de(best_bid, 2)} km)", on_click=_set_bid, args=(bid_key, best_bid), key="best_lie")
else:
    st.caption("Für diesen Spediteur gibt es in diesem Mechanismus keine lohnende Einzel-Lüge (alle Schwellen durchgerechnet).")
cands = a.candidates(mech, li)
grid = np.array(sorted(set(np.linspace(0.0, bmax, 160).tolist() + [c for c in cands if c <= bmax] + [truth, float(bid)])))
utils = a.utility_curve(mech, li, grid)
st.plotly_chart(build_liar(a, mech, li, float(bid), grid, utils), width="stretch", key="liar_chart")
u_truth = M.utility(a.v, a.run(mech, a.v), li)
u_bid = M.utility(a.v, a.run(mech, np.where(np.arange(n) == li, float(bid), a.v)), li)
if u_bid > u_truth + 1e-6:
    st.warning(f"⚠️ Die Lüge lohnt sich: Nutzen {de(u_bid, 2)} km statt {de(u_truth, 2)} km bei ehrlichem Bieten (Gewinn {de(u_bid - u_truth, 2)} km).")
elif u_bid < u_truth - 1e-6:
    st.info(f"Die Lüge schadet: Nutzen {de(u_bid, 2)} km statt {de(u_truth, 2)} km bei ehrlichem Bieten.")
else:
    st.success(f"✅ Kein Unterschied: Nutzen {de(u_bid, 2)} km, genau wie bei ehrlichem Bieten ({de(u_truth, 2)} km).")
all_dev = {m: [g for g, _ in dev[m]] for m in C.MECHANISMS}
st.dataframe({"Mechanismus": [C.MECHANISM_LABELS[m] for m in C.MECHANISMS], "Größter Gewinn einer Einzel-Lüge (km)": [de(max(all_dev[m]), 2) for m in C.MECHANISMS],
              "Spediteure mit lohnender Lüge": [", ".join(str(i + 1) for i, g in enumerate(all_dev[m]) if g > 1e-6) or "keiner" for m in C.MECHANISMS]}, hide_index=True)

st.markdown("##### Zweier-Absprachen")
if n > 7:
    st.info("Die Suche nach Absprachen zu zweit prüft alle Gebotspaare und ist auf höchstens 7 Spediteure begrenzt.")
elif st.button("Nach Zweier-Absprachen suchen (dauert einige Sekunden)", key="pair_start"):
    with st.spinner("Prüfe alle Paare..."):
        found, scanned = pair_search(a, mech)
    if found:
        i, j = found["i"], found["j"]
        st.warning(f"⚠️ Ja: Spediteur {i + 1} bietet {de(found['bids'][0], 2)} km und Spediteur {j + 1} bietet {de(found['bids'][1], 2)} km statt der wahren {de(a.v[i], 2)} und {de(a.v[j], 2)} km - "
                   f"keiner der beiden steht schlechter da, und die Gewinne betragen {de(max(found['gains'][0], 0.0), 2)} bzw. {de(max(found['gains'][1], 0.0), 2)} km.")
    else:
        st.success(f"✅ Keine Absprache zu zweit gefunden ({scanned} Paare geprüft" + ("; alle Schwellen durchgerechnet" if mech != "vcg" else "; Gitter aus Vielfachen des Werts, also nur eine untere Grenze") + ").")

st.markdown("---")

# --- Warum Shapley hier nicht reicht ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ist der Shapley-Anteil kreuzmonoton?")
if a.shapley_cm:
    st.success("✅ Bei dieser Instanz sind die Shapley-Anteile kreuzmonoton: kein Spediteur zahlt mehr, wenn die Gruppe wächst. Die Moulin-Garantie gilt hier - aber nur für diese Instanz.")
else:
    i, s, j, before, after = a.shapley_violation
    st.warning(f"⚠️ Nein: Spediteur {i + 1} zahlt in der Gruppe {group_label(s, n)} {de(before, 2)} km, aber {de(after, 2)} km, wenn Spediteur {j + 1} dazukommt - mehr Mitfahrer, höherer Anteil. Die Moulin-Garantie für "
               "Shapley-Anteile gilt hier nicht.")
st.caption(f"Die Tourkosten sind {'submodular' if a.submodular else 'nicht submodular'} (abnehmende Grenzkosten für alle Gruppen: {'ja' if a.submodular else 'nein'}). "
           "Die Folk-Anteile des Spannbaums sind dagegen bei jeder Instanz kreuzmonoton (Beweis im 📐-Abschnitt, Test über alle Gruppen).")

st.markdown("---")

# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Sind die Voraussetzungen erfüllt?")
st.caption(f"Je {len(C.PREREQ_SEEDS)} feste Instanzen für {C.PREREQ_NS[0]} bis {C.PREREQ_NS[-1]} Spediteure und beide Lagen; geprüft werden alle Gruppen jeder Instanz.")
if st.button("Voraussetzungen messen", key="prereq_start"):
    st.session_state["prereq_on"] = True
if st.session_state.get("prereq_on"):
    with st.spinner("Prüfe alle Gruppen..."):
        rows_p = _prereq(C.PREREQ_NS, C.PREREQ_SEEDS)
    st.plotly_chart(build_prereq(rows_p), width="stretch", key="prereq_chart")
    uni = [r for r in rows_p if r["layout"] == "uniform"]
    st.warning(
        f"**Befund:** Die Tourkosten sind fast nie submodular - bei gleichmäßig verteilten Stopps in {pct(uni[0]['submodular'])} der Instanzen bei {uni[0]['n']} Spediteuren, in {pct(uni[-1]['submodular'])} bei {uni[-1]['n']}. "
        f"Entsprechend sind die Shapley-Anteile nur in {pct(uni[0]['shapley_cm'])} bzw. {pct(uni[-1]['shapley_cm'])} kreuzmonoton (bei Ballungszentren häufiger: {pct([r for r in rows_p if r['layout'] == 'clustered'][0]['shapley_cm'])} bis "
        f"{pct([r for r in rows_p if r['layout'] == 'clustered'][-1]['shapley_cm'])}). Die Folk-Anteile sind in {pct(min(r['folk_cm'] for r in rows_p))} aller Instanzen kreuzmonoton, und ihre Summe ist jedes Mal die Spannbaumlänge "
        f"(größte Abweichung {de(max(r['folk_budget_gap'] for r in rows_p), 6)} km). Der Preis: das Doppelte der Spannbaumlänge liegt im Mittel {pct(min(r['overcharge'] for r in rows_p) - 1)} bis {pct(max(r['overcharge'] for r in rows_p) - 1)} "
        f"über der Tourlänge (im Extremfall {pct(max(r['overcharge_max'] for r in rows_p) - 1)}; Alleinfahrten ausgenommen)."
    )

st.markdown("---")

# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die drei Mechanismen im Vergleich")
st.caption(f"{C.COMPARE_N} Spediteure, {len(C.COMPARE_SEEDS)} feste Instanzen mit je {C.COMPARE_PROFILES} Wert-Profilen (Niveau {de(settings.theta)}, Faktor {de(settings.factor)}); geprüft werden Wohlfahrt, Budget und jede mögliche "
           "Lüge allein und zu zweit.")
if st.button("Mechanismen vergleichen (dauert etwa 20 Sekunden)", key="compare_start"):
    st.session_state["compare_on"] = True
if st.session_state.get("compare_on"):
    with st.spinner("Prüfe alle Gebote..."):
        res_c = _compare(C.COMPARE_N, C.COMPARE_SEEDS, C.COMPARE_PROFILES, settings.theta, settings.factor)
    st.plotly_chart(build_compare_experiment(res_c), width="stretch", key="compare_experiment_chart")
    f_, s_, v_ = res_c["moulin_folk"], res_c["moulin_shapley"], res_c["vcg"]
    st.warning(
        f"**Befund** ({res_c['n_profiles']} Profile, im Mittel {de(res_c['eff_served'], 1)} Mitfahrer im Wohlfahrtsmaximum): "
        f"**Grenzkosten-Mechanismus:** {pct(v_['welfare_share'])} der Wohlfahrt, Einnahmen {pct(v_['budget'])} der Kosten, Defizit in {pct(v_['deficit'])} der gefahrenen Touren; lohnende Lüge allein in {pct(v_['ind'])}, zu zweit in "
        f"{pct(v_['pair'])} der Profile (untere Grenze, Gitter). "
        f"**Moulin mit Spannbaum-Anteilen** (Faktor {de(settings.factor)}): {pct(f_['welfare_share'])} der Wohlfahrt, Einnahmen {pct(f_['budget'])} der Kosten, Defizit in {pct(f_['deficit'])} der Touren; lohnende Lüge allein in "
        f"{pct(f_['ind'])}, zu zweit in {pct(f_['pair'])} der Profile (alle Schwellen durchgerechnet). "
        f"**Moulin mit Shapley-Anteilen:** {pct(s_['welfare_share'])} der Wohlfahrt, Einnahmen {pct(s_['budget'])} der Kosten; lohnende Lüge allein in {pct(s_['ind'])}, zu zweit in {pct(s_['pair'])} der Profile - obwohl "
        f"{res_c['non_cm_instances']} von {res_c['n_inst']} Instanzen nicht kreuzmonoton sind; wie selten Lügen dort sind, zeigt die gezielte Suche unten. "
        "Das ist das Dilemma von Moulin/Shenker: strategiesicher und kostendeckend kostet Wohlfahrt, wohlfahrtsmaximal macht Verlust und ist nur für Einzelne ehrlich."
    )

st.markdown("---")

# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was kostet der Faktor auf die Folk-Anteile?")
st.caption(f"{C.FACTOR_N} Spediteure, {len(C.FACTOR_SEEDS)} feste Instanzen mit je 3 Wert-Profilen (Niveau {de(settings.theta)}); der Faktor von 1,0 bis 2,0.")
if st.button("Faktor durchrechnen", key="factor_start"):
    st.session_state["factor_on"] = True
if st.session_state.get("factor_on"):
    rows_f = _factor(C.FACTOR_N, C.FACTOR_SEEDS, C.FACTOR_LEVELS, settings.theta)
    st.plotly_chart(build_factor(rows_f), width="stretch", key="factor_chart")
    lo_, hi_ = rows_f[0], rows_f[-1]
    st.warning(
        f"**Befund:** Bei Faktor {de(lo_['factor'])} sind {pct(lo_['deficit'])} der gefahrenen Touren nicht kostendeckend (Einnahmen {pct(lo_['budget'])} der Kosten), dafür werden {pct(lo_['welfare_share'])} der Wohlfahrt erreicht. "
        f"Bei Faktor {de(hi_['factor'])} ist die Deckung garantiert ({pct(hi_['deficit'])} Unterdeckung, Einnahmen {pct(hi_['budget'])} der Kosten), die Wohlfahrt sinkt auf {pct(hi_['welfare_share'])}. Der Faktor 2 ist nötig, "
        "weil eine Alleinfahrt doppelt so lang ist wie ihr Spannbaum (Hin- und Rückweg)."
    )

st.markdown("---")

# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Lohnt sich Lügen im Shapley-Mechanismus überhaupt?")
st.caption("Nur Instanzen mit nicht kreuzmonotonen Shapley-Anteilen; je Instanz zufällige Wert-Profile (5 Spediteure, gleichmäßig). Gezählt werden Profile, in denen irgendein Spediteur allein durch ein falsches Gebot "
           "gewinnt - im Shapley-Mechanismus und, als Kontrolle, im Folk-Mechanismus mit denselben Profilen.")
if st.button("Gezielt nach Lügen suchen", key="search_start"):
    st.session_state["search_on"] = True
if st.session_state.get("search_on"):
    with st.spinner("Suche..."):
        res_s = _search(5, tuple(range(60)), 20, settings.factor)
    c1, c2, c3 = st.columns(3)
    c1.metric("Profile in nicht kreuzmonotonen Instanzen", str(res_s["profiles"]))
    c2.metric("Lohnende Lüge im Shapley-Mechanismus", str(res_s["shapley_hits"]))
    c3.metric("Lohnende Lüge im Folk-Mechanismus", str(res_s["folk_hits"]))
    ex = res_s["example"]
    st.warning(
        f"**Befund:** In {res_s['shapley_hits']} von {res_s['profiles']} Profilen ({pct(res_s['shapley_hits'] / res_s['profiles'], 2)}) gewinnt ein Spediteur durch eine Lüge im Shapley-Mechanismus - selten, aber es gibt sie; "
        f"im Folk-Mechanismus mit denselben Profilen in {res_s['folk_hits']}. "
        + (f"Beispiel: Vehikel-Seed {ex['seed']}, Spediteur {ex['carrier'] + 1} mit Wert {de(ex['value'], 2)} km bietet {de(ex['bid'], 2)} km und gewinnt {de(ex['gain'], 2)} km. " if ex else "")
        + "Das Preset \"Ein Lügner gewinnt (Shapley)\" zeigt einen Fall zum Nachklicken."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Kosten sind submodular** | Dann sind Shapley-Anteile kreuzmonoton und Moulin ist kostendeckend und wohlfahrtsoptimal unter den strategiesicheren, budgetausgeglichenen Regeln (Moulin/Shenker 2001). Im Tourenspiel gilt das meist nicht (Experiment oben). | Spannbaum-Anteile (hier), Jain/Vazirani 2001 |
| **Kostendeckung und Effizienz zugleich** | Unmöglich: budgetausgeglichene, strategiesichere Mechanismen sind nicht wohlfahrtsmaximal, und der wohlfahrtsmaximale macht Verlust (Moulin/Shenker 2001; Experiment oben). | – |
| **Jeder Spediteur hat einen Stopp und eine Zahlungsbereitschaft** | Mit mehreren Stopps oder Wertfunktionen über Gruppen von Stopps wird aus der Frage "bedient oder nicht" eine kombinatorische Auktion. | auction-demo (Auktionen, VCG) |
| **Kosten und Werte in derselben Einheit** | Der Nutzen ist quasi-linear (Wert minus Zahlung); Budgetgrenzen und Risikoscheu sind nicht abgebildet. | – |
| **Die Kosten kennt der Mechanismus** | Hier sind die Kosten öffentlich bekannt (Tourlängen); Bieter, die sie melden, könnten sie verzerren. | myerson-satterthwaite-demo (Nachfolger) |
"""
)
st.caption(
    "Verwandt: [kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) (Vorgänger: Kern und Nukleolus), [shapley-demo](https://sebastianhanisch-shapley-demo.streamlit.app/) (der Shapley-Wert), "
    "[maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise, die Externalitäten einpreisen - der Zwilling der Kostenteilung)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Spediteure $N = \{1,\dots,n\}$, private Werte $v_i \ge 0$, Gebote $b_i$. Tourkosten $c(S)$ = kürzeste Rundtour ab Depot durch die Stopps in $S$ (Held-Karp). Nutzen: $u_i = v_i - p_i$ wenn $i$ bedient wird, sonst 0.
Wohlfahrt $W(S) = \sum_{i \in S} v_i - c(S)$.

**Moulin-Mechanismus** zu einem Kostenteilungsverfahren $\xi$: $S_0 = N$; $S_{k+1} = \{ i \in S_k : b_i \ge \xi_i(S_k) \}$ bis $S_{k+1} = S_k =: S^*$; $i \in S^*$ zahlt $\xi_i(S^*)$.
$\xi$ heißt **kreuzmonoton**, wenn $\xi_i(S) \ge \xi_i(T)$ für alle $i \in S \subseteq T$. Dann ist der Mechanismus gruppen-strategiesicher (Moulin 1999): kein Zusammenschluss kann jedem seiner Mitglieder weakly nützen und einem echt.

**Shapley-Anteile.** $\xi_i(S) = \sum_{T \subseteq S \setminus i} \frac{|T|!\,(|S|-|T|-1)!}{|S|!}\,[c(T \cup i) - c(T)]$; bei submodularem $c$ kreuzmonoton und $\sum_i \xi_i(S) = c(S)$.

**Folk-Anteile des Spannbaums.** Kruskal auf $S \cup \{0\}$; mit steigendem Schwellenwert $t$ bilden Stopps Cluster (Kanten $\le t$); jeder Cluster ohne Depot verteilt die Zuwachsrate 1 gleichmäßig auf seine Spediteure:
$\xi^F_i(S) = \int_0^\infty \mathbb 1[i \text{ nicht am Depot}]\,/\,|C_t(i)|\ dt$. Wächst $S$, werden Cluster größer und binden früher ans Depot an, also fällt der Integrand überall: **kreuzmonoton**. Die Summe ist die Spannbaumlänge
$\mathrm{MST}(S \cup 0)$, und $\mathrm{MST}(S \cup 0) \le \mathrm{TSP}(S) \le 2\,\mathrm{MST}(S \cup 0)$; der Mechanismus verwendet $\lambda\,\xi^F$ mit $\lambda \in [1, 2]$ ($\lambda = 2$ deckt jede Tour).

**Grenzkosten-Mechanismus (VCG).** $S^* = \arg\max_S \sum_{i \in S} b_i - c(S)$ (bei Gleichstand die größte Menge); $i \in S^*$ zahlt $p_i = b_i - [W_b(N) - W_b(N \setminus i)]$. Für Einzelne strategiesicher (Clarke), aber nicht gruppen-strategiesicher und
im Allgemeinen nicht kostendeckend.

Implementiert in `mo_shares.py` (Anteile), `mo_mechanisms.py` (Mechanismen, Lügensuche), `mo_evaluation.py` (Analyse, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spieltheorie: von Nash bis Myerson-Satterthwaite](https://sebastianhanisch.net/konzepte-spieltheorie.html)."
)
