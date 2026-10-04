# 🗳️ Moulin-Mechanismus – Kosten teilen, wenn niemand ehrlich sein muss

**[→ Demo live ausprobieren](https://sebastianhanisch-moulin-demo.streamlit.app/)**

Neuntes Stück der **Spieltheorie-&-Mechanism-Design-Linie** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning, und Nachfolger von [kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) im zweiten Ast
(kooperative Spieltheorie). Bisher war bekannt, was jeder Spediteur wert ist; hier ist die **Zahlungsbereitschaft privat**, und jeder kann bei der Anmeldung lügen.

## Warum dieses Problem

Spediteure teilen die Tour eines gemeinsamen Depots. Ein **Mechanismus** entscheidet allein aus den genannten Geboten, wer mitfährt und wer wie viel zahlt. Der **Moulin-Mechanismus** (Moulin/Shenker 2001)
macht ehrliches Bieten – auch in Gruppen – zur besten Strategie, wenn das zugrunde liegende Kostenteilungsverfahren **kreuzmonoton** ist (kein Anteil steigt, wenn die Gruppe wächst). Für **submodulare** Kosten
leisten das die Shapley-Anteile. Die Demo prüft, ob das im Tourenspiel der Vorgänger-Stücke gilt – und zeigt, was man tut, wenn nicht.

## Modell

**Vehikel B "Spediteurs-Kooperation"** (`mo_scenario.py`, wie in shapley-demo/kern-demo): $n$ Spediteure mit je einem Stopp in einem 100 × 100 km großen Gebiet, Depot in der Mitte, Koalitionskosten $c(S)$ = kürzeste
Rundtour (Held-Karp, `mo_game.py`). **Neu:** private Werte $v_i = \theta_i \cdot c(\{i\})$ mit $\theta_i \sim U(0, \theta)$ (eigener Zufallsstrom je Seed). Nutzen = $v_i$ − Zahlung, wenn bedient, sonst 0. Wohlfahrt =
Summe der Werte der Mitfahrer − Tourlänge.

Drei Mechanismen (`mo_mechanisms.py`):
- **Moulin mit Shapley-Anteilen** (`mo_shares.shapley_table`): jede Gruppe zahlt genau ihre Tour.
- **Moulin mit Spannbaum-Anteilen** (`mo_shares.folk_table`): die Kosten von $S$ sind der kürzeste Spannbaum über Depot und $S$, aufgeteilt nach der **Folk-Regel** (Kruskal, jeder Cluster ohne Depot verteilt die
  Zuwachsrate 1 gleichmäßig auf seine Spediteure); mit einem Faktor $\lambda \in [1,2]$ multipliziert. Kreuzmonoton, weil größere Gruppen Cluster nur vergrößern und früher ans Depot anbinden; die Summe ist die
  Spannbaumlänge, und $\mathrm{MST} \le \mathrm{TSP} \le 2\,\mathrm{MST}$ – $\lambda = 2$ deckt jede Tour.
- **Grenzkosten-Mechanismus (VCG/Clarke)**: wohlfahrtsmaximale Gruppe, jeder zahlt Gebot minus Beitrag zur Wohlfahrt.

**Lügen-Suche (exakt):** Beim Moulin-Mechanismus hängt der Ausgang nur davon ab, welche Schwellen (Anteile in irgendeiner Gruppe) das Gebot erreicht; Gebote genau auf den Schwellen decken jedes Verhalten ab.
Damit sind Einzel-Lügen und Zweier-Absprachen **vollständig** geprüft. Beim VCG-Mechanismus prüft ein Gitter (Vielfache des Werts plus Schwelle) – dort nur eine untere Grenze für Absprachen.

## Methodik

- **Handrechnungen:** Folk-Anteile (Depotabstände 3/5/4 → 3 und 4; zwei nahe Stopps 10 vom Depot → 6 und 6), Moulin mit zwei Spediteuren (alle Fälle), VCG mit zwei Spediteuren (Zahlungen 3 und 3, Defizit 1),
  Shapley-Anteile eines nicht-submodularen Drei-Spieler-Spiels (35/6 statt 5,5 bei Zutritt eines Dritten).
- **Gegenproben:** Folk-Summe gegen einen unabhängigen Prim (jede Gruppe), Kreuzmonotonie gegen eine Schleife über alle Obermengen-Paare, Shapley-Tabelle gegen die Permutations-Definition (jedes Teilspiel), Moulin-Endmenge
  = Vereinigung aller stabilen Mengen (Vollaufzählung) und unabhängig von der Aussteige-Reihenfolge, Schwellen-Gebote gegen Zufallsgebote.
- **Eigenschaften:** Moulin freiwillig und ohne Auszahlungen an Bieter, VCG wohlfahrtsmaximal (gegen Brute-Force) und individuell rational, individuell strategiesicher (Gitter + Zufallsgebote).
- **Literatur** (per Recherche geprüft, nicht nachgebaut): Moulin/Shenker 2001 ("Strategyproof sharing of submodular costs: budget balance versus efficiency", Economic Theory 18), Moulin 1999 ("Incremental cost sharing:
  characterization by coalition strategy-proofness"), Jain/Vazirani 2001 ("Applications of approximation algorithms to cooperative games", STOC; kreuzmonotone 2-budgetausgeglichene Steiner-Baum-Anteile),
  Norde/Moretti/Tijs 2004 (Spannbaum-Spiele und bevölkerungsmonotone Aufteilungen). Die Folk-Kreuzmonotonie ist hier **selbst hergeleitet** (Zeitintegral, Beweisskizze im 📐-Abschnitt) und über alle Gruppen exakt getestet.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Sind die Tourkosten submodular? | Fast nie: bei gleichmäßig verteilten Stopps in 27 % der 30 Instanzen mit 5 Spediteuren, in 0 % bei 8. Ballungszentren etwas häufiger (30 % bei 5, 0 % bei 8). | `test_tour_costs_are_rarely_submodular_and_shapley_rarely_cross_monotone` |
| Sind die Shapley-Anteile kreuzmonoton? | Nur selten: 43 % (5 Spediteure) bis 0 % (8) bei gleichmäßiger Verteilung, 90 % bis 23 % bei Ballungszentren. Die Moulin-Garantie für Shapley gilt also im Tourenspiel meist nicht. | dito |
| Sind es die Folk-Anteile? | In allen 240 Instanzen (30 je Größe und Lage, 5 bis 8 Spediteure), jede Gruppe geprüft; ihre Summe ist jedes Mal die Spannbaumlänge (Abweichung < 1e-6 km). | `test_folk_is_always_cross_monotone_and_sums_to_the_spanning_tree`, `test_folk_is_cross_monotone` |
| Was kostet der Spannbaum? | Das Doppelte der Spannbaumlänge liegt im Mittel 13 % bis 23 % über der Tourlänge (Gruppen ab 2 Spediteuren), im Extremfall 64 %. Für einzelne Spediteure ist es genau die Alleinfahrt. | `test_doubled_spanning_tree_overcharges_moderately` |
| Standardfall (7 Spediteure, Niveau 1,5) | Wohlfahrtsmaximum 131,8 km (Spediteure 1, 2, 4, 5, VCG). VCG nimmt nur 96,7 von 174,4 km Tourkosten ein (Defizit rund 78 km). Moulin-Shapley fährt mit 1, 4, 5: 89 % des Maximums, Kosten genau gedeckt (149,9 km). Moulin-Spannbaum fährt mit 4, 5: 81 %, 153,7 km eingenommen für 127,8 km Tour. | `test_preset_standard` |
| Drei Mechanismen im Vergleich (6 Spediteure, 120 Profile) | VCG: 100 % der Wohlfahrt, aber nur 59 % der Kosten eingenommen und in 99 % der gefahrenen Touren ein Defizit; für Einzelne kein Lügen-Gewinn, aber in 78 % der Profile eine lohnende **Absprache zu zweit** (mindestens). Moulin-Spannbaum: 79 % der Wohlfahrt, 118 % der Kosten eingenommen, **keine** lohnende Lüge allein oder zu zweit. Moulin-Shapley: 93 % der Wohlfahrt, Kosten genau gedeckt, in diesen 120 Profilen keine gefundene Lüge – obwohl 27 von 30 Instanzen nicht kreuzmonoton sind. | `test_mechanism_comparison` |
| Lohnt sich Lügen im Shapley-Mechanismus überhaupt? | Selten, aber ja: in nicht kreuzmonotonen Instanzen (5 Spediteure, 600 Zufallsprofile) in 1 von 600 Profilen (0,17 %); im Folk-Mechanismus mit denselben Profilen in 0. Beispiel (Preset "Ein Lügner gewinnt"): Spediteur 3 (Wert 38,0 km) fährt ehrlich nicht mit; bietet er 39,51 km, bleibt er in Runde 1, Spediteur 5 steigt aus, sein Anteil fällt von 39,51 auf 33,85 km, Gewinn 4,15 km. | `test_targeted_lie_search`, `test_shapley_mechanism_has_a_profitable_lie_on_the_preset_instance`, `test_preset_liar_wins` |
| Was kostet der Faktor auf die Folk-Anteile? | 7 Spediteure, 20 Instanzen: bei Faktor 1,0 sind alle gefahrenen Touren unterdeckt (Einnahmen 64 % der Kosten), die Wohlfahrt liegt bei 94 %; bei Faktor 2,0 ist die Deckung garantiert (Einnahmen 120 %), die Wohlfahrt sinkt auf 78 %. Faktor 2 ist nötig, weil eine Alleinfahrt doppelt so lang ist wie ihr Spannbaum. | `test_factor_trade_off` |
| Wenn niemand zahlen will | Niveau 0,8 (Preset): das Wohlfahrtsmaximum ist 0, kein Mechanismus fährt. | `test_preset_low_willingness` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Kosten sind submodular** | Dann sind Shapley-Anteile kreuzmonoton und Moulin ist kostendeckend und unter den strategiesicheren, budgetausgeglichenen Regeln wohlfahrtsoptimal (Moulin/Shenker 2001). Im Tourenspiel gilt das meist nicht (Befund oben); die Spannbaum-Anteile sind ein Ausweg mit Preis. | Jain/Vazirani 2001 (Steiner-Baum), hier: Folk-Anteile |
| **Kostendeckung und Effizienz zugleich** | Unmöglich: strategiesichere, budgetausgeglichene Mechanismen sind nicht wohlfahrtsmaximal, der wohlfahrtsmaximale macht Verlust (Moulin/Shenker 2001; hier gemessen). | – |
| **Jeder Spediteur hat einen Stopp und einen Wert** | Mit mehreren Stopps oder Werten für Gruppen von Stopps wird die Frage "bedient oder nicht" zur kombinatorischen Auktion. | auction-demo |
| **Quasilineare Nutzen** | Wert minus Zahlung; Budgetgrenzen und Risikoscheu sind nicht abgebildet. | – |
| **Die Kosten sind öffentlich** | Hier kennt der Mechanismus die Tourlängen; wer sie meldet, könnte sie verzerren. | myerson-satterthwaite-demo (Nachfolger) |

Die Vergleichs- und Faktor-Messungen gelten für diese Vehikelfamilie und die genannten Größen (6 bis 8 Spediteure), die Zweier-Absprachen beim VCG-Mechanismus sind eine untere Grenze (Gitter). Dass beim Shapley-Mechanismus
nur 1 Lüge in 600 Profilen gefunden wurde, sagt nichts darüber, dass es bei anderen Wert-Verteilungen nicht häufiger ist.

Verwandt: [kern-demo](https://sebastianhanisch-kern-demo.streamlit.app/) (Vorgänger), [shapley-demo](https://sebastianhanisch-shapley-demo.streamlit.app/) (der Shapley-Wert),
[maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise, die Externalitäten einpreisen – der Zwilling der Kostenteilung).

## Tests

Pytest-Suite (`pytest tests/ -v`): Held-Karp und Shapley-Grundlagen (aus shapley-demo/kern-demo übernommen), Folk-Anteile per Handrechnung und gegen einen unabhängigen Prim, Kreuzmonotonie (Vollprüfung), Shapley-Tabelle gegen die
Permutations-Definition, Moulin und VCG per Handrechnung, Eigenschaften und Strategiesicherheit durch Vollprüfung aller Schwellen-Gebote (einzeln und zu zweit), Vehikel und Experimente (Form, Reproduzierbarkeit),
AppTest-Rauchtests (jedes Preset, Lügen-Widget, Permalink-Grenzen, vier Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README; Einzelinstanzen auf die gezeigte Rundung, Mehr-Instanz-Zahlen mit
großzügigen Bändern).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `mo_constants.py` | Vehikel-Konstanten, Regler-Grenzen, Experiment-Seeds |
| `mo_presets.py` | Permalink/Presets-Mechanik |
| `mo_scenario.py` | Vehikel B (Depot, Stopps, Entfernungen) |
| `mo_game.py` | Held-Karp, Shapley, Vergleichs-Aufteilungen |
| `mo_shares.py` | Anteilstabellen (Shapley, Folk/Spannbaum), Kreuzmonotonie, Submodularität |
| `mo_mechanisms.py` | Moulin, VCG, Lügen-Suche (einzeln, Paare) |
| `mo_evaluation.py` | Analyse einer Instanz, vier Experimente |
| `mo_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Myerson-Satterthwaite (private Kosten und Werte beidseitig): eigenes Stück 10 der Linie ([myerson-satterthwaite-demo](https://sebastianhanisch-myerson-satterthwaite-demo.streamlit.app/)).
- Mehr als 9 Spediteure (alle $2^n$ Gruppen und ihre Touren in Python) und Absprachen zu dritt.
- Weitere kreuzmonotone Verfahren (Jain/Vazirani-Moat-Growing) – die Folk-Anteile genügen als Gegenpol.
- Ein PDF-Export – wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spieltheorie: von Nash bis Myerson-Satterthwaite](https://sebastianhanisch.net/konzepte-spieltheorie.html).
