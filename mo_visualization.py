"""Plotly-Abbildungen der Moulin-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mo_constants as C
from mo_mechanisms import utility

LINE_COLOR = "#4c78a8"
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
BAD = "#e45756"
WARN = "#f58518"
PURPLE = "#b279a2"
MECH_COLORS = {"moulin_folk": PURPLE, "moulin_shapley": GOOD, "vcg": WARN}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_outcome(an, mech):
    """Je Spediteur: Wert (hell), Zahlung (dunkel, nur wenn bedient) und Kosten der Alleinfahrt (Raute)."""
    n = an.n
    out = an.outcomes[mech]
    names = [f"Spediteur {i + 1}" for i in range(n)]
    served = [(out.served >> i) & 1 for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=an.v, orientation="h", marker_color=["#c7d9ec" if s else "#e6e6e6" for s in served], name="Wert (Zahlungsbereitschaft)", hovertemplate="Wert %{x:.1f} km<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=out.pay, orientation="h", marker_color=[MECH_COLORS[mech] if s else "rgba(0,0,0,0)" for s in served], name="Zahlung (nur wenn bedient)", width=0.35,
                         hovertemplate="Zahlung %{x:.1f} km<extra></extra>"))
    fig.add_trace(go.Scatter(y=names, x=an.alone, mode="markers", marker=dict(symbol="diamond", size=10, color=REF_COLOR, line=dict(color="white", width=1)), name="Kosten der Alleinfahrt",
                             hovertemplate="allein %{x:.1f} km<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="km Tourlänge")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 130 + 34 * n).update_layout(margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))


def build_compare(an):
    """Links: erreichte Wohlfahrt je Mechanismus gegen das Wohlfahrtsmaximum; rechts: Einnahmen gegen Kosten der gefahrenen Tour."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Wohlfahrt (km)", "Einnahmen und Kosten (km)"), horizontal_spacing=0.15)
    names = ["Moulin<br>Spannbaum (Folk)", "Moulin<br>Shapley", "Grenzkosten<br>(VCG)"]
    fig.add_trace(go.Bar(x=names, y=[an.welfare(m) for m in C.MECHANISMS], marker_color=[MECH_COLORS[m] for m in C.MECHANISMS], name="Wohlfahrt", showlegend=False,
                         text=[f"{an.welfare(m):.1f}" for m in C.MECHANISMS], textposition="outside"), row=1, col=1)
    fig.add_hline(y=an.eff_welfare, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Maximum", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[an.revenue(m) for m in C.MECHANISMS], marker_color=LINE_COLOR, name="Einnahmen (Zahlungen)", text=[f"{an.revenue(m):.1f}" for m in C.MECHANISMS], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=[an.cost(m) for m in C.MECHANISMS], marker_color=REF_COLOR, name="Kosten der Tour", text=[f"{an.cost(m):.1f}" for m in C.MECHANISMS], textposition="outside"), row=1, col=2)
    top = max([an.eff_welfare] + [an.revenue(m) for m in C.MECHANISMS] + [an.cost(m) for m in C.MECHANISMS] + [1.0]) * 1.2
    fig.update_yaxes(range=[0, top], row=1, col=1)
    fig.update_yaxes(range=[0, top], row=1, col=2)
    fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_liar(an, mech, i, bid, grid, utilities):
    """Nutzen von Spediteur i in Abhängigkeit vom Gebot; senkrecht: das wahre Gebot (grün) und das gewählte (orange)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(grid), y=list(utilities), mode="lines", line=dict(color=MECH_COLORS[mech], width=2.5, shape="hv"), name="Nutzen", hovertemplate="Gebot %{x:.1f} km: Nutzen %{y:.1f} km<extra></extra>"))
    truth = an.v[i]
    u_truth = float(np.interp(truth, grid, utilities))
    fig.add_trace(go.Scatter(x=[truth], y=[u_truth], mode="markers", marker=dict(size=13, color=GOOD, symbol="circle", line=dict(color="white", width=1)), name="ehrlich bieten"))
    u_bid = an.run(mech, np.where(np.arange(an.n) == i, bid, an.v))
    fig.add_trace(go.Scatter(x=[bid], y=[utility(an.v, u_bid, i)], mode="markers", marker=dict(size=13, color=WARN, symbol="diamond", line=dict(color="white", width=1)), name="gewähltes Gebot"))
    fig.update_xaxes(title_text=f"Gebot von Spediteur {i + 1} (km)")
    fig.update_yaxes(title_text="Nutzen (km)", rangemode="tozero")
    return _base(fig, 320)


def build_prereq(rows):
    """Je (Spediteure, Lage): Anteil der Instanzen mit submodularen Kosten, mit kreuzmonotonen Shapley-Anteilen und mit kreuzmonotonen Folk-Anteilen (Prozent)."""
    labels = [f"{r['n']} · {'gleichm.' if r['layout'] == 'uniform' else 'Ballung'}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["submodular"] for r in rows], name="Kosten submodular", marker_color=REF_COLOR))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["shapley_cm"] for r in rows], name="Shapley-Anteile kreuzmonoton", marker_color=GOOD))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["folk_cm"] for r in rows], name="Folk-Anteile kreuzmonoton", marker_color=PURPLE))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Spediteure · Lage der Stopps")
    fig.update_yaxes(title_text="% der Instanzen", range=[0, 110])
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.35))


def build_compare_experiment(res):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Wohlfahrt (% des Maximums)", "Einnahmen / Kosten (%)", "Profile mit lohnender Lüge (%)"), horizontal_spacing=0.09)
    labels = ["Moulin-<br>Folk", "Moulin-<br>Shapley", "VCG"]
    ms = list(C.MECHANISMS)
    fig.add_trace(go.Bar(x=labels, y=[100 * res[m]["welfare_share"] for m in ms], marker_color=[MECH_COLORS[m] for m in ms], showlegend=False, text=[f"{100 * res[m]['welfare_share']:.0f}" for m in ms], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[100 * res[m]["budget"] for m in ms], marker_color=[MECH_COLORS[m] for m in ms], showlegend=False, text=[f"{100 * res[m]['budget']:.0f}" for m in ms], textposition="outside"), row=1, col=2)
    fig.add_hline(y=100, line=dict(color=REF_COLOR, dash="dot"), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=[100 * res[m]["ind"] for m in ms], name="allein", marker_color=LINE_COLOR, text=[f"{100 * res[m]['ind']:.0f}" for m in ms], textposition="outside"), row=1, col=3)
    fig.add_trace(go.Bar(x=labels, y=[100 * res[m]["pair"] for m in ms], name="zu zweit", marker_color=BAD, text=[f"{100 * res[m]['pair']:.0f}" for m in ms], textposition="outside"), row=1, col=3)
    fig.update_yaxes(range=[0, 125], row=1, col=1)
    fig.update_yaxes(range=[0, 125], row=1, col=2)
    fig.update_yaxes(range=[0, 105], row=1, col=3)
    fig.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_factor(rows):
    xs = [r["factor"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["welfare_share"] for r in rows], mode="lines+markers", name="Wohlfahrt (% des Maximums)", line=dict(color=PURPLE, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["budget"] for r in rows], mode="lines+markers", name="Einnahmen / Kosten (%)", line=dict(color=LINE_COLOR, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["deficit"] for r in rows], mode="lines+markers", name="Touren mit Unterdeckung (%)", line=dict(color=BAD, width=2.5)))
    fig.add_hline(y=100, line=dict(color=REF_COLOR, dash="dot"))
    fig.update_xaxes(title_text="Faktor auf die Folk-Anteile", dtick=0.2)
    fig.update_yaxes(title_text="Prozent", range=[0, 115])
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))
