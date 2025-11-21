# mvf_utvikling.py
# Interaktiv rapport:
# - Pan (drag), scroll-zoom
# - Dropdown som FREMHEVER valgt brudd (kraftig rød overlay + rød kant + annotasjon) – ingen auto-zoom
# - Svak rød bakgrunn i alle bruddperioder (baseline)
# - Statistikk som "knapp" (<details>/<summary>) i rapport_interaktiv.html
# - Kun HTML (ingen PNG/PDF)
# - Stor figur og god top-margin (ingen overlapp)
#
# Bruddlogikk:
# Brudd når (Min.fillna(0) + Overløp.fillna(0)) < Krav  OG (hvis tilgjengelig) Effekt > EFF_MIN_MW,
# MEN kun på tidssteg der minst én av {Min, Overløp} er DOKUMENTERT (ikke NaN).
# -> Helt "tomme" punkter (Min=NaN og Overløp=NaN) kan ALDRI bidra til brudd (de bryter segmentet).
#
# Visualisering:
# - Alle serier får egen kontinuitet (brudd) ut fra egne NaN/gap.
# - "Datahull" i legend markerer lys gule bånd der Min mangler (sist i legend, av som standard).
# - Overløp og Effekt vises også når Min mangler.

from pathlib import Path
from datetime import datetime
import webbrowser
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# === KONFIG ===
EXCEL = Path("data_rapport.xlsx")  # <-- endre ved behov
SHEET = "data"

COL_DATO = "Dato"
COL_MVF  = "Minstevannføring (l/s)"
COL_KRAV = "Krav (l/s)"
COL_EFF  = "Effekt produksjon (MW)"
COL_OVER = "Overløp vannføring (l/s)"  # Kan mangle

# Effekt-terskel for å ignorere spiker/støy
EFF_MIN_MW = 0.05  # <-- endre om du vil (f.eks. 0.1)

# Hull-deteksjon (robust for ulik oppløsning) for "Datahull" (basert på Min):
# gap hvis Δt > GAP_FACTOR × median(Δt)
GAP_FACTOR = 1.8
GAP_FILL   = "rgba(255, 235, 59, 0.25)"
BRUDD_FILL = "rgba(220, 53, 69, 0.12)"   # lys gul markering av hull (legend-toggle)

OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
STAMP_SAFE = datetime.now().strftime("%Y%m%d_%H%M%S")

PLOTLY_CONFIG = {"scrollZoom": True, "displayModeBar": True}

# ---------- Hjelpere ----------
def to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = (
            s.astype(str)
            .str.replace("\u00A0", " ", regex=False)
            .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(s, errors="coerce")

def fmt_dt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")

# ---------- Bruddlogikk ----------
def build_segments(df_raw: pd.DataFrame, eff_min_mw: float = EFF_MIN_MW):
    """
    Finn bruddperioder:
      brudd = (release < Krav)  og (hvis effekt finnes) Effekt > eff_min_mw,
      der release = Min.fillna(0) + Overløp.fillna(0),
      og KUN når has_any_data = (~isna(Min)) | (~isna(Overløp)).
    -> Punkter der både Min og Overløp mangler (NaN) kan ikke inngå i brudd og bryter segmentet.
    Overløp (NaN) behandles som 0 i release. Min-NaN behandles som 0 i release for at overløp kan dekke kravet,
    MEN punktet må ha minst én dokumentert verdi (Min ELLER Overløp) for å kunne telle som brudd.
    Effektfilter brukes kun dersom effektkolonnen finnes og har data; ellers flagges 'ikke verifisert'.
    Returnerer: (segments: list[(start, slutt)], eff_filter_applied: bool)
    """
    required = [COL_DATO, COL_MVF, COL_KRAV]
    if not all(c in df_raw.columns for c in required):
        return [], False

    cols = [COL_DATO, COL_MVF, COL_KRAV]
    has_over = COL_OVER in df_raw.columns
    if has_over: cols.append(COL_OVER)
    has_eff = (COL_EFF in df_raw.columns) and (not df_raw[COL_EFF].isna().all())
    if has_eff: cols.append(COL_EFF)

    df = df_raw[cols].copy()
    df[COL_DATO] = pd.to_datetime(df[COL_DATO], errors="coerce", dayfirst=True)
    for c in [COL_MVF, COL_KRAV] + ([COL_OVER] if has_over else []) + ([COL_EFF] if has_eff else []):
        df[c] = to_num(df[c])
    df = df.dropna(subset=[COL_DATO, COL_KRAV]).sort_values(COL_DATO)

    # Dokumentasjonsmaske: minst én av Min/Overløp finnes
    has_any_release_data = pd.Series(True, index=df.index)
    if has_over:
        has_any_release_data = df[COL_MVF].notna() | df[COL_OVER].notna()
        over = df[COL_OVER].fillna(0.0)
    else:
        has_any_release_data = df[COL_MVF].notna()  # bare Min finnes
        over = 0.0

    release = df[COL_MVF].fillna(0.0) + over  # overløp kan "redde" når Min mangler

    # Grunnbetingelse: release < Krav, men KUN der vi har dokumenterte slipp/overløp
    base = (release < df[COL_KRAV]) & has_any_release_data

    if has_eff:
        cond = base & (df[COL_EFF].fillna(0.0) > eff_min_mw)
        eff_used = True
    else:
        cond = base
        eff_used = False

    cond = cond.fillna(False)

    # Bygg segmenter (kontinuerlige True-blokker)
    segments = []
    times = df[COL_DATO].values
    flags = cond.values
    in_seg, start_ts = False, None
    for i, flag in enumerate(flags):
        if flag and not in_seg:
            in_seg, start_ts = True, times[i]
        elif not flag and in_seg:
            segments.append((start_ts, times[i - 1]))
            in_seg, start_ts = False, None
    if in_seg:
        segments.append((start_ts, times[-1]))

    segments = [(min(a, b), max(a, b)) for (a, b) in segments]
    return segments, eff_used

# ---------- Hull: deteksjon (for "Datahull" basert på Min) ----------
def detect_gaps(times: pd.Series, gap_factor: float = GAP_FACTOR):
    """
    times: pd.Series av tidsstempel (sortert).
    Returnerer gaps: liste av (t_prev, t_curr) der diff > threshold
    """
    t = pd.to_datetime(times).dropna().sort_values()
    if len(t) < 2:
        return []
    dts = t.diff().dt.total_seconds().to_numpy()
    pos = dts[1:] if len(dts) > 1 else dts
    pos = pos[pos > 0]
    if pos.size == 0:
        return []
    thr = gap_factor * np.median(pos)
    gaps = []
    for i in range(1, len(t)):
        if (t.iloc[i] - t.iloc[i - 1]).total_seconds() > thr:
            gaps.append((t.iloc[i - 1], t.iloc[i]))
    return gaps

def series_break_mask(x_ts: pd.Series, y: pd.Series, gap_factor: float = GAP_FACTOR):
    """
    Returnerer break_before-mask for en gitt serie:
      True ved indeks i --> legg inn None før punkt i for å skape synlig brudd.
    Bruker både NaN i y og store steg i tid (for den serien alene).
    """
    t = pd.to_datetime(x_ts)
    yv = y
    n = len(t)
    if n == 0:
        return []
    break_before = [False] * n

    valid_idx = yv.notna().to_numpy()
    idx = np.flatnonzero(valid_idx)
    if idx.size >= 2:
        deltas = (t.iloc[idx].diff().dt.total_seconds()).to_numpy()
        pos = deltas[1:] if len(deltas) > 1 else deltas
        pos = pos[pos > 0]
        if pos.size > 0:
            thr = gap_factor * np.median(pos)
            for k in range(1, len(idx)):
                i = idx[k]
                if (t.iloc[i] - t.iloc[idx[k-1]]).total_seconds() > thr:
                    break_before[i] = True
    # brudd også når y er NaN (start etter NaN)
    for i in range(1, n):
        if pd.isna(yv.iloc[i-1]) and pd.notna(yv.iloc[i]):
            break_before[i] = True
    return break_before

def apply_breaks(x_list, y_list, break_before):
    """Sett inn None mellom punkter der break_before[i] er True (for synlig brudd i linjen)."""
    x_new, y_new = [], []
    for i, (xi, yi) in enumerate(zip(x_list, y_list)):
        if i > 0 and break_before[i]:
            x_new.append(None)
            y_new.append(None)
        x_new.append(xi)
        y_new.append(yi)
    return x_new, y_new

# ---------- Figur ----------
def build_figure(df_plot: pd.DataFrame, segs):
    # Behold ALLE rader med gyldig dato
    df_plot = df_plot.dropna(subset=[COL_DATO]).sort_values(COL_DATO).copy()

    has_min  = COL_MVF  in df_plot.columns
    has_krav = COL_KRAV in df_plot.columns
    has_over = COL_OVER in df_plot.columns
    has_eff  = COL_EFF  in df_plot.columns

    X = df_plot[COL_DATO].tolist()

    # Datahull (gul) basert på tidsstempler der MIN finnes (som avtalt)
    if has_min:
        gaps = detect_gaps(df_plot.loc[df_plot[COL_MVF].notna(), COL_DATO])
    else:
        gaps = []
    print(f"Datahull funnet: {len(gaps)}")

    # Primær-akse y-område (for gule polygoner)
    y_cols = [c for c in [COL_MVF, COL_KRAV, COL_OVER] if c in df_plot.columns]
    if y_cols:
        y_all = pd.concat([df_plot[c] for c in y_cols], axis=0)
        y_min = float(np.nanmin(y_all.values)) if np.isfinite(np.nanmin(y_all.values)) else 0.0
        y_max = float(np.nanmax(y_all.values)) if np.isfinite(np.nanmax(y_all.values)) else 1.0
        rng = max(y_max - y_min, 1e-6)
        y_lo = y_min - 0.06 * rng
        y_hi = y_max + 0.06 * rng
    else:
        y_lo, y_hi = 0.0, 1.0

    # Polygon for alle hull (separert med None)
    x_poly, y_poly = [], []
    for (x0, x1) in gaps:
        x_poly += [x0, x1, x1, x0, None]
        y_poly += [y_lo, y_lo, y_hi, y_hi, None]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # Bruddmasker per serie (egen kontinuitet)
    if has_min:
        br_min  = series_break_mask(df_plot[COL_DATO], df_plot[COL_MVF])
    if has_krav:
        br_krav = series_break_mask(df_plot[COL_DATO], df_plot[COL_KRAV])
    if has_over:
        br_over = series_break_mask(df_plot[COL_DATO], df_plot[COL_OVER])
    if has_eff:
        br_eff  = series_break_mask(df_plot[COL_DATO], df_plot[COL_EFF])

    # Min (primær)
    if has_min:
        y_min_list = df_plot[COL_MVF].tolist()
        x_m, y_m = apply_breaks(X, y_min_list, br_min)
        fig.add_trace(go.Scatter(
            x=x_m, y=y_m, mode="lines", name=COL_MVF,
            line=dict(color="#1f77b4", width=2),
            connectgaps=False,
        ), 1, 1, secondary_y=False)

    # Krav (primær)
    if has_krav:
        y_k_list = df_plot[COL_KRAV].tolist()
        x_k, y_k = apply_breaks(X, y_k_list, br_krav)
        fig.add_trace(go.Scatter(
            x=x_k, y=y_k, mode="lines", name=COL_KRAV,
            line=dict(color="#d62728", width=1.8, dash="dash"),
            connectgaps=False,
        ), 1, 1, secondary_y=False)

    # Overløp (primær) – AV default
    if has_over:
        y_o_list = df_plot[COL_OVER].tolist()
        x_o, y_o = apply_breaks(X, y_o_list, br_over)
        fig.add_trace(go.Scatter(
            x=x_o, y=y_o, mode="lines", name=COL_OVER,
            line=dict(color="#9467bd", width=1.6),
            visible="legendonly",
            connectgaps=False,
        ), 1, 1, secondary_y=False)

    # Effekt (sekundær) – AV default
    if has_eff:
        y_e_list = df_plot[COL_EFF].tolist()
        x_e, y_e = apply_breaks(X, y_e_list, br_eff)
        fig.add_trace(go.Scatter(
            x=x_e, y=y_e, mode="lines", name=COL_EFF,
            line=dict(color="#2ca02c", width=1.8),
            visible="legendonly",
            connectgaps=False,
        ), 1, 1, secondary_y=True)

    # (SIST I LEGEND) Datahull (gul) – av som standard
    if gaps:
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, mode="lines",
            line=dict(width=0), fill="toself", fillcolor=GAP_FILL,
            name="Datahull", showlegend=True, visible="legendonly", hoverinfo="skip",
        ), 1, 1, secondary_y=False)

    # Akser
    fig.update_xaxes(title_text="Dato", type="date", rangeslider=dict(visible=False))
    fig.update_yaxes(title_text="Vannføring (l/s)", secondary_y=False)
    fig.update_yaxes(title_text="Effekt produksjon (MW)", secondary_y=True)

    # Layout
    fig.update_layout(
        title=dict(text=f"Minstevannføring – interaktiv rapport — {STAMP}",
                   x=0.5, xanchor="center", y=0.985, yanchor="top", pad=dict(t=6, b=6)),
        template="plotly_white", hovermode="x unified", dragmode="pan",
        autosize=True, height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0,
                    bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(l=80, r=40, t=140, b=80),
    )

    # Bruddmarkering som polygon-trace (togglbar i legend)
    try:
        y_dom = list(fig.layout.yaxis.domain) if fig.layout.yaxis.domain else [0.0, 1.0]
    except Exception:
        y_dom = [0.0, 1.0]

    x_brudd_poly, y_brudd_poly = [], []
    for (x0, x1) in segs:
        if pd.isna(x0) or pd.isna(x1) or x1 <= x0:
            continue
        x_brudd_poly += [x0, x1, x1, x0, None]
        y_brudd_poly += [y_lo, y_lo, y_hi, y_hi, None]

    if x_brudd_poly:
        fig.add_trace(go.Scatter(
            x=x_brudd_poly, y=y_brudd_poly, mode="lines",
            line=dict(width=0), fill="toself", fillcolor="rgba(220, 53, 69, 0.12)",
            name="Bruddperioder", showlegend=True, visible=True, hoverinfo="skip",
        ), 1, 1, secondary_y=False)

    # Highlight (kraftig rød + kant) – startsynlig False
    base_shapes = []
    highlight_idx = 0
    dmin = df_plot[COL_DATO].min() if len(df_plot) else pd.Timestamp.now()
    base_shapes.append(dict(
        type="rect", xref="x", yref="paper",
        x0=dmin, x1=dmin, y0=y_dom[0], y1=y_dom[1],
        fillcolor="rgba(220, 53, 69, 0.55)",
        line=dict(width=2, color="rgba(200,0,0,0.9)"),
        layer="above", visible=False
    ))
    fig.update_layout(shapes=base_shapes)

    # Annotasjon (vises/skjules via dropdown)
    ann = [dict(
        xref="x", yref="paper", x=dmin, y=y_dom[1], yanchor="bottom",
        showarrow=False, bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(200,0,0,0.9)", borderwidth=1,
        font=dict(color="rgba(200,0,0,0.95)", size=12), text="", visible=False
    )]
    fig.update_layout(annotations=ann)

    # Dropdown for å fremheve valgt brudd
    buttons = [dict(
        label="Fjern markering",
        method="relayout",
        args=[{f"shapes[{highlight_idx}].visible": False, "annotations[0].visible": False}]
    )]
    if len(segs) > 0:
        for i, (x0, x1) in enumerate(segs, start=1):
            mid = pd.Timestamp(x0) + (pd.Timestamp(x1) - pd.Timestamp(x0)) / 2
            label = f"Brudd #{i}: {fmt_dt(x0)} – {fmt_dt(x1)}"
            buttons.append(dict(
                label=label, method="relayout",
                args=[{
                    f"shapes[{highlight_idx}].x0": pd.Timestamp(x0),
                    f"shapes[{highlight_idx}].x1": pd.Timestamp(x1),
                    f"shapes[{highlight_idx}].visible": True,
                    "annotations[0].x": mid,
                    "annotations[0].text": label,
                    "annotations[0].visible": True
                }]
            ))
    fig.update_layout(updatemenus=[dict(
        type="dropdown", direction="down", x=1.0, xanchor="right",
        y=1.13, yanchor="top", buttons=buttons, pad=dict(r=10, t=2), showactive=False
    )])

    return fig

# ---------- Statistikk (varighet = slutt - start) + Begrunnelse ----------
def build_stats_html(df_raw: pd.DataFrame, segs):
    # Datafangst (på mvf)
    tot_obs = int(df_raw[COL_MVF].shape[0]) if COL_MVF in df_raw.columns else 0
    missing_min = int(df_raw[COL_MVF].isna().sum()) if COL_MVF in df_raw.columns else 0
    missing_pct = (100.0 * missing_min / tot_obs) if tot_obs > 0 else 0.0
    completeness_pct = 100.0 - missing_pct
    is_ok = (completeness_pct >= 97.0)

    # For underskudd i segmenter: release = Min.fillna(0) + Overløp.fillna(0)
    cols_need = [COL_DATO, COL_MVF, COL_KRAV]
    has_over = COL_OVER in df_raw.columns
    if has_over: cols_need.append(COL_OVER)
    has_eff = COL_EFF in df_raw.columns and not df_raw[COL_EFF].isna().all()

    use_cols = [c for c in cols_need + ([COL_EFF] if COL_EFF in df_raw.columns else []) if c in df_raw.columns]
    df = df_raw[use_cols].copy()
    df[COL_DATO] = pd.to_datetime(df[COL_DATO], errors="coerce", dayfirst=True)
    for c in [COL_MVF, COL_KRAV, COL_OVER, COL_EFF]:
        if c in df.columns: df[c] = to_num(df[c])
    df = df.dropna(subset=[COL_DATO, COL_KRAV]).sort_values(COL_DATO)

    items = []
    total_h = 0.0

    for idx, (x0, x1) in enumerate(segs, start=1):
        seg = df[(df[COL_DATO] >= x0) & (df[COL_DATO] <= x1)]
        if seg.empty:
            continue

        over = seg[COL_OVER].fillna(0.0) if COL_OVER in seg.columns else 0.0
        release = seg[COL_MVF].fillna(0.0) + over
        deficit = (seg[COL_KRAV] - release).clip(lower=0)

        duration_h = (seg[COL_DATO].iloc[-1] - seg[COL_DATO].iloc[0]).total_seconds() / 3600.0
        total_h += duration_h

        max_def = float(deficit.max()) if not deficit.empty else 0.0
        mean_def = float(deficit.mean()) if not deficit.empty else 0.0

        # Begrunnelse (kriterier)
        if has_eff:
            eff_ok_all = (seg[COL_EFF].fillna(0.0) > EFF_MIN_MW).all()
            if eff_ok_all:
                reason = "Underskudd verifisert (Min+Overløp < Krav, Effekt > terskel)"
            else:
                reason = "Underskudd (varierende produksjon i segmentet)"
        else:
            reason = "Underskudd (ikke verifisert – effekt mangler)"

        # Tilleggsmarkør hvis data er delvis udokumentert i segmentet:
        # (Min og Overløp BEGGE mangler i minst ett punkt innen segmentet)
        has_both_missing = True
        if COL_OVER in seg.columns:
            both_missing_mask = seg[COL_MVF].isna() & seg[COL_OVER].isna()
            has_both_missing = both_missing_mask.any()
        else:
            # Når Overløp ikke finnes i dataset, regner vi ikke dette som "begge mangler".
            has_both_missing = False

        if has_both_missing:
            reason += " — Delvis udokumentert (Min/Overløp mangler i deler av perioden)"

        # Også nyttig markør hvis Min mangler i deler, men Overløp finnes:
        if seg[COL_MVF].isna().any() and (COL_OVER in seg.columns):
            reason += " — Overløp supplerer manglende Min i deler av perioden"

        items.append((
            idx,
            fmt_dt(seg[COL_DATO].iloc[0]),
            fmt_dt(seg[COL_DATO].iloc[-1]),
            f"{duration_h:.2f}",
            f"{max_def:.1f}",
            f"{mean_def:.1f}",
            reason
        ))

    # Sammendrag
    def build_summary_html():
        rows = []
        rows.append(("Datafangst (fullstendighet)", f"{completeness_pct:.1f} %", "ok" if is_ok else "bad"))
        rows.append(("Manglende Minstevannf. (%)", f"{missing_pct:.1f} %", ""))
        rows.append(("Antall brudd", f"{len(segs)}", ""))
        rows.append(("Total bruddvarighet (h)", f"{total_h:.2f}", ""))

        eff_available = COL_EFF in df_raw.columns and not df_raw[COL_EFF].isna().all()
        if eff_available:
            rows.append(("Effektdata tilgjengelig", "Ja", ""))
            rows.append(("Effektfilter brukt i bruddsøk", f"Ja (terskel > {EFF_MIN_MW:.3f} MW)", ""))
        else:
            rows.append(("Effektdata tilgjengelig", "Nei", ""))
            rows.append(("Effektfilter brukt i bruddsøk", "Nei (effekt mangler)", "bad"))

        tr_html = []
        for label, val, flag in rows:
            cls = "okrow" if flag == "ok" else ("badrow" if flag == "bad" else "")
            tr_html.append(f"<tr class='{cls}'><td>{label}</td><td>{val}</td></tr>")
        return "<table class='sum'><thead><tr><th>Metrikk</th><th>Verdi</th></tr></thead><tbody>" + "".join(tr_html) + "</tbody></table>"

    # Brudd-detaljer
    def build_brudd_html():
        if not items:
            warn = ""
            if COL_EFF not in df_raw.columns or df_raw[COL_EFF].isna().all():
                warn = (
                    "<div class='note' style='color:#b30000'>"
                    "Merk: Effektdata mangler – brudd er ikke verifisert mot produksjon."
                    "</div>"
                )
            return warn + "<p><em>Ingen brudd funnet.</em></p>"

        th = "<tr><th>#</th><th>Start</th><th>Slutt</th><th>Varighet (h)</th><th>Maks underskudd (l/s)</th><th>Gj.snitt underskudd (l/s)</th><th>Begrunnelse (kriterier)</th></tr>"
        trs = "".join(
            [
                f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td></tr>"
                for r in items
            ]
        )
        return "<table class='brudd'><thead>" + th + "</thead><tbody>" + trs + "</tbody></table>"
    return build_summary_html(), build_brudd_html()

# ---------- HTML-skriving ----------
def write_html(fig, summary_html, brudd_html):
    PAGE_CSS = """
body{font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px; color:#333;}
h1,h2,h3{margin:0 0 8px 0}
details{margin-top:16px;}
details > summary{font-weight:600; cursor:pointer; font-size:15px; list-style:none; outline:none;}
/* Knappestil for statistikktoggle */
details > summary.btn {
  display:inline-block; background:#005a9e; color:#fff; border-radius:4px;
  padding:6px 12px; border:none; user-select:none; font-size:14px;
}
details[open] > summary.btn { background:#004c82; }
details > summary.btn::-webkit-details-marker { display:none; }

/* Sub-details for logic explanation */
details.logic { margin-top:8px; margin-bottom:16px; border:1px solid #eee; border-radius:4px; padding:8px; background:#fafafa; }
details.logic > summary { color:#666; font-weight:normal; font-size:13px; }
details.logic[open] > summary { margin-bottom:8px; border-bottom:1px solid #ddd; padding-bottom:4px; }

table{border-collapse:collapse; margin-top:12px; width:100%; max-width:1200px; font-size:14px;}
table.sum th, table.sum td, table.brudd th, table.brudd td{border-bottom:1px solid #eee; padding:8px 12px; text-align:left;}
table.sum thead th, table.brudd thead th{background:#f8f9fa; color:#444; font-weight:600; border-bottom:2px solid #ddd;}
.okrow{background:rgba(40,167,69,0.08);}
.badrow{background:rgba(220,53,69,0.08);}
.note{color:#666; font-size:13px; margin-bottom:4px;}
.js-plotly-plot, .plotly-graph-div {max-width:100% !important;}
"""
    meta_no_cache = """
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<meta http-equiv="Pragma" content="no-cache"/>
<meta http-equiv="Expires" content="0"/>
"""
    fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config=PLOTLY_CONFIG)

    # Forklaring flyttet inn i en sub-details
    EXPLAIN = """
<details class="logic">
  <summary>Klikk for å lese om definisjoner og datagrunnlag</summary>
  <div style="font-size:13px; line-height:1.5; color:#444;">
  <b>Varighet og datagrunnlag for brudd:</b><br>
  Brudd identifiseres når <u>(mvf + Overløp) &lt; Krav</u>, og – dersom produksjonsdata finnes – <u>Effekt &gt; terskel</u>.<br>
  Varigheten beregnes som <u>slutt‑tidspunkt minus start‑tidspunkt</u> for perioder der disse vilkårene er oppfylt.<br>
  Et tidssteg kan kun bidra til brudd dersom minst én av <em>mvf</em> eller <em>Overløp</em> er dokumentert.<br>
  Perioder der <em>både</em> Min og Overløp mangler, regnes som udokumentert og vises ikke som brudd (marker gjerne «Datahull» i legend).<br>
  Overløp og effekt vises i grafen også når minstevannføring mangler.<br>
  Dersom effektdata mangler, verifiseres brudd ikke mot produksjon og bør vurderes med forsiktighet.
  </div>
</details>
"""

    html = f"""<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="utf-8">
{meta_no_cache}
<style>{PAGE_CSS}</style>
</head>
<body>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; font-size:13px; color:#555;">
  <div>Generert: {STAMP}</div>
  <div><em>Tips: Velg «Brudd #» i menyen over plottet for detaljer.</em></div>
</div>

{fig_html}

<details>
  <summary class="btn">Vis statistikk og detaljer</summary>
  <div style="margin-top:12px;">
    {summary_html}
    {EXPLAIN}
    <h3 style="margin-top:24px; font-size:16px; color:#333;">Bruddlogg</h3>
    {brudd_html}
  </div>
</details>
</body>
</html>"""

    out_ts = OUT_DIR / f"rapport_interaktiv_{STAMP_SAFE}.html"
    out_latest = OUT_DIR / "rapport_interaktiv.html"
    out_ts.write_text(html, encoding="utf-8")
    out_latest.write_text(html, encoding="utf-8")
    return out_ts, out_latest

# ---------- Hovedløp ----------
def main():
    print("=== KJØRER RAPPORT ===")
    print(f"Script-katalog : {Path(__file__).parent.resolve()}")
    print(f"Excel-fil      : {EXCEL.resolve()} (finnes: {EXCEL.exists()})")
    if not EXCEL.exists():
        raise FileNotFoundError(f"Excel-filen ble ikke funnet: {EXCEL.resolve()}")

    # Les og standardiser
    df_raw = pd.read_excel(EXCEL, sheet_name=SHEET, engine="openpyxl")
    print(f"Kolonner : {list(df_raw.columns)} - rader: {len(df_raw)}")
    if COL_DATO in df_raw.columns: df_raw[COL_DATO] = pd.to_datetime(df_raw[COL_DATO], errors="coerce", dayfirst=True)
    for c in [COL_MVF, COL_KRAV, COL_EFF, COL_OVER]:
        if c in df_raw.columns: df_raw[c] = to_num(df_raw[c])

    # Datasett for plott: BEHOLD alle rader med gyldig dato
    keep = [c for c in [COL_DATO, COL_MVF, COL_KRAV, COL_OVER, COL_EFF] if c in df_raw.columns]
    df_plot = df_raw[keep].dropna(subset=[COL_DATO]).sort_values(COL_DATO)

    # Brudd (felles logikk + effekt-terskel + dokumentasjonskrav)
    segs, eff_used = build_segments(df_raw, eff_min_mw=EFF_MIN_MW)
    print(f"Antall brudd : {len(segs)} (Effektfilter brukt: {eff_used}, terskel: > {EFF_MIN_MW} MW)")

    # Figur
    fig = build_figure(df_plot, segs)

    # Statistikk (varighet = slutt - start) + begrunnelse
    summary_html, brudd_html = build_stats_html(df_raw, segs)

    # HTML-ut
    out_ts, out_latest = write_html(fig, summary_html, brudd_html)
    print(f"Skrev HTML (tidsstemplet): {out_ts.resolve()}")
    print(f"Skrev HTML (latest)     : {out_latest.resolve()}")

    # Åpne i nettleser
    webbrowser.open_new_tab(out_ts.resolve().as_uri())
    print("=== FERDIG ===")

if __name__ == "__main__":
    main()