import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# === KONFIGURASJON ===
st.set_page_config(page_title="Minstevannføring Rapport", layout="wide")

# Standard kolonnenavn
COL_DATO = "Dato"
COL_MVF  = "Minstevannføring (l/s)"
COL_KRAV = "Krav (l/s)"
COL_EFF  = "Effekt produksjon (MW)"
COL_OVER = "Overløp vannføring (l/s)"

# Standard parametere
DEFAULT_EFF_MIN_MW = 0.05
DEFAULT_GAP_FACTOR = 1.8

# Farger
GAP_FILL   = "rgba(255, 235, 59, 0.25)"
BRUDD_FILL = "rgba(220, 53, 69, 0.12)"

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
def build_segments(df_raw: pd.DataFrame, eff_min_mw: float):
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

    has_any_release_data = pd.Series(True, index=df.index)
    if has_over:
        has_any_release_data = df[COL_MVF].notna() | df[COL_OVER].notna()
        over = df[COL_OVER].fillna(0.0)
    else:
        has_any_release_data = df[COL_MVF].notna()
        over = 0.0

    release = df[COL_MVF].fillna(0.0) + over
    base = (release < df[COL_KRAV]) & has_any_release_data

    if has_eff:
        cond = base & (df[COL_EFF].fillna(0.0) > eff_min_mw)
        eff_used = True
    else:
        cond = base
        eff_used = False

    cond = cond.fillna(False)

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

# ---------- Hull: deteksjon ----------
def detect_gaps(times: pd.Series, gap_factor: float):
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

def series_break_mask(x_ts: pd.Series, y: pd.Series, gap_factor: float):
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
    for i in range(1, n):
        if pd.isna(yv.iloc[i-1]) and pd.notna(yv.iloc[i]):
            break_before[i] = True
    return break_before

def apply_breaks(x_list, y_list, break_before):
    x_new, y_new = [], []
    for i, (xi, yi) in enumerate(zip(x_list, y_list)):
        if i > 0 and break_before[i]:
            x_new.append(None)
            y_new.append(None)
        x_new.append(xi)
        y_new.append(yi)
    return x_new, y_new

# ---------- Figur ----------
def build_figure(df_plot: pd.DataFrame, segs, gap_factor, selected_brudd_indices=None):
    df_plot = df_plot.dropna(subset=[COL_DATO]).sort_values(COL_DATO).copy()

    has_min  = COL_MVF  in df_plot.columns
    has_krav = COL_KRAV in df_plot.columns
    has_over = COL_OVER in df_plot.columns
    has_eff  = COL_EFF  in df_plot.columns

    X = df_plot[COL_DATO].tolist()

    if has_min:
        gaps = detect_gaps(df_plot.loc[df_plot[COL_MVF].notna(), COL_DATO], gap_factor)
    else:
        gaps = []

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

    x_poly, y_poly = [], []
    for (x0, x1) in gaps:
        x_poly += [x0, x1, x1, x0, None]
        y_poly += [y_lo, y_lo, y_hi, y_hi, None]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    if has_min:
        br_min  = series_break_mask(df_plot[COL_DATO], df_plot[COL_MVF], gap_factor)
        y_min_list = df_plot[COL_MVF].tolist()
        x_m, y_m = apply_breaks(X, y_min_list, br_min)
        fig.add_trace(go.Scatter(
            x=x_m, y=y_m, mode="lines", name=COL_MVF,
            line=dict(color="#1f77b4", width=2),
            connectgaps=False,
        ), 1, 1, secondary_y=False)

    if has_krav:
        br_krav = series_break_mask(df_plot[COL_DATO], df_plot[COL_KRAV], gap_factor)
        y_k_list = df_plot[COL_KRAV].tolist()
        x_k, y_k = apply_breaks(X, y_k_list, br_krav)
        fig.add_trace(go.Scatter(
            x=x_k, y=y_k, mode="lines", name=COL_KRAV,
            line=dict(color="#d62728", width=1.8, dash="dash"),
            connectgaps=False,
        ), 1, 1, secondary_y=False)

    if has_over:
        br_over = series_break_mask(df_plot[COL_DATO], df_plot[COL_OVER], gap_factor)
        y_o_list = df_plot[COL_OVER].tolist()
        x_o, y_o = apply_breaks(X, y_o_list, br_over)
        fig.add_trace(go.Scatter(
            x=x_o, y=y_o, mode="lines", name=COL_OVER,
            line=dict(color="#9467bd", width=1.6),
            visible="legendonly",
            connectgaps=False,
        ), 1, 1, secondary_y=False)

    if has_eff:
        br_eff  = series_break_mask(df_plot[COL_DATO], df_plot[COL_EFF], gap_factor)
        y_e_list = df_plot[COL_EFF].tolist()
        x_e, y_e = apply_breaks(X, y_e_list, br_eff)
        fig.add_trace(go.Scatter(
            x=x_e, y=y_e, mode="lines", name=COL_EFF,
            line=dict(color="#2ca02c", width=1.8),
            visible="legendonly",
            connectgaps=False,
        ), 1, 1, secondary_y=True)

    if gaps:
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly, mode="lines",
            line=dict(width=0), fill="toself", fillcolor=GAP_FILL,
            name="Datahull", showlegend=True, visible="legendonly", hoverinfo="skip",
        ), 1, 1, secondary_y=False)

    fig.update_xaxes(title_text="Dato", type="date")
    fig.update_yaxes(title_text="Vannføring (l/s)", secondary_y=False)
    fig.update_yaxes(title_text="Effekt produksjon (MW)", secondary_y=True)

    fig.update_layout(
        template="plotly_white", hovermode="x unified", dragmode="pan",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0,
                    bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(l=50, r=20, t=80, b=50),
    )

    x_brudd_poly, y_brudd_poly = [], []
    for (x0, x1) in segs:
        if pd.isna(x0) or pd.isna(x1) or x1 <= x0:
            continue
        x_brudd_poly += [x0, x1, x1, x0, None]
        y_brudd_poly += [y_lo, y_lo, y_hi, y_hi, None]

    if x_brudd_poly:
        fig.add_trace(go.Scatter(
            x=x_brudd_poly, y=y_brudd_poly, mode="lines",
            line=dict(width=0), fill="toself", fillcolor=BRUDD_FILL,
            name="Bruddperioder", showlegend=True, visible=True, hoverinfo="skip",
        ), 1, 1, secondary_y=False)

    # Highlight valgte brudd (flere)
    if selected_brudd_indices:
        for idx in selected_brudd_indices:
            if 0 <= idx < len(segs):
                x0, x1 = segs[idx]
                # Bruk add_shape med yref="paper" for å dekke hele Y-aksen
                fig.add_shape(
                    type="rect",
                    xref="x", yref="paper",
                    x0=x0, x1=x1,
                    y0=0, y1=1,
                    fillcolor="rgba(255, 0, 0, 0.6)", 
                    line=dict(width=3, color="rgba(200,0,0,1.0)"),
                    layer="above"
                )
                mid = pd.Timestamp(x0) + (pd.Timestamp(x1) - pd.Timestamp(x0)) / 2
                fig.add_annotation(
                    x=mid, y=1, yref="paper",
                    text=f"Brudd #{idx + 1}",
                    showarrow=False, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(200,0,0,0.9)", borderwidth=1,
                    font=dict(color="rgba(200,0,0,0.95)", size=12)
                )

    return fig

# ---------- Statistikk ----------
def get_stats_data(df_raw: pd.DataFrame, segs, eff_min_mw):
    tot_obs = int(df_raw[COL_MVF].shape[0]) if COL_MVF in df_raw.columns else 0
    missing_min = int(df_raw[COL_MVF].isna().sum()) if COL_MVF in df_raw.columns else 0
    missing_pct = (100.0 * missing_min / tot_obs) if tot_obs > 0 else 0.0
    completeness_pct = 100.0 - missing_pct
    is_ok = (completeness_pct >= 97.0)

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

        if has_eff:
            eff_ok_all = (seg[COL_EFF].fillna(0.0) > eff_min_mw).all()
            if eff_ok_all:
                reason = "Underskudd verifisert (Min+Overløp < Krav, Effekt > terskel)"
            else:
                reason = "Underskudd (varierende produksjon i segmentet)"
        else:
            reason = "Underskudd (ikke verifisert – effekt mangler)"

        has_both_missing = True
        if COL_OVER in seg.columns:
            both_missing_mask = seg[COL_MVF].isna() & seg[COL_OVER].isna()
            has_both_missing = both_missing_mask.any()
        else:
            has_both_missing = False

        if has_both_missing:
            reason += " — Delvis udokumentert (Min/Overløp mangler)"
        
        if seg[COL_MVF].isna().any() and (COL_OVER in seg.columns):
            reason += " — Overløp supplerer manglende Min"

        items.append({
            "#": idx,
            "Start": fmt_dt(seg[COL_DATO].iloc[0]),
            "Slutt": fmt_dt(seg[COL_DATO].iloc[-1]),
            "Varighet (h)": round(duration_h, 2),
            "Maks underskudd (l/s)": round(max_def, 1),
            "Snitt underskudd (l/s)": round(mean_def, 1),
            "Begrunnelse": reason
        })
    
    summary = {
        "Datafangst (%)": round(completeness_pct, 1),
        "Manglende Min (%)": round(missing_pct, 1),
        "Antall brudd": len(segs),
        "Total bruddvarighet (h)": round(total_h, 2),
        "Effektdata tilgjengelig": "Ja" if has_eff else "Nei"
    }

    return summary, items

# ---------- Hovedapp ----------
def main():
    st.title("Minstevannføring Rapport")
    st.markdown("""
    Last opp Excel-fil med vannføringsdata. Appen analyserer dataene for brudd på minstevannføringskravet.
    """)

    with st.sidebar:
        st.header("Innstillinger")
        uploaded_file = st.file_uploader("Last opp Excel-fil", type=["xlsx"])
        
        eff_min_mw = st.number_input("Effekt-terskel (MW)", value=DEFAULT_EFF_MIN_MW, step=0.01, format="%.2f")
        gap_factor = st.number_input("Gap-faktor (for hull-deteksjon)", value=DEFAULT_GAP_FACTOR, step=0.1)

    if uploaded_file:
        try:
            # Bruk pd.ExcelFile for å lese arknavn først
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet_names = xls.sheet_names
            
            # Prøv å velge "data" som default hvis det finnes
            default_index = 0
            if "data" in sheet_names:
                default_index = sheet_names.index("data")
            
            selected_sheet = st.sidebar.selectbox("Velg ark", sheet_names, index=default_index)
            
            df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet, engine="openpyxl")
            
            # Valider kolonner
            if COL_DATO not in df_raw.columns:
                st.error(f"Mangler kolonne: '{COL_DATO}' i arket '{selected_sheet}'.")
                st.write("Fant følgende kolonner:", list(df_raw.columns))
                return

            segs, eff_used = build_segments(df_raw, eff_min_mw)
            
            # Statistikk
            summary, brudd_items = get_stats_data(df_raw, segs, eff_min_mw)
            
            # Vis sammendrag
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Antall brudd", summary["Antall brudd"])
            c2.metric("Total varighet (t)", summary["Total bruddvarighet (h)"])
            
            # Fargelegg Datafangst
            datafangst_val = summary['Datafangst (%)']
            datafangst_color = "green" if datafangst_val >= 97 else "red"
            c3.markdown(f"**Datafangst**<br><span style='color:{datafangst_color}; font-size: 24px; font-weight: bold'>{datafangst_val}%</span>", unsafe_allow_html=True)
            
            c4.metric("Effektdata", summary["Effektdata tilgjengelig"])

            # Container for plottet (øverst)
            plot_container = st.container()

            # Bruddvelger via tabell (nederst)
            selected_brudd_indices = []
            
            if brudd_items:
                st.subheader("Bruddlogg")
                
                st.write("Velg én eller flere rader i tabellen for å markere bruddene i grafen.")
                
                df_brudd = pd.DataFrame(brudd_items)
                # Konfigurer tabell for seleksjon
                event = st.dataframe(
                    df_brudd,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    hide_index=True
                )
                
                if event.selection.rows:
                    selected_brudd_indices = event.selection.rows
            else:
                st.info("Ingen brudd funnet.")

            # Generer figur og legg i containeren øverst
            keep = [c for c in [COL_DATO, COL_MVF, COL_KRAV, COL_OVER, COL_EFF] if c in df_raw.columns]
            df_plot = df_raw[keep].dropna(subset=[COL_DATO]).sort_values(COL_DATO)
            
            fig = build_figure(df_plot, segs, gap_factor, selected_brudd_indices)
            plot_container.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

            with st.expander("Se definisjoner og logikk"):
                st.markdown("""
                **Varighet og datagrunnlag for brudd:**
                *   Brudd identifiseres når `(mvf + Overløp) < Krav`, og – dersom produksjonsdata finnes – `Effekt > terskel`.
                *   Varigheten beregnes som `slutt‑tidspunkt minus start‑tidspunkt` for perioder der disse vilkårene er oppfylt.
                *   Et tidssteg kan kun bidra til brudd dersom minst én av *mvf* eller *Overløp* er dokumentert.
                *   Perioder der *både* Min og Overløp mangler, regnes som udokumentert og vises ikke som brudd.
                """)

        except Exception as e:
            st.error(f"Feil ved lesing av fil: {e}")
    else:
        st.info("Vennligst last opp en fil for å starte.")

if __name__ == "__main__":
    main()
