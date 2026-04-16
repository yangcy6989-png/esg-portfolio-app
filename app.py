"""
ESG Portfolio Optimiser — Streamlit Decision Tool
==================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from nsga2       import (load_data, nsga2, pareto_to_dataframe, RF_RATE, W_MAX)
from mv_esg      import (load_mv_data, run_mv_esg_frontier, frontier_to_dataframe)
from minimax_esg import (load_minimax_data,
                          minimax_esg_max_sharpe, portfolio_metrics,
                          ESG_MAX as MM_ESG_MAX)

st.set_page_config(
    page_title="ESG Portfolio Optimiser",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0e1a; color: #e8eaf0; }
  .main { background-color: #0a0e1a; }
  h1, h2, h3 { font-family: 'DM Serif Display', serif; color: #ffffff; }
  .hero { background: linear-gradient(135deg, #0d1f3c 0%, #0a2e1f 50%, #0d1f3c 100%); border: 1px solid #1e3a5f; border-radius: 16px; padding: 2.5rem; margin-bottom: 2rem; text-align: center; }
  .hero h1 { font-size: 2.8rem; background: linear-gradient(90deg, #4ade80, #22d3ee, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
  .hero p { color: #94a3b8; font-size: 1.1rem; font-weight: 300; }
  .metric-card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
  .metric-card .value { font-size: 1.9rem; font-weight: 600; color: #4ade80; font-family: 'DM Serif Display', serif; }
  .metric-card .label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
  .stSlider > div > div > div > div { background: #4ade80 !important; }
  section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e293b; }
  .stButton > button { background: linear-gradient(135deg, #166534, #065f46); color: white; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 1rem; width: 100%; }
  div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.85rem; }
  .section-title { font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #ffffff; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b; }
  .model-badge-mv      { background:#1a3a5c; border:1px solid #2563eb; border-radius:6px; padding:2px 10px; color:#60a5fa; font-size:0.8rem; font-weight:600; }
  .model-badge-mm      { background:#2d1a3a; border:1px solid #7c3aed; border-radius:6px; padding:2px 10px; color:#a78bfa; font-size:0.8rem; font-weight:600; }
  .model-badge-ns      { background:#1a3a2d; border:1px solid #16a34a; border-radius:6px; padding:2px 10px; color:#4ade80; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Colours
PLOT_BG  = "#0d1117"
PAPER_BG = "#0a0e1a"
GRID_C   = "#1e293b"
FONT_C   = "#94a3b8"
C_MV     = "#60a5fa"   # blue   — MV-ESG
C_MM     = "#a78bfa"   # purple — Minimax
C_NS     = "#4ade80"   # green  — NSGA-II

def base_layout(title, margin=None, height=None):
    d = dict(
        title=dict(text=title, font=dict(color="white", size=14)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_C, family="DM Sans"),
        xaxis=dict(gridcolor=GRID_C, zerolinecolor=GRID_C),
        yaxis=dict(gridcolor=GRID_C, zerolinecolor=GRID_C),
        margin=margin or dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
    )
    if height:
        d["height"] = height
    return d

def metric_card(val, lbl, color="#4ade80"):
    return (f'<div class="metric-card">'
            f'<div class="value" style="color:{color};font-size:1.4rem">{val}</div>'
            f'<div class="label">{lbl}</div></div>')

# ══════════════════════════════════════════════
#  CACHED DATA + RUNS
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_data():
    mu_ns, cov_ns, esg_ns, tickers_ns, names_ns, sectors_ns = load_data(
        "sp500_price_data.csv", "ESG_top150.xlsx"
    )
    mu_mv, Sigma_mv, theta_mv, raw_esg_mv, tickers_mv, esg_df = load_mv_data()
    mu_mm, Sigma_mm, erp, srp, grp, _, beta, raw_esg_mm, tickers_mm, _ = load_minimax_data()
    return (mu_ns, cov_ns, esg_ns, tickers_ns,
            mu_mv, Sigma_mv, theta_mv, raw_esg_mv, tickers_mv,
            mu_mm, Sigma_mm, erp, srp, grp, beta, raw_esg_mm, tickers_mm,
            esg_df)

@st.cache_data(show_spinner=False)
def _run_nsga2(esg_max, pop_size, n_gen, seed=42):
    mu, cov, esg, tickers, _, _ = load_data("sp500_price_data.csv", "ESG_top150.xlsx")
    pf, history = nsga2(mu, cov, esg, pop_size=pop_size, n_gen=n_gen,
                        esg_max=esg_max, seed=seed, verbose=False)
    df = pareto_to_dataframe(pf, tickers, mu, cov, esg)
    return df, history

@st.cache_data(show_spinner=False)
def _run_mv_esg(esg_max, n_samples, seed=42):
    mu, Sigma, theta, raw_esg, tickers, _ = load_mv_data()
    frontier = run_mv_esg_frontier(mu, Sigma, theta, raw_esg,
                                    n_samples=n_samples, esg_max=esg_max, seed=seed)
    return frontier_to_dataframe(frontier, tickers, raw_esg)

@st.cache_data(show_spinner=False)
def _run_minimax(esg_max):
    mu, Sigma, erp, srp, grp, _, beta, raw_esg, tickers, _ = load_minimax_data()
    _w_s1, _t_s1, w_s2, _t_s2, sh_s2 = minimax_esg_max_sharpe(
        mu, Sigma, erp, srp, grp, raw_esg, beta,
        w_erp=10, w_srp=10, w_grp=10,
        esg_max=esg_max, seed=42
    )
    ret, vol, _ = portfolio_metrics(w_s2, mu, Sigma)
    row = {
        "Portfolio_ID":      1,
        "Combo_Name":        "Equal (33/33/33)",
        "Annual_Return":     round(ret, 6),
        "Annual_Volatility": round(vol, 6),
        "Sharpe_Ratio":      round(sh_s2, 4),
        "ESG_Score":         round(float(raw_esg @ w_s2), 4),
        "Minimax_T":         round(float(_t_s2), 6),
    }
    for t, wt in zip(tickers, w_s2):
        row[f"w_{t}"] = round(float(wt), 6)
    return pd.DataFrame([row]), tickers

# ══════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🌿 ESG Portfolio Optimiser</h1>
  <p>MV-ESG &nbsp;·&nbsp; Minimax ESG &nbsp;·&nbsp; NSGA-II &nbsp;·&nbsp;
     150 S&P 500 constituents &nbsp;·&nbsp; 2023–2024</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 Constraints")
    st.markdown("---")
    esg_floor = st.slider("Max ESG Risk Score", 0, 35, 20, 5,
        help="Hard ceiling on portfolio totalEsg. Lower = stricter ESG screen.")
    st.markdown("---")
    st.markdown("### 🔍 Filter Results")
    max_vol    = st.slider("Max Annual Volatility (%)", 10, 60, 40, 5)
    min_return = st.slider("Min Annual Return (%)",      0, 100, 10, 5)
    min_sharpe = st.slider("Min Sharpe Ratio",         0.0, 4.0, 0.5, 0.25)
    st.markdown("---")
    st.markdown("### ⚙️ Algorithm Settings")
    n_samples = st.selectbox("MV-ESG Samples",    [100, 200, 350], index=2)
    pop_size  = st.selectbox("NSGA-II Population", [100, 200, 300], index=1)
    n_gen     = st.selectbox("NSGA-II Generations",[100, 200, 300], index=2)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.8'>
    <b style='color:#64748b'>Models</b><br>
    <span style='color:#60a5fa'>■</span> <b>MV-ESG</b> — scalarised mean-variance<br>
    <span style='color:#a78bfa'>■</span> <b>Minimax ESG</b> — lexicographic 2-stage<br>
    <span style='color:#4ade80'>■</span> <b>NSGA-II</b> — multi-objective Pareto
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  RUN ALL MODELS
# ══════════════════════════════════════════════
with st.spinner("⚙️ Running all three models… (cached after first run)"):
    esg_df = pd.read_excel("ESG_top150.xlsx")
    (mu_ns, cov_ns, esg_ns, tickers_ns,
     mu_mv, Sigma_mv, theta_mv, raw_esg_mv, tickers_mv,
     mu_mm, Sigma_mm, erp, srp, grp, beta, raw_esg_mm, tickers_mm,
     _) = _load_data()

    mv_df          = _run_mv_esg(esg_floor, n_samples)
    mm_df, mm_tick = _run_minimax(esg_floor)
    ns_df, ns_hist = _run_nsga2(esg_floor, pop_size, n_gen)

def apply_filters(df):
    return df[
        (df["Annual_Volatility"] * 100 <= max_vol) &
        (df["Annual_Return"]     * 100 >= min_return) &
        (df["Sharpe_Ratio"]             >= min_sharpe)
    ].copy()

mv_f = apply_filters(mv_df)
mm_f = apply_filters(mm_df)
ns_f = apply_filters(ns_df)

bm_mv = mv_f.loc[mv_f["Sharpe_Ratio"].idxmax()] if len(mv_f) else mv_df.loc[mv_df["Sharpe_Ratio"].idxmax()]
bm_mm = mm_f.loc[mm_f["Sharpe_Ratio"].idxmax()] if len(mm_f) else mm_df.loc[mm_df["Sharpe_Ratio"].idxmax()]
bm_ns = ns_f.loc[ns_f["Sharpe_Ratio"].idxmax()] if len(ns_f) else ns_df.loc[ns_df["Sharpe_Ratio"].idxmax()]

# ══════════════════════════════════════════════
#  COMPARISON OVERVIEW
# ══════════════════════════════════════════════
st.markdown('<div class="section-title">⚖️ Model Comparison</div>', unsafe_allow_html=True)

# --- Top metrics row
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<span class="model-badge-mv">MV-ESG</span>', unsafe_allow_html=True)
    a, b, c_ = st.columns(3)
    a.markdown(metric_card(f"{bm_mv['Annual_Return']*100:.1f}%", "Best Return", C_MV), unsafe_allow_html=True)
    b.markdown(metric_card(f"{bm_mv['Sharpe_Ratio']:.2f}", "Best Sharpe", C_MV), unsafe_allow_html=True)
    c_.markdown(metric_card(f"{bm_mv['ESG_Score']:.1f}", "ESG Risk ↓", C_MV), unsafe_allow_html=True)
with c2:
    st.markdown('<span class="model-badge-mm">Minimax ESG</span>', unsafe_allow_html=True)
    a, b, c_ = st.columns(3)
    a.markdown(metric_card(f"{bm_mm['Annual_Return']*100:.1f}%", "Best Return", C_MM), unsafe_allow_html=True)
    b.markdown(metric_card(f"{bm_mm['Sharpe_Ratio']:.2f}", "Best Sharpe", C_MM), unsafe_allow_html=True)
    c_.markdown(metric_card(f"{bm_mm['ESG_Score']:.1f}", "ESG Risk ↓", C_MM), unsafe_allow_html=True)
with c3:
    st.markdown('<span class="model-badge-ns">NSGA-II</span>', unsafe_allow_html=True)
    a, b, c_ = st.columns(3)
    a.markdown(metric_card(f"{bm_ns['Annual_Return']*100:.1f}%", "Best Return", C_NS), unsafe_allow_html=True)
    b.markdown(metric_card(f"{bm_ns['Sharpe_Ratio']:.2f}", "Best Sharpe", C_NS), unsafe_allow_html=True)
    c_.markdown(metric_card(f"{bm_ns['ESG_Score']:.1f}", "ESG Risk ↓", C_NS), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Comparison charts
ctab1, ctab2, ctab3 = st.tabs(["Risk-Return Scatter", "Sharpe & ESG Bars", "Portfolio Count"])

with ctab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mv_df["Annual_Volatility"]*100, y=mv_df["Annual_Return"]*100,
        mode="markers", name=f"MV-ESG ({len(mv_df)} pts)",
        marker=dict(size=7, color=C_MV, opacity=0.45),
        hovertemplate="MV-ESG<br>Return: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=mm_df["Annual_Volatility"]*100, y=mm_df["Annual_Return"]*100,
        mode="markers", name=f"Minimax ({len(mm_df)} pts)",
        marker=dict(size=10, color=C_MM, symbol="diamond", opacity=0.7),
        hovertemplate="Minimax<br>Return: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ns_df["Annual_Volatility"]*100, y=ns_df["Annual_Return"]*100,
        mode="markers", name=f"NSGA-II ({len(ns_df)} Pareto)",
        marker=dict(size=7, color=C_NS, opacity=0.55),
        hovertemplate="NSGA-II<br>Return: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>",
    ))
    # Best portfolio stars
    for bm, c, sym, nm in [(bm_mv, C_MV, "star", "Best MV-ESG"),
                            (bm_mm, C_MM, "star-diamond", "Best Minimax"),
                            (bm_ns, C_NS, "star-square", "Best NSGA-II")]:
        fig.add_trace(go.Scatter(
            x=[bm["Annual_Volatility"]*100], y=[bm["Annual_Return"]*100],
            mode="markers+text", name=nm,
            marker=dict(size=18, color=c, symbol=sym,
                        line=dict(color="white", width=1.5)),
            text=[f"  Sharpe {bm['Sharpe_Ratio']:.2f}"],
            textfont=dict(color="white", size=9),
            textposition="middle right",
        ))
    fig.update_layout(**base_layout("Risk-Return Frontier — All Models"), height=450)
    fig.update_xaxes(title="Annual Volatility (%)")
    fig.update_yaxes(title="Annual Return (%)")
    st.plotly_chart(fig, use_container_width=True)

with ctab2:
    col_left, col_right = st.columns(2)
    with col_left:
        fig_sh = go.Figure(go.Bar(
            x=["MV-ESG", "Minimax ESG\n(Stage 2)", "NSGA-II"],
            y=[bm_mv["Sharpe_Ratio"], bm_mm["Sharpe_Ratio"], bm_ns["Sharpe_Ratio"]],
            marker_color=[C_MV, C_MM, C_NS],
            text=[f"{v:.3f}" for v in [bm_mv["Sharpe_Ratio"], bm_mm["Sharpe_Ratio"], bm_ns["Sharpe_Ratio"]]],
            textposition="outside", textfont=dict(color="white"),
        ))
        fig_sh.update_layout(**base_layout("Best Sharpe Ratio per Model", height=350))
        fig_sh.update_yaxes(title="Sharpe Ratio")
        st.plotly_chart(fig_sh, use_container_width=True)
    with col_right:
        fig_esg = go.Figure(go.Bar(
            x=["MV-ESG", "Minimax ESG\n(Stage 2)", "NSGA-II"],
            y=[bm_mv["ESG_Score"], bm_mm["ESG_Score"], bm_ns["ESG_Score"]],
            marker_color=[C_MV, C_MM, C_NS],
            text=[f"{v:.2f}" for v in [bm_mv["ESG_Score"], bm_mm["ESG_Score"], bm_ns["ESG_Score"]]],
            textposition="outside", textfont=dict(color="white"),
        ))
        fig_esg.add_hline(y=esg_floor, line_color="#f87171", line_dash="dash",
                          annotation_text=f"Ceiling = {esg_floor}", annotation_font_color="#f87171")
        fig_esg.update_layout(**base_layout("ESG Risk Score — Best Portfolio (lower = better)", height=350))
        fig_esg.update_yaxes(title="totalEsg Score")
        st.plotly_chart(fig_esg, use_container_width=True)

with ctab3:
    fig_cnt = go.Figure(go.Bar(
        x=["MV-ESG", "Minimax ESG", "NSGA-II"],
        y=[len(mv_df), len(mm_df), len(ns_df)],
        marker_color=[C_MV, C_MM, C_NS],
        text=[len(mv_df), len(mm_df), len(ns_df)],
        textposition="outside", textfont=dict(color="white"),
    ))
    fig_cnt.update_layout(**base_layout("Number of Portfolios Generated", height=350))
    fig_cnt.update_yaxes(title="# Portfolios")
    st.plotly_chart(fig_cnt, use_container_width=True)

# ══════════════════════════════════════════════
#  PER-MODEL SECTIONS
# ══════════════════════════════════════════════
st.markdown('<div class="section-title">🔬 Explore Each Model</div>', unsafe_allow_html=True)

ticker_sector = dict(zip(esg_df["Symbol"], esg_df["GICS Sector"]))
ticker_name   = dict(zip(esg_df["Symbol"], esg_df["Full Name"]))

def portfolio_detail(port, tickers, color, key_prefix):
    """Render holdings bar + sector pie + ESG metrics for a single portfolio."""
    weight_cols = [f"w_{t}" for t in tickers]
    w     = port[weight_cols].values.astype(float)
    idx   = np.argsort(w)[::-1][:10]
    top_t = [tickers[i] for i in idx]
    top_w = w[idx] * 100
    top_n = [ticker_name.get(t, t) for t in top_t]

    col1, col2 = st.columns(2)
    with col1:
        fig_b = go.Figure(go.Bar(
            x=top_w[::-1], y=top_n[::-1], orientation="h",
            marker=dict(color=top_w[::-1],
                        colorscale=[[0, "#1e293b"], [1, color]], showscale=False),
            text=[f"{v:.1f}%" for v in top_w[::-1]],
            textposition="outside", textfont=dict(color="white", size=10),
        ))
        fig_b.update_layout(**base_layout("Top 10 Holdings",
                             margin=dict(l=10, r=60, t=40, b=10)), height=300)
        fig_b.update_xaxes(title="Weight (%)")
        st.plotly_chart(fig_b, use_container_width=True, key=f"{key_prefix}_bar")

    with col2:
        sector_w = {}
        for t, wt in zip(tickers, w):
            s = ticker_sector.get(t, "Other")
            sector_w[s] = sector_w.get(s, 0) + wt
        sector_w = {k: v for k, v in sector_w.items() if v > 0.005}
        fig_p = go.Figure(go.Pie(
            labels=list(sector_w.keys()),
            values=[v * 100 for v in sector_w.values()],
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Safe),
            textfont=dict(size=9, color="white"),
        ))
        fig_p.update_layout(**base_layout("Sector Allocation",
                             margin=dict(l=10, r=10, t=40, b=10)), height=300)
        fig_p.update_layout(legend=dict(font=dict(size=8, color=FONT_C),
                                        bgcolor="#111827", bordercolor="#1e293b"))
        st.plotly_chart(fig_p, use_container_width=True, key=f"{key_prefix}_pie")

    e1, e2, e3, e4 = st.columns(4)
    for c_, val, lbl, clr in [
        (e1, f"{port['Annual_Return']*100:.1f}%", "Annual Return",     color),
        (e2, f"{port['Annual_Volatility']*100:.1f}%", "Volatility",   "#f87171"),
        (e3, f"{port['Sharpe_Ratio']:.2f}",       "Sharpe Ratio",      "#a78bfa"),
        (e4, f"{port['ESG_Score']:.1f}",           "ESG Risk (↓ better)", "#34d399"),
    ]:
        with c_:
            st.markdown(metric_card(val, lbl, clr), unsafe_allow_html=True)

def portfolio_table(df, key):
    disp = df[["Portfolio_ID", "Annual_Return", "Annual_Volatility",
               "Sharpe_Ratio", "ESG_Score"]].copy()
    disp["Annual_Return"]     = (disp["Annual_Return"]     * 100).round(2)
    disp["Annual_Volatility"] = (disp["Annual_Volatility"] * 100).round(2)
    disp["Sharpe_Ratio"]      = disp["Sharpe_Ratio"].round(3)
    disp["ESG_Score"]         = disp["ESG_Score"].round(2)
    disp.columns = ["ID", "Return (%)", "Volatility (%)", "Sharpe", "ESG Risk (↓)"]
    sort_opt = st.selectbox("Sort by", ["Sharpe", "Return (%)", "ESG Risk (↓)", "Volatility (%)"],
                             key=f"sort_{key}")
    asc = sort_opt == "ESG Risk (↓)"
    st.dataframe(disp.sort_values(sort_opt, ascending=asc),
                 use_container_width=True, height=280, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Download CSV", csv, f"{key}_portfolios.csv",
                       "text/csv", key=f"dl_{key}")

# ── Tabs per model
tab_mv, tab_mm, tab_ns = st.tabs(["📘 MV-ESG", "🔮 Minimax ESG (Stage 2)", "🌿 NSGA-II"])

# ── MV-ESG
with tab_mv:
    n_match = len(mv_f)
    st.caption(f"{len(mv_df)} portfolios generated · {n_match} match filters")

    sub1, sub2 = st.tabs(["Best Portfolio", "All Portfolios"])
    with sub1:
        st.markdown("**Best Sharpe Ratio portfolio**")
        portfolio_detail(bm_mv, tickers_mv, C_MV, "mv_best")
    with sub2:
        fig_mv = go.Figure()
        fig_mv.add_trace(go.Scatter(
            x=mv_df["Annual_Volatility"]*100, y=mv_df["Annual_Return"]*100,
            mode="markers",
            marker=dict(size=7, color=mv_df["ESG_Score"], colorscale="YlOrRd",
                        showscale=True, colorbar=dict(title="ESG Risk", tickfont=dict(color=FONT_C)),
                        opacity=0.65),
            hovertemplate="Return: %{y:.1f}%<br>Vol: %{x:.1f}%<br>ESG: %{marker.color:.1f}<extra></extra>",
        ))
        if n_match:
            fig_mv.add_trace(go.Scatter(
                x=mv_f["Annual_Volatility"]*100, y=mv_f["Annual_Return"]*100,
                mode="markers", name="Matches filters",
                marker=dict(size=9, color=C_MV, line=dict(color="white", width=1)),
            ))
        fig_mv.update_layout(**base_layout("MV-ESG: Return vs Volatility"), height=380)
        fig_mv.update_xaxes(title="Volatility (%)")
        fig_mv.update_yaxes(title="Return (%)")
        st.plotly_chart(fig_mv, use_container_width=True)
        portfolio_table(mv_df, "mv")

# ── Minimax ESG
with tab_mm:
    st.caption("Equal E/S/G weights (10/10/10) · Two-stage lexicographic optimisation")
    st.markdown("**Stage 2 portfolio — max-Sharpe within ESG lock (equal E/S/G weights)**")
    portfolio_detail(bm_mm, mm_tick, C_MM, "mm_best")

# ── NSGA-II
with tab_ns:
    n_match = len(ns_f)
    st.caption(f"{len(ns_df)} Pareto-optimal portfolios · {n_match} match filters")

    sub1, sub2, sub3 = st.tabs(["Best Portfolio", "Pareto Front", "Convergence"])
    with sub1:
        st.markdown("**Best Sharpe Ratio portfolio**")
        portfolio_detail(bm_ns, tickers_ns, C_NS, "ns_best")
    with sub2:
        t1, t2, t3 = st.tabs(["Return vs Risk", "Return vs ESG", "3D"])
        with t1:
            fig_ns1 = go.Figure()
            fig_ns1.add_trace(go.Scatter(
                x=ns_df["Annual_Volatility"]*100, y=ns_df["Annual_Return"]*100,
                mode="markers",
                marker=dict(size=8, color=ns_df["ESG_Score"], colorscale="YlOrRd",
                            showscale=True, colorbar=dict(title="ESG Risk", tickfont=dict(color=FONT_C)),
                            opacity=0.65),
                hovertemplate="Return: %{y:.1f}%<br>Vol: %{x:.1f}%<br>ESG: %{marker.color:.1f}<extra></extra>",
            ))
            if n_match:
                fig_ns1.add_trace(go.Scatter(
                    x=ns_f["Annual_Volatility"]*100, y=ns_f["Annual_Return"]*100,
                    mode="markers", name="Matches filters",
                    marker=dict(size=10, color=C_NS, line=dict(color="white", width=1)),
                ))
            fig_ns1.update_layout(**base_layout("NSGA-II Pareto Front"), height=380)
            fig_ns1.update_xaxes(title="Volatility (%)")
            fig_ns1.update_yaxes(title="Return (%)")
            st.plotly_chart(fig_ns1, use_container_width=True)
        with t2:
            fig_ns2 = go.Figure(go.Scatter(
                x=ns_df["ESG_Score"], y=ns_df["Annual_Return"]*100, mode="markers",
                marker=dict(size=8, color=ns_df["Sharpe_Ratio"], colorscale="Plasma",
                            showscale=True, colorbar=dict(title="Sharpe", tickfont=dict(color=FONT_C)),
                            opacity=0.7),
                hovertemplate="Return: %{y:.1f}%<br>ESG: %{x:.1f}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
            ))
            fig_ns2.update_layout(**base_layout("Return vs ESG Risk Score"), height=380)
            fig_ns2.update_xaxes(title="ESG Risk Score (lower=better)")
            fig_ns2.update_yaxes(title="Return (%)")
            st.plotly_chart(fig_ns2, use_container_width=True)
        with t3:
            fig_ns3 = go.Figure(data=go.Scatter3d(
                x=ns_df["Annual_Volatility"]*100, y=ns_df["Annual_Return"]*100,
                z=ns_df["ESG_Score"], mode="markers",
                marker=dict(size=4, color=ns_df["Sharpe_Ratio"], colorscale="Viridis",
                            showscale=True, colorbar=dict(title="Sharpe", tickfont=dict(color=FONT_C)),
                            opacity=0.8),
                hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<br>ESG: %{z:.1f}<extra></extra>",
            ))
            fig_ns3.update_layout(
                paper_bgcolor=PAPER_BG,
                scene=dict(bgcolor=PLOT_BG,
                           xaxis=dict(title="Volatility (%)", gridcolor=GRID_C, color=FONT_C),
                           yaxis=dict(title="Return (%)",     gridcolor=GRID_C, color=FONT_C),
                           zaxis=dict(title="ESG Risk (↓)",   gridcolor=GRID_C, color=FONT_C)),
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(color=FONT_C), height=450,
            )
            st.plotly_chart(fig_ns3, use_container_width=True)
        portfolio_table(ns_df, "ns")
    with sub3:
        fig_cv = go.Figure(go.Scatter(
            y=ns_hist, mode="lines",
            line=dict(color=C_NS, width=2),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.08)",
        ))
        fig_cv.update_layout(**base_layout("Pareto Front Size per Generation"), height=260)
        fig_cv.update_xaxes(title="Generation")
        fig_cv.update_yaxes(title="Pareto Front Size")
        st.plotly_chart(fig_cv, use_container_width=True)

# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem; margin-top:2rem;'>
  ESG Portfolio Optimiser · MV-ESG · Minimax ESG · NSGA-II · 150 S&P 500 Constituents · 2023–2024<br>
  Built for FTEC4999 Final Year Project
</div>
""", unsafe_allow_html=True)
