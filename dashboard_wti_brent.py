"""
================================================================================
  BRENT / WTI PAIR TRADING SYSTEM  —  Streamlit Dashboard
  Kalman Filter | Walk-Forward | Full Risk Engine
  Run: streamlit run app.py
================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pykalman import KalmanFilter
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brent/WTI Pair Trading",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .signal-box {
        border-radius: 10px;
        padding: 20px 28px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-short { background: rgba(230,57,70,0.15); border: 2px solid #e63946; color: #e63946; }
    .signal-long  { background: rgba(6,214,160,0.15); border: 2px solid #06d6a0; color: #06d6a0; }
    .signal-wait  { background: rgba(255,209,102,0.12); border: 2px solid #ffd166; color: #ffd166; }
    .signal-off   { background: rgba(255,0,110,0.10); border: 2px solid #ff006e; color: #ff006e; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border-radius: 6px 6px 0 0;
        color: #8b949e;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] { background: #21262d; color: #e6edf3; }
    div[data-testid="stMetric"] label { color: #8b949e !important; }
    div[data-testid="stMetric"] div   { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    st.markdown("**Periods**")
    train_start = st.text_input("Train Start", "2018-01-01")
    train_end   = st.text_input("Train End",   "2021-12-31")
    test_start  = st.text_input("Test Start",  "2022-01-01")

    st.markdown("**Z-score Signal**")
    z_entry  = st.slider("Z Entry",          1.0, 4.0, 2.0, 0.1)
    z_exit   = st.slider("Z Exit",           0.0, 1.5, 0.3, 0.05)
    z_stop   = st.slider("Z Stop",           2.5, 6.0, 4.0, 0.1)
    z_window = st.slider("Z Window (days)",  10,  60,  30,  5)

    st.markdown("**Risk & Sizing**")
    capitale       = st.number_input("Capital ($)", 10_000, 10_000_000, 100_000, 10_000)
    risk_per_trade = st.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25) / 100
    max_contracts  = st.slider("Max Contracts", 1, 20, 10, 1)

    st.markdown("**Transaction Costs**")
    slippage = st.number_input("Slippage ($/bbl/side)", 0.005, 0.10, 0.015, 0.005, format="%.3f")
    comm     = st.number_input("Commission ($/contract RT)", 0.50, 5.0, 1.50, 0.25)

    st.markdown("**Regime Filter**")
    regime_window  = st.slider("Regime Window (days)", 20, 120, 60, 10)
    min_corr       = st.slider("Min Rolling Correlation", 0.50, 0.99, 0.70, 0.05)
    max_coint_pval = st.slider("Max Coint p-value",       0.01, 0.20, 0.10, 0.01)

    st.markdown("**Live Contracts**")
    live_brent = st.text_input("Brent Ticker", "BZK26.NYM")
    live_wti   = st.text_input("WTI Ticker",   "CLK26.NYM")
    live_start = st.text_input("Live Data Start", "2025-09-01")

    run_btn = st.button("🚀  Run Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BARREL_LOT       = 1_000
SPREAD_VOL_FLOOR = 0.05
OBS_COV          = 0.5
TRANS_COV        = 0.001

C = dict(
    brent="#f4a261", wti="#457b9d", spread="#a8dadc",
    zpos="#e63946",  zneg="#2a9d8f", equity="#06d6a0",
    hedge="#c77dff", gold="#ffd166", pink="#ff006e",
    bg="#0d1117",    panel="#161b22", grid="#21262d", text="#e6edf3",
)
PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
    font=dict(color=C["text"], size=11),
    xaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C["grid"], showgrid=True, zeroline=False),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor=C["panel"], bordercolor=C["grid"], borderwidth=1),
)


# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def download_pair(ticker_b, ticker_w, start, end=None):
    kw = dict(progress=False)
    if end:
        kw["end"] = end
    b = yf.download(ticker_b, start=start, **kw)["Close"]
    w = yf.download(ticker_w, start=start, **kw)["Close"]
    df = pd.concat([b, w], axis=1).dropna()
    df.columns = ["Brent", "WTI"]
    for col in ["Brent", "WTI"]:
        ret = df[col].pct_change().abs()
        df.loc[ret > 0.15, col] = np.nan
    return df.ffill().dropna()


def cointegration_stats(df):
    _, pval, _ = coint(df["Brent"], df["WTI"])
    adf_b = adfuller(df["Brent"], autolag="AIC")
    adf_w = adfuller(df["WTI"],   autolag="AIC")
    joh   = coint_johansen(df[["Brent", "WTI"]], det_order=0, k_ar_diff=1)
    c95   = bool(joh.lr1[0] > joh.cvt[0][1])
    X     = sm.add_constant(df["WTI"])
    ols   = sm.OLS(df["Brent"], X).fit()
    hedge = float(ols.params["WTI"])
    spread      = df["Brent"] - hedge * df["WTI"]
    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna().reindex(spread_lag.index)
    ar   = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
    beta = float(ar.params.iloc[1])
    hl   = -np.log(2) / beta if beta < 0 else float("nan")
    return dict(
        eg_pval=pval, adf_b=adf_b[1], adf_w=adf_w[1],
        joh_trace=joh.lr1[0], joh_cv95=joh.cvt[0][1], joh_ok=c95,
        hedge=hedge, half_life=hl,
    )


def compute_kalman(df, hedge_init, init_cov=None):
    df  = df.copy()
    y   = df["Brent"].values
    x   = df["WTI"].values
    obs = np.vstack([x, np.ones(len(x))]).T[:, np.newaxis]
    im  = hedge_init if isinstance(hedge_init, np.ndarray) else np.array([hedge_init, 0.0])
    ic  = init_cov if init_cov is not None else 10.0 * np.eye(2)
    kf  = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=im, initial_state_covariance=ic,
        transition_matrices=np.eye(2), observation_matrices=obs,
        observation_covariance=OBS_COV,
        transition_covariance=TRANS_COV * np.eye(2),
    )
    sm_out, sc_out = kf.filter(y)
    df["Hedge_K"]     = sm_out[:, 0]
    df["Intercept_K"] = sm_out[:, 1]
    df["Spread_K"]    = df["Brent"] - df["Hedge_K"] * df["WTI"] - df["Intercept_K"]
    return df, sm_out[-1], sc_out[-1]


def add_zscore(df, window, floor):
    df  = df.copy()
    m   = df["Spread_K"].rolling(window).mean()
    s   = df["Spread_K"].rolling(window).std().clip(lower=floor)
    df["Z"]         = (df["Spread_K"] - m) / s
    df["Roll_Mean"] = m
    df["Roll_Std"]  = s
    return df


def add_regime_filter(df, window, min_corr_, max_pval):
    df     = df.copy()
    n      = len(df)
    regime = np.zeros(n, dtype=int)
    corr_r = np.full(n, np.nan)
    adf_p  = np.full(n, np.nan)
    for i in range(window, n):
        wb       = df["Brent"].iloc[i-window:i].values
        ww       = df["WTI"].iloc[i-window:i].values
        cv       = np.corrcoef(wb, ww)[0, 1]
        corr_r[i] = cv
        try:
            ap = adfuller(df["Spread_K"].iloc[i-window:i].values, autolag=None, maxlag=1)[1]
        except Exception:
            ap = 1.0
        adf_p[i] = ap
        if cv >= min_corr_ and ap <= max_pval:
            regime[i] = 1
    df["Regime"]    = regime
    df["Corr_Roll"] = corr_r
    df["ADF_Pval"]  = adf_p
    return df


def compute_n(capital, risk_pct, z_stop_, spread_vol, barrel_lot, max_c):
    dr = capital * risk_pct
    sd = z_stop_ * spread_vol * barrel_lot
    if sd <= 0:
        return 1
    return max(1, min(int(np.floor(dr / sd)), max_c))


def run_backtest(df, z_entry_, z_exit_, z_stop_, capital_, risk_pt, max_c,
                 slip, com, barrel_lot, use_regime=True):
    df      = df.copy().dropna(subset=["Z", "Spread_K"])
    capital = capital_
    trades  = []
    pos = 0; entry_spread = 0.0; entry_idx = None; n_c = 0
    curve   = [capital]

    for i in range(1, len(df)):
        z   = df["Z"].iloc[i]
        spn = df["Spread_K"].iloc[i]
        rok = (df["Regime"].iloc[i] == 1) if use_regime else True
        sv  = df["Roll_Std"].iloc[i]
        dt  = df.index[i]

        if pos == 0 and not np.isnan(z) and rok:
            if z >= z_entry_:
                pos = -1
            elif z <= -z_entry_:
                pos = 1
            if pos != 0:
                n_c          = compute_n(capital, risk_pt, z_stop_, sv, barrel_lot, max_c)
                entry_spread = spn
                entry_idx    = dt

        elif pos != 0:
            done = abs(z) <= z_exit_ or abs(z) >= z_stop_ or not rok or i == len(df) - 1
            if done:
                raw  = pos * (spn - entry_spread) * barrel_lot * n_c
                cost = slip * barrel_lot * 2 * n_c * 2 + com * n_c * 2
                net  = raw - cost
                capital += net
                et = ("stop"   if abs(z) >= z_stop_  else
                      "regime" if not rok             else "exit")
                trades.append(dict(
                    Entry_Date=entry_idx, Exit_Date=dt,
                    Direction="LONG" if pos == 1 else "SHORT",
                    Z_Exit=round(z, 3), N_Contracts=n_c,
                    Gross_PnL=raw, Cost=cost, Net_PnL=net, Exit_Type=et,
                ))
                pos = 0; entry_idx = None; n_c = 0
        curve.append(capital)

    eq = pd.Series(curve[:len(df)], index=df.index[:len(curve)])
    return eq, pd.DataFrame(trades)


def calc_metrics(equity, trades_df):
    ret   = equity.pct_change().dropna()
    tr    = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    ann   = ((equity.iloc[-1] / equity.iloc[0]) ** (252 / max(len(equity), 1)) - 1) * 100
    sh    = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    neg   = ret[ret < 0]
    so    = ret.mean() / neg.std() * np.sqrt(252) if len(neg) > 1 else 0
    dd    = (equity - equity.cummax()) / equity.cummax()
    mdd   = dd.min() * 100
    cal   = ann / abs(mdd) if mdd != 0 else 0
    var95 = float(np.percentile(ret, 5) * 100)
    cv95  = float(ret[ret <= np.percentile(ret, 5)].mean() * 100)
    wins  = trades_df[trades_df["Net_PnL"] > 0]  if len(trades_df) > 0 else pd.DataFrame()
    loss  = trades_df[trades_df["Net_PnL"] <= 0] if len(trades_df) > 0 else pd.DataFrame()
    wr    = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    pf    = (wins["Net_PnL"].sum() / abs(loss["Net_PnL"].sum())
             if len(loss) > 0 and loss["Net_PnL"].sum() != 0 else float("inf"))
    return dict(
        total_ret=tr, ann_ret=ann, sharpe=sh, sortino=so, calmar=cal,
        max_dd=mdd, var95=var95, cvar95=cv95, win_rate=wr, profit_factor=pf,
        n_trades=len(trades_df),
        stops=(trades_df["Exit_Type"] == "stop").sum() if len(trades_df) > 0 else 0,
        avg_pnl=trades_df["Net_PnL"].mean()  if len(trades_df) > 0 else 0,
        total_cost=trades_df["Cost"].sum()    if len(trades_df) > 0 else 0,
        net_pnl=trades_df["Net_PnL"].sum()    if len(trades_df) > 0 else 0,
        avg_win=wins["Net_PnL"].mean()  if len(wins)  > 0 else 0,
        avg_loss=loss["Net_PnL"].mean() if len(loss)  > 0 else 0,
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#e6edf3; font-size:1.8rem; margin-bottom:0;'>
  🛢️ Brent / WTI Pair Trading System
</h1>
<p style='color:#8b949e; margin-top:4px; font-size:0.9rem;'>
  Kalman Filter Hedge Ratio &nbsp;·&nbsp; Regime Filter &nbsp;·&nbsp; Walk-Forward Validation
</p>
<hr style='border-color:#21262d; margin:12px 0 20px;'>
""", unsafe_allow_html=True)

if not run_btn:
    st.info("👈  Configure parameters in the sidebar and click **Run Analysis**")
    st.stop()

# ─────────────────────────────────────────────
# DATA & COMPUTATION
# ─────────────────────────────────────────────
with st.spinner("Downloading market data..."):
    df_train = download_pair("BZ=F", "CL=F", train_start, train_end)
    df_test  = download_pair("BZ=F", "CL=F", test_start,  None)
    df_all   = download_pair("BZ=F", "CL=F", train_start, None)
    df_live  = download_pair(live_brent, live_wti, live_start)
    if len(df_live) < 20:
        df_live     = download_pair("BZ=F", "CL=F", live_start)
        live_label  = "Continuous (fallback)"
    else:
        live_label  = f"{live_brent} / {live_wti}"

st.success(f"Data loaded — Train: {len(df_train)} bars | Test: {len(df_test)} bars | Live ({live_label}): {len(df_live)} bars")

with st.spinner("Running cointegration tests..."):
    stats_train = cointegration_stats(df_train)
    stats_test  = cointegration_stats(df_test)

with st.spinner("Running Kalman filter..."):
    df_pre     = download_pair("BZ=F", "CL=F", "2016-01-01", "2017-12-31")
    X_pre      = sm.add_constant(df_pre["WTI"])
    hedge_init = float(sm.OLS(df_pre["Brent"], X_pre).fit().params["WTI"])

    df_train, last_state, last_cov = compute_kalman(df_train, hedge_init)
    df_train = add_zscore(df_train, z_window, SPREAD_VOL_FLOOR)
    df_train = add_regime_filter(df_train, regime_window, min_corr, max_coint_pval)

    df_test, _, _ = compute_kalman(df_test, last_state, last_cov)
    df_test = add_zscore(df_test, z_window, SPREAD_VOL_FLOOR)
    df_test = add_regime_filter(df_test, regime_window, min_corr, max_coint_pval)

    df_all, _, _ = compute_kalman(df_all, hedge_init)
    df_all = add_zscore(df_all, z_window, SPREAD_VOL_FLOOR)
    df_all = add_regime_filter(df_all, regime_window, min_corr, max_coint_pval)

with st.spinner("Running backtest..."):
    bt_args = dict(
        z_entry_=z_entry, z_exit_=z_exit, z_stop_=z_stop,
        capital_=capitale, risk_pt=risk_per_trade, max_c=max_contracts,
        slip=slippage, com=comm, barrel_lot=BARREL_LOT,
    )
    eq_train, tr_train = run_backtest(df_train, **bt_args)
    eq_test,  tr_test  = run_backtest(df_test,  **bt_args)
    m_train = calc_metrics(eq_train, tr_train)
    m_test  = calc_metrics(eq_test,  tr_test)
    all_trades = (pd.concat([tr_train, tr_test]).reset_index(drop=True)
                  if len(tr_train) > 0 and len(tr_test) > 0
                  else (tr_train if len(tr_train) > 0 else tr_test))

with st.spinner("Walk-forward validation..."):
    test_idx   = df_all.index[df_all.index >= pd.Timestamp(test_start)]
    split_size = len(test_idx) // 5
    wf_rows    = []
    for i in range(5):
        si  = test_idx[i * split_size]
        ei  = test_idx[min((i+1) * split_size - 1, len(test_idx)-1)]
        dtr = df_all[df_all.index < si]
        dte = df_all[(df_all.index >= si) & (df_all.index <= ei)]
        if len(dtr) < 200 or len(dte) < 20:
            continue
        hl  = float(dtr["Hedge_K"].iloc[-1]) if "Hedge_K" in dtr.columns else hedge_init
        dtk, _, _ = compute_kalman(dte, hl)
        dtk = add_zscore(dtk, z_window, SPREAD_VOL_FLOOR)
        dtk = add_regime_filter(dtk, regime_window, min_corr, max_coint_pval)
        eq_w, tr_w = run_backtest(dtk, **bt_args)
        if len(tr_w) == 0:
            continue
        r    = (eq_w.iloc[-1] / eq_w.iloc[0] - 1) * 100
        rets = eq_w.pct_change().dropna()
        sh   = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        wf_rows.append(dict(Split=i+1, Start=str(si.date()), End=str(ei.date()),
                            Return_pct=round(r, 2), Sharpe=round(sh, 2), N_Trades=len(tr_w)))
    wf_df = pd.DataFrame(wf_rows)

# Live signal computation
df_live_k, _, _ = compute_kalman(df_live, df_test["Hedge_K"].iloc[-1])
df_live_k = add_zscore(df_live_k, z_window, SPREAD_VOL_FLOOR)
df_live_k = add_regime_filter(df_live_k, regime_window, min_corr, max_coint_pval)
last          = df_live_k.iloc[-1]
z_live        = float(last["Z"])
hedge_live    = float(last["Hedge_K"])
spread_live   = float(last["Spread_K"])
regime_live   = int(last["Regime"])
corr_live     = float(last["Corr_Roll"])
brent_px      = float(last["Brent"])
wti_px        = float(last["WTI"])
sv_live       = float(last["Roll_Std"])
n_live        = compute_n(capitale, risk_per_trade, z_stop, sv_live, BARREL_LOT, max_contracts)
cost_live     = slippage * BARREL_LOT * 2 * n_live * 2 + comm * n_live * 2
stop_live     = z_stop * sv_live * BARREL_LOT * n_live
target_live   = z_exit * sv_live * BARREL_LOT * n_live


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_live, tab_bt, tab_coint, tab_charts, tab_wf = st.tabs([
    "🔴  Live Signal",
    "📊  Backtest",
    "🔬  Cointegration",
    "📈  Charts",
    "🔄  Walk-Forward",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNAL
# ═══════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### Live Signal — May 2026 Contracts (K26)")
    st.caption(f"Data source: {live_label}  ·  Last update: {df_live_k.index[-1].date()}")

    # Signal box
    if regime_live == 0:
        st.markdown(
            '<div class="signal-box signal-off">⚠️  REGIME INACTIVE — No operational signal</div>',
            unsafe_allow_html=True)
    elif abs(z_live) >= z_entry:
        if z_live > 0:
            st.markdown(
                f'<div class="signal-box signal-short">'
                f'🔴 SHORT SPREAD — SELL {n_live}x {live_brent} &nbsp;|&nbsp; BUY {n_live}x {live_wti}'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="signal-box signal-long">'
                f'🟢 LONG SPREAD — BUY {n_live}x {live_brent} &nbsp;|&nbsp; SELL {n_live}x {live_wti}'
                f'</div>', unsafe_allow_html=True)
    else:
        remaining = z_entry - abs(z_live)
        st.markdown(
            f'<div class="signal-box signal-wait">'
            f'⏳ NO SIGNAL &nbsp;·&nbsp; Z = {z_live:+.3f} &nbsp;·&nbsp; {remaining:.2f} z-points to entry threshold'
            f'</div>', unsafe_allow_html=True)

    # Market data row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brent (BZK26)", f"${brent_px:.2f}")
    c2.metric("WTI (CLK26)",   f"${wti_px:.2f}")
    c3.metric("Raw Spread (B−W)", f"${brent_px - wti_px:.2f}")
    c4.metric("Z-score",          f"{z_live:+.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Kalman Hedge Ratio",    f"{hedge_live:.4f}")
    c6.metric("Kalman Adj. Spread",    f"{spread_live:.4f}")
    c7.metric("Rolling Correlation",   f"{corr_live:.3f}")
    c8.metric("Regime",  "✅ Active" if regime_live == 1 else "🔴 Inactive")

    if abs(z_live) >= z_entry and regime_live == 1:
        st.markdown("---")
        st.markdown("#### Order Details")
        co1, co2, co3, co4 = st.columns(4)
        co1.metric("Contracts",          str(n_live))
        co2.metric("Estimated Cost",     f"${cost_live:.0f}")
        co3.metric(f"Stop (z={z_stop})", f"${stop_live:,.0f}")
        co4.metric(f"Target (z={z_exit})", f"${target_live:,.0f}")
        noz = brent_px * BARREL_LOT * n_live
        st.info(f"💡  Notional per leg ≈ **${noz:,.0f}** ({n_live} contracts × {BARREL_LOT:,} bbl × ${brent_px:.2f})")

    # Recent Z-score chart
    st.markdown("---")
    st.markdown("#### Z-score — Last 90 Days")
    df_recent = df_live_k.tail(90)
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=df_recent.index, y=df_recent["Z"], name="Z-score",
        line=dict(color=C["spread"], width=1.5),
        fill="tozeroy", fillcolor="rgba(168,218,220,0.08)"))
    for lvl, col, dash in [(z_entry, C["zpos"], "dash"), (-z_entry, C["zneg"], "dash"),
                            (z_stop,  C["pink"], "dot"),  (-z_stop,  C["pink"], "dot")]:
        fig_z.add_hline(y=lvl, line=dict(color=col, dash=dash, width=1))
    fig_z.add_hline(y=0, line=dict(color="#ffffff", dash="dot", width=0.5))
    fig_z.update_layout(**PLOTLY_LAYOUT, height=300,
                         title=dict(text="Live Z-score (last 90 days)", font=dict(color=C["text"])))
    fig_z.update_xaxes(gridcolor=C["grid"])
    fig_z.update_yaxes(gridcolor=C["grid"], range=[-6, 6])
    st.plotly_chart(fig_z, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ═══════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown("### Performance Report")

    for label, m, _ in [
        ("IN-SAMPLE (2018–2021)",     m_train, C["equity"]),
        ("OUT-OF-SAMPLE (2022–today)", m_test,  C["gold"]),
    ]:
        st.markdown(f"#### {label}")
        r1 = st.columns(5)
        r1[0].metric("Total Return",    f"{m['total_ret']:+.2f}%")
        r1[1].metric("Ann. Return",     f"{m['ann_ret']:+.2f}%")
        r1[2].metric("Sharpe Ratio",    f"{m['sharpe']:.3f}")
        r1[3].metric("Sortino Ratio",   f"{m['sortino']:.3f}")
        r1[4].metric("Max Drawdown",    f"{m['max_dd']:.2f}%")

        r2 = st.columns(5)
        r2[0].metric("Calmar Ratio",    f"{m['calmar']:.3f}")
        r2[1].metric("VaR 95%",         f"{m['var95']:.3f}%")
        r2[2].metric("CVaR 95%",        f"{m['cvar95']:.3f}%")
        r2[3].metric("Win Rate",        f"{m['win_rate']:.1f}%")
        r2[4].metric("Profit Factor",   f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞")

        r3 = st.columns(5)
        r3[0].metric("# Trades",        str(m["n_trades"]))
        r3[1].metric("Stop Loss Hits",  str(m["stops"]))
        r3[2].metric("Avg Win",         f"${m['avg_win']:+,.0f}")
        r3[3].metric("Avg Loss",        f"${m['avg_loss']:+,.0f}")
        r3[4].metric("Net PnL",         f"${m['net_pnl']:+,.0f}")
        st.divider()

    # Trade log
    if len(all_trades) > 0:
        st.markdown("#### Full Trade Log")
        display_cols = ["Entry_Date", "Exit_Date", "Direction", "N_Contracts",
                        "Gross_PnL", "Cost", "Net_PnL", "Exit_Type"]
        tl = all_trades[display_cols].copy()
        tl["Net_PnL"]   = tl["Net_PnL"].round(0).astype(int)
        tl["Gross_PnL"] = tl["Gross_PnL"].round(0).astype(int)
        tl["Cost"]      = tl["Cost"].round(0).astype(int)

        def color_pnl(val):
            if isinstance(val, (int, float)):
                return f"color: {'#06d6a0' if val > 0 else '#e63946'}"
            return ""

        st.dataframe(
            tl.style.applymap(color_pnl, subset=["Net_PnL", "Gross_PnL"]),
            use_container_width=True, height=420)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — COINTEGRATION
# ═══════════════════════════════════════════════════════════════
with tab_coint:
    st.markdown("### Cointegration Statistics")

    col_tr, col_te = st.columns(2)
    for col_, stats_, lbl_ in [
        (col_tr, stats_train, "In-Sample 2018–2021"),
        (col_te, stats_test,  "Out-of-Sample 2022–today"),
    ]:
        with col_:
            st.markdown(f"#### {lbl_}")
            rows = [
                ("Engle-Granger p-value",
                 f"{stats_['eg_pval']:.4f}",
                 "✅ Cointegrated" if stats_["eg_pval"] < 0.05 else "⚠️ Not significant"),
                ("ADF Brent p-value",
                 f"{stats_['adf_b']:.4f}",
                 "✅ I(1)" if stats_["adf_b"] > 0.05 else "⚠️ Stationary"),
                ("ADF WTI p-value",
                 f"{stats_['adf_w']:.4f}",
                 "✅ I(1)" if stats_["adf_w"] > 0.05 else "⚠️ Stationary"),
                ("Johansen trace (r=0)",
                 f"{stats_['joh_trace']:.2f}  (CV95% = {stats_['joh_cv95']:.2f})",
                 "✅" if stats_["joh_ok"] else "⚠️"),
                ("OLS Hedge Ratio",
                 f"{stats_['hedge']:.4f}", "—"),
                ("Half-life (days)",
                 f"{stats_['half_life']:.1f}",
                 "✅ Optimal" if 5 < stats_["half_life"] < 60 else "⚠️ Out of range"),
            ]
            for name, val, status in rows:
                ca, cb, cc = st.columns([3, 2, 1])
                ca.write(name)
                cb.write(f"**{val}**")
                cc.write(status)

    st.divider()
    st.markdown(f"**Kalman initial hedge ratio** (pre-train 2016–2017): `{hedge_init:.4f}`")
    rg_tr = df_train["Regime"].iloc[regime_window:].mean() * 100
    rg_te = df_test["Regime"].iloc[regime_window:].mean() * 100
    st.markdown(f"**Active regime** — Train: `{rg_tr:.1f}%` | Test: `{rg_te:.1f}%` of trading days")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — CHARTS
# ═══════════════════════════════════════════════════════════════
with tab_charts:

    # Prices
    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(x=df_all.index, y=df_all["Brent"],
                                name="Brent", line=dict(color=C["brent"], width=1.3)))
    fig_px.add_trace(go.Scatter(x=df_all.index, y=df_all["WTI"],
                                name="WTI", line=dict(color=C["wti"], width=1.3)))
    fig_px.add_vrect(x0=train_start, x1=train_end, fillcolor="cyan", opacity=0.04,
                     layer="below", annotation_text="In-Sample",
                     annotation_position="top left", annotation_font_color=C["text"])
    fig_px.add_vrect(x0=test_start, x1=str(df_all.index[-1].date()),
                     fillcolor="lime", opacity=0.04, layer="below",
                     annotation_text="Out-of-Sample", annotation_position="top left",
                     annotation_font_color=C["text"])
    fig_px.update_layout(**PLOTLY_LAYOUT, height=300,
                          title=dict(text="Brent vs WTI — Continuous Prices", font=dict(color=C["text"])))
    fig_px.update_xaxes(gridcolor=C["grid"])
    fig_px.update_yaxes(gridcolor=C["grid"])
    st.plotly_chart(fig_px, use_container_width=True)

    # Hedge ratio + Spread
    c_l, c_r = st.columns(2)
    with c_l:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=df_all.index, y=df_all["Hedge_K"],
                                   name="Kalman", line=dict(color=C["hedge"], width=1.3)))
        fig_h.add_hline(y=stats_train["hedge"],
                        line=dict(color=C["gold"], dash="dash", width=1),
                        annotation_text=f"OLS = {stats_train['hedge']:.3f}",
                        annotation_font_color=C["gold"])
        fig_h.update_layout(**PLOTLY_LAYOUT, height=280,
                             title=dict(text="Kalman Hedge Ratio (dynamic)", font=dict(color=C["text"])))
        fig_h.update_xaxes(gridcolor=C["grid"])
        fig_h.update_yaxes(gridcolor=C["grid"])
        st.plotly_chart(fig_h, use_container_width=True)

    with c_r:
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=df_all.index, y=df_all["Spread_K"],
                                    name="Spread", line=dict(color=C["spread"], width=1.0),
                                    fill="tozeroy", fillcolor="rgba(168,218,220,0.07)"))
        fig_sp.add_hline(y=0, line=dict(color="#ffffff", dash="dot", width=0.7))
        fig_sp.update_layout(**PLOTLY_LAYOUT, height=280,
                              title=dict(text="Kalman Spread", font=dict(color=C["text"])))
        fig_sp.update_xaxes(gridcolor=C["grid"])
        fig_sp.update_yaxes(gridcolor=C["grid"])
        st.plotly_chart(fig_sp, use_container_width=True)

    # Z-score full history
    fig_zf = go.Figure()
    z_s = df_all["Z"].copy()
    fig_zf.add_trace(go.Scatter(x=df_all.index, y=z_s, name="Z-score",
                                line=dict(color=C["spread"], width=0.9)))
    for lvl, col, dash, lbl in [
        ( z_entry, C["zpos"], "dash", f"Entry +{z_entry}"),
        (-z_entry, C["zneg"], "dash", f"Entry -{z_entry}"),
        ( z_stop,  C["pink"], "dot",  f"Stop +{z_stop}"),
        (-z_stop,  C["pink"], "dot",  f"Stop -{z_stop}"),
    ]:
        fig_zf.add_hline(y=lvl, line=dict(color=col, dash=dash, width=1),
                         annotation_text=lbl, annotation_font_color=col)
    regime_off_idx = df_all[df_all["Regime"] == 0].index
    if len(regime_off_idx) > 0:
        fig_zf.add_trace(go.Scatter(
            x=list(regime_off_idx) + list(regime_off_idx[::-1]),
            y=[6] * len(regime_off_idx) + [-6] * len(regime_off_idx),
            fill="toself", fillcolor="rgba(255,0,110,0.08)",
            line=dict(width=0), name="Regime OFF", showlegend=True,
        ))
    fig_zf.add_vline(x=str(pd.Timestamp(test_start)),
                     line=dict(color="#ffffff", dash="dot", width=0.8))
    fig_zf.update_layout(**PLOTLY_LAYOUT, height=340,
                          title=dict(text="Kalman Z-score (full history)", font=dict(color=C["text"])))
    fig_zf.update_xaxes(gridcolor=C["grid"])
    fig_zf.update_yaxes(gridcolor=C["grid"], range=[-6, 6])
    st.plotly_chart(fig_zf, use_container_width=True)

    # Equity curves
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq_train.index, y=eq_train.values,
                                name="Train Equity", line=dict(color=C["equity"], width=1.8),
                                fill="tozeroy", fillcolor="rgba(6,214,160,0.06)"))
    fig_eq.add_trace(go.Scatter(x=eq_test.index, y=eq_test.values,
                                name="Test Equity", line=dict(color=C["gold"], width=1.8),
                                fill="tozeroy", fillcolor="rgba(255,209,102,0.06)"))
    fig_eq.add_hline(y=capitale, line=dict(color="#ffffff", dash="dot", width=0.7))
    fig_eq.update_layout(**PLOTLY_LAYOUT, height=320,
                          title=dict(text="Equity Curve — Train vs Test", font=dict(color=C["text"])))
    fig_eq.update_xaxes(gridcolor=C["grid"])
    fig_eq.update_yaxes(gridcolor=C["grid"])
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown
    dd_tr = (eq_train / eq_train.cummax() - 1) * 100
    dd_te = (eq_test  / eq_test.cummax()  - 1) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_tr.index, y=dd_tr.values, name="Train DD",
                                line=dict(color="cyan", width=1.2),
                                fill="tozeroy", fillcolor="rgba(0,255,255,0.07)"))
    fig_dd.add_trace(go.Scatter(x=dd_te.index, y=dd_te.values, name="Test DD",
                                line=dict(color=C["gold"], width=1.2),
                                fill="tozeroy", fillcolor="rgba(255,209,102,0.07)"))
    fig_dd.update_layout(**PLOTLY_LAYOUT, height=260,
                          title=dict(text="Drawdown (%)", font=dict(color=C["text"])))
    fig_dd.update_xaxes(gridcolor=C["grid"])
    fig_dd.update_yaxes(gridcolor=C["grid"])
    st.plotly_chart(fig_dd, use_container_width=True)

    # P&L per trade
    if len(all_trades) > 0:
        pnl_vals  = all_trades["Net_PnL"].values
        bar_colors = [C["zneg"] if v > 0 else C["zpos"] for v in pnl_vals]
        split_i   = (all_trades["Entry_Date"].apply(
            lambda d: pd.Timestamp(d) < pd.Timestamp(test_start))).sum()
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Bar(x=list(range(len(all_trades))), y=pnl_vals,
                                 marker_color=bar_colors, name="Net P&L", opacity=0.85))
        fig_pnl.add_hline(y=0, line=dict(color="#ffffff", dash="dot", width=0.7))
        fig_pnl.add_vline(x=split_i - 0.5, line=dict(color="#ffffff", dash="dot", width=1),
                          annotation_text="Train / Test", annotation_font_color=C["text"])
        fig_pnl.update_layout(**PLOTLY_LAYOUT, height=280,
                               title=dict(text="P&L per Trade (Net USD)", font=dict(color=C["text"])),
                               xaxis_title="Trade #")
        fig_pnl.update_xaxes(gridcolor=C["grid"])
        fig_pnl.update_yaxes(gridcolor=C["grid"])
        st.plotly_chart(fig_pnl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════════
with tab_wf:
    st.markdown("### Walk-Forward Validation (5 splits, expanding window)")
    st.caption("Each split uses all preceding data as training. No future data leakage.")

    if len(wf_df) > 0:
        wf_display = wf_df.rename(columns={
            "Return_pct": "Return (%)", "N_Trades": "# Trades"
        })
        st.dataframe(
            wf_display.style.applymap(
                lambda v: ("color: #06d6a0" if isinstance(v, float) and v > 0
                           else "color: #e63946" if isinstance(v, float) and v < 0 else ""),
                subset=["Return (%)"]
            ), use_container_width=True)

        consistency = (wf_df["Return_pct"] > 0).mean() * 100
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Avg Return",       f"{wf_df['Return_pct'].mean():+.2f}%")
        col_b.metric("Avg Sharpe",       f"{wf_df['Sharpe'].mean():.2f}")
        col_c.metric("Consistency",      f"{consistency:.0f}%")

        fig_wf = go.Figure()
        bar_c  = [C["zneg"] if v > 0 else C["zpos"] for v in wf_df["Return_pct"]]
        fig_wf.add_trace(go.Bar(
            x=[f"Split {r}" for r in wf_df["Split"]],
            y=wf_df["Return_pct"],
            marker_color=bar_c,
            text=[f"{v:+.1f}%" for v in wf_df["Return_pct"]],
            textposition="outside",
        ))
        fig_wf.add_hline(y=0, line=dict(color="#ffffff", dash="dot", width=0.7))
        fig_wf.update_layout(**PLOTLY_LAYOUT, height=320,
                              title=dict(text="Return (%) per Split", font=dict(color=C["text"])),
                              yaxis_title="Return (%)")
        fig_wf.update_xaxes(gridcolor=C["grid"])
        fig_wf.update_yaxes(gridcolor=C["grid"])
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.warning("No walk-forward splits completed with sufficient data.")