# 🛢️ Brent / WTI Pair Trading System

> Institutional-grade statistical arbitrage dashboard for the Brent–WTI crude oil spread.
> Kalman Filter hedge ratio · Regime Filter · Walk-Forward Validation · Streamlit UI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project implements a **mean-reversion pair trading strategy** on the Brent/WTI crude oil spread using a dynamic Kalman Filter hedge ratio, a rolling regime filter, and a rigorous walk-forward validation framework. The strategy is wrapped in an interactive Streamlit dashboard designed for daily monitoring at the desk.

The system is built with institutional-grade standards in mind:

- **No look-ahead bias** — the Kalman Filter is strictly causal; the test period initialises from the last state of the training period
- **Realistic transaction costs** — flat commission per contract (not per barrel) + per-side slippage
- **Risk-based position sizing** — contracts sized as a function of capital at risk, z-stop distance, and spread volatility
- **Regime filter** — trading is only allowed when rolling correlation and ADF p-value confirm the spread is actively mean-reverting
- **Walk-forward validation** — 5 expanding-window splits on the out-of-sample period to test consistency

---

## Methodology

### 1. Cointegration Testing

Before fitting any model, the spread is validated using:

- **Engle-Granger test** — confirms long-run cointegration between Brent and WTI
- **Augmented Dickey-Fuller (ADF)** — confirms each series is integrated of order I(1)
- **Johansen trace test** — provides an independent cointegration rank check
- **OLS half-life** — estimates the mean-reversion speed via AR(1) on the spread

### 2. Kalman Filter Hedge Ratio

The hedge ratio is estimated dynamically using a **Kalman Filter** rather than a static OLS regression. This allows the hedge ratio to adapt to structural changes in the Brent/WTI relationship over time (geopolitical events, quality differentials, logistics shifts).

The filter is initialised using a hedge ratio estimated on a **pre-training period (2016–2017)** — entirely separate from both the in-sample and out-of-sample periods — eliminating any warm-up bias.

```
State vector:  [hedge_ratio, intercept]
Observation:   Brent_t = hedge_t * WTI_t + intercept_t + noise
Transition:    random walk (hedge ratio can drift)
```

### 3. Z-score Signal

```
Z_t = (Spread_t - rolling_mean_t) / rolling_std_t
```

| Level | Action |
|-------|--------|
| Z > +entry | SHORT spread (Sell Brent, Buy WTI) |
| Z < −entry | LONG spread (Buy Brent, Sell WTI) |
| \|Z\| < exit | Close position (mean reversion achieved) |
| \|Z\| > stop | Stop loss |

### 4. Regime Filter

Trades are only taken when **both** conditions hold over a rolling window:

- Rolling correlation between Brent and WTI ≥ threshold (default: 0.70)
- ADF p-value on the rolling spread ≤ threshold (default: 0.10)

This prevents trading during periods of structural decoupling (e.g., embargoes, extreme backwardation divergence, major refinery outages).

### 5. Position Sizing

```
n_contracts = floor( (capital × risk%) / (z_stop × spread_vol × barrel_lot) )
```

Risk is defined in dollar terms, sized so that a stop-loss hit equals exactly `risk%` of current capital.

### 6. Transaction Costs

```
total_cost = slippage ($/bbl) × barrel_lot × 2 legs × 2 sides
           + commission ($/contract RT) × n_contracts × 2 sides
```

Default values reflect institutional desk rates:
- Slippage: **$0.015/bbl per side** (~1.5 ticks)
- Commission: **$1.50/contract round-turn** (flat)

### 7. Walk-Forward Validation

The out-of-sample period is divided into 5 sequential, non-overlapping splits. Each split uses all preceding data as its training set (expanding window). This tests whether the strategy's edge is consistent over time, not just an artefact of parameter fitting.

---

## Project Structure

```
.
├── dashboard_wti_brent.py              # Streamlit dashboard (main entry point)
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Pabliito13/Crude-Oil-Stat-Arb.git
cd Crude-Oil-Stat-Arb

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard_wti_brent.py
```

---

## Dashboard

The dashboard is organised into five tabs:

| Tab | Content |
|-----|---------|
| 🔴 Live Signal | Current z-score, signal direction, order details (contracts, cost, stop, target) |
| 📊 Backtest | Full performance report — Sharpe, Sortino, Calmar, Max DD, VaR, CVaR, trade log |
| 🔬 Cointegration | Engle-Granger, ADF, Johansen, OLS hedge ratio, half-life for both periods |
| 📈 Charts | Interactive Plotly charts — prices, hedge ratio, spread, z-score, equity, drawdown, P&L |
| 🔄 Walk-Forward | 5-split expanding walk-forward with return, Sharpe, and consistency metrics |

All parameters (z-score thresholds, capital, risk %, costs, regime filter, live tickers) are configurable from the sidebar without touching the code.

---

## Live Contracts

The dashboard is configured by default for **May 2026 futures**:

| Leg | Ticker | Exchange |
|-----|--------|----------|
| Brent | `BZK26.NYM` | ICE (via CME Globex) |
| WTI | `CLK26.NYM` | NYMEX |

Tickers can be changed from the sidebar. If the specific contract has insufficient data, the system automatically falls back to continuous front-month contracts (`BZ=F` / `CL=F`).

> **Contract month codes:** F=Jan G=Feb H=Mar J=Apr **K=May** M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec

---

## Backtest Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Train period | 2018–2021 | In-sample parameter fitting |
| Test period | 2022–today | Out-of-sample evaluation |
| Pre-train | 2016–2017 | Kalman initialisation only |
| Z Entry | 2.0 | Signal entry threshold |
| Z Exit | 0.3 | Mean-reversion exit |
| Z Stop | 4.0 | Stop loss |
| Z Window | 30 days | Rolling z-score window |
| Capital | $100,000 | Starting equity |
| Risk/trade | 1% | Max capital at risk per trade |
| Max contracts | 10 | Position cap |
| Slippage | $0.015/bbl | Per side |
| Commission | $1.50/RT | Per contract |
| Regime window | 60 days | Rolling regime assessment |
| Min correlation | 0.70 | Minimum rolling Brent/WTI correlation |
| Max coint p-val | 0.10 | ADF significance threshold |

---

## Dependencies

```
streamlit>=1.35
yfinance>=0.2.40
pandas>=2.0
numpy>=1.26
statsmodels>=0.14
pykalman>=0.9.7
plotly>=5.20
```

---

## Limitations & Known Caveats

- **Continuous contract data** — yfinance continuous futures (`BZ=F`, `CL=F`) are not Panama-adjusted. Roll spikes are filtered with a 15% daily return threshold, but residual roll noise may generate occasional spurious signals.
- **Small trade count** — the strategy generates ~6 trades/year. Metrics such as win rate and profit factor carry wide confidence intervals at this sample size.
- **Single-leg assumption** — the backtest assumes simultaneous execution of both legs. In practice, leg risk between order placement and fill is non-zero, especially in fast markets.
- **No intraday granularity** — signals are computed on daily closes. For live trading, intraday z-score monitoring is recommended before entering.

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past backtest performance does not guarantee future results. Always conduct your own due diligence before trading.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
