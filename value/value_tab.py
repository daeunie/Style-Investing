"""
Value tab -- real-yield value on US Treasury ETFs (duration timing within a band)
----------------------------------------------------------------------------------
From us_value_style_data_processing.ipynb (grid spec chosen in value_export_cell.py):
    value z_n  = (real yield_n − trailing mean) / trailing std over WINDOW months (n = 2 / 5 / 10 / 20y)
                 high z = real yields high vs their own history = bonds cheap
    target D   = center + half × (2Φ(mean z) − 1)          (e.g. 7 ± 1 years for "Band 6-8")
    weights    = closest to 25% each, long-only, sum 100%, portfolio OAD = target D
Real yields come from a model dataset that is not updated live, so this tab reads the notebook export only:
    value/data/value_backtest.csv, value/data/value_signals.csv, value/data/value_config.json
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from carry.carry_tab import (PALETTE, _style, _ts, _x, line_chart, perf_stats, relative_stats,
                             style_perf_table, weights_chart)

DATA_DIR = Path(__file__).resolve().parent / "data"
ETFS = ["SHY", "IEI", "IEF", "TLT"]
TENOR = {"SHY": 2, "IEI": 5, "IEF": 10, "TLT": 20}

try:
    CFG = json.loads((DATA_DIR / "value_config.json").read_text())
except Exception:
    CFG = {"signal": "Z120", "window": 120, "rule": "Band 6-8", "center": 7.0, "half": 1.0,
           "z_scale": 2.0, "cost_bps": 10.0, "rf": "1-year Treasury"}

_center = (f"{CFG['center']:.0f}" if isinstance(CFG["center"], (int, float)) else CFG["center"])
METHOD_SUMMARY = (
    f"Each month, the real yield at 2 / 5 / 10 / 20 years is compared with its own last {CFG['window']} months as a "
    "z-score (high = real yields high = bonds cheap). The average z sets the target portfolio duration: "
    f"{_center} ± {CFG['half']:.1f} years, longer when bonds are cheap and shorter when expensive. Weights stay as "
    "close to 25% each as the duration target allows (long-only)."
)


@st.cache_data(show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name, index_col="ym")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def z_chart(z: pd.DataFrame):
    long = z.reset_index(names="date").melt("date", var_name="ETF", value_name="z")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=_x(), y=alt.Y("z:Q", title="Real-yield z-score"),
        color=alt.Color("ETF:N", scale=alt.Scale(domain=ETFS, range=[PALETTE[e] for e in ETFS])),
        tooltip=["date:T", "ETF:N", alt.Tooltip("z:Q", format=".2f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9AA3B5").encode(y="y:Q")
    return _style(zero + lines, 280)


def render(show_header: bool = True):
    if show_header:
        st.header("Value")
        st.caption(METHOD_SUMMARY)

    try:
        bt, sig = _load("value_backtest.csv"), _load("value_signals.csv")
    except FileNotFoundError:
        st.info("Value results are not uploaded yet. Run value_export_cell.py at the end of the value notebook "
                "and commit value/data/ to the repo.")
        return

    sub1, sub2 = st.tabs(["This month's signal", "Historical backtest"])

    # ---------- Latest signal ----------
    with sub1:
        t = sig.index[-1]
        last, prev = sig.iloc[-1], sig.iloc[-2]
        st.subheader("This Month's Recommended Positioning")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positioning for", (t + 1).strftime("%B %Y"))
        c2.metric("Signal as of (month-end)", t.strftime("%b %Y"))
        c3.metric("Average z", f"{last['z_mean']:+.2f}", "cheap" if last["z_mean"] > 0 else "expensive",
                  delta_color="off")
        c4.metric("Target duration", f"{last['dur_target']:.2f} yrs")

        rows = [{"Instrument": e, "Real yield tenor": f"{TENOR[e]}y", "Z-score": f"{last[f'z_{e}']:+.2f}",
                 "OAD (yrs)": f"{last[f'oad_{e}']:.2f}", "Weight (%)": f"{last[f'w_{e}'] * 100:.1f}",
                 "Change (pp)": f"{(last[f'w_{e}'] - prev[f'w_{e}']) * 100:+.1f}"} for e in ETFS]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"Settings: {CFG['signal']} ({CFG['window']}-month z-score), duration rule {CFG['rule']}. "
                   "Real yields are a model dataset, so this signal updates when the notebook is re-exported "
                   f"(latest: {t.strftime('%b %Y')}).")

        st.subheader("Real-Yield Z-score, Last 10 Years")
        z = sig[[f"z_{e}" for e in ETFS]].rename(columns=lambda c: c[2:]).loc[t - 119:t]
        z.index = _ts(z.index)
        st.altair_chart(z_chart(z), width="stretch")

    # ---------- Historical backtest ----------
    with sub2:
        cmp = bt.dropna(subset=["ret_bbg"])
        rets = pd.DataFrame({"Strategy": cmp["ret_strategy"], "Equal weight (25% each)": cmp["ret_ew"],
                             "Bloomberg US Treasury Index": cmp["ret_bbg"]})
        rf = cmp["ret_tbill"]

        st.subheader("Backtest Period")
        c1, c2, c3 = st.columns(3)
        c1.metric("Start", rets.index[0].strftime("%b %Y"))
        c2.metric("End", rets.index[-1].strftime("%b %Y"))
        c3.metric("Length", f"{len(rets)} months ({len(rets) / 12:.1f} yrs)")
        st.caption(f"Monthly rebalancing, ETF total returns, {CFG['cost_bps']:.0f}bp transaction costs on the strategy "
                   "and the equal-weight benchmark.")

        st.subheader("Growth of $1")
        wealth = (1 + rets).cumprod()
        wealth.index = _ts(wealth.index)
        st.altair_chart(line_chart(wealth, None, fmt="$.2f", zero=False), width="stretch")

        st.subheader("Performance Summary")
        st.dataframe(style_perf_table(pd.DataFrame({c: perf_stats(rets[c], rf) for c in rets}).T), width="stretch")
        rel = pd.DataFrame({f"vs {b}": relative_stats(rets["Strategy"], rets[b])
                            for b in ["Equal weight (25% each)", "Bloomberg US Treasury Index"]}).T
        st.dataframe(rel.style.format("{:.2f}"), width="stretch")
        st.caption(f"Sharpe and Sortino use the {CFG['rf']} yield as the risk-free rate.")

        st.subheader("Drawdown")
        dd = wealth / wealth.cummax() - 1
        st.altair_chart(line_chart(dd, None, fmt=".0%", height=240), width="stretch")

        st.subheader("Asset Weights")
        w = bt[[f"w_{e}" for e in ETFS]].rename(columns=lambda c: c[2:]).assign(TBILL=0.0)
        w.index = _ts(w.index)
        st.altair_chart(weights_chart(w), width="stretch")

        st.subheader("Portfolio Duration")
        d = bt[["dur_strategy", "dur_target", "dur_ew", "dur_bbg"]].rename(columns={
            "dur_strategy": "Strategy", "dur_target": "Target", "dur_ew": "Equal weight (25% each)",
            "dur_bbg": "Bloomberg US Treasury Index"})
        d.index = _ts(d.index)
        colors = {"Strategy": "#3f6bc4", "Target": "#a4c0ec", "Equal weight (25% each)": "#222222",
                  "Bloomberg US Treasury Index": "#e8843a"}
        st.altair_chart(line_chart(d, "Years (OAD)", fmt=".0f", height=280, colors=colors,
                                   interpolate="step-after"), width="stretch")

        st.subheader("Turnover")
        c1, c2 = st.columns(2)
        c1.metric("Two-way turnover / yr", f"{bt['turnover'].sum() / (len(bt) / 12):.0%}")
        c2.metric("Average duration vs EW", f"{(bt['dur_strategy'] - bt['dur_ew']).mean():+.2f} yrs")
