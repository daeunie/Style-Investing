"""
Momentum tab -- Treasury ETF momentum v2 (long-only, T-bill up to 30%, rolling turnover budget)
------------------------------------------------------------------------------------------------
Signal (month-end T−1, held over T), ported from treasury_momentum_v2_backtest.ipynb (MAIN spec):
    m_i    = sum of last 12 monthly excess returns of a constant-maturity par bond (2 / 5 / 10 / 20y CMT)
    tilt   s_i = clip((m_i − m̄) / σ_x, ±2), re-centred to sum 0   (σ_x: expanding window, past only)
    T-bill w_BIL = 30% × clip(−m̄ / σ_m, 0, 1)
    target w = (1 − w_BIL) × Π_[0,60%](25% + 10% · s)
    budget two-way turnover ≤ 300% per rolling 12 months (250% signal trades + 50% reserve), ≤ 50% a month

Live Signal sub-tab : recomputes the signal from FRED yields (cached 6h; falls back to carry/data/yields_monthly.csv)
                      and applies the turnover budget to the drifted weights stored by the last export.
Historical sub-tab  : momentum/data/momentum_backtest.csv (exported by momentum_export.py).
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from carry.carry_tab import (LABELS, LINE_COLORS, PALETTE, _style, _ts, _x, line_chart, load_yields,
                             perf_stats, relative_stats, style_perf_table, weights_chart)

DATA_DIR = Path(__file__).resolve().parent / "data"
ETFS = ["SHY", "IEI", "IEF", "TLT"]
ASSETS = ETFS + ["BIL"]
YIELD_MAP = {"SHY": "DGS2", "IEI": "DGS5", "IEF": "DGS10", "TLT": "DGS20"}
TENOR = {"SHY": 2, "IEI": 5, "IEF": 10, "TLT": 20}
DATA_START = pd.Period("2005-11", "M")          # same data start as the research notebook (σ is expanding)

_DEFAULT = dict(k=0.10, s_clip=2.0, cash_cap=0.30, w_max=0.60, burn_in=12,
                to_budget=3.00, to_reserve=0.50, to_month=0.50, tc=0.0, lookback=12)
try:
    NEXT = json.loads((DATA_DIR / "momentum_next.json").read_text())
    CFG = {**_DEFAULT, **NEXT.get("config", {})}
except Exception:
    NEXT, CFG = None, _DEFAULT

METHOD_SUMMARY = (
    "Each month, the last 12 months of excess return (over the 3-month T-bill) of a 2 / 5 / 10 / 20-year "
    f"constant-maturity Treasury decide the tilt: ETFs with stronger momentum than the average get up to "
    f"±{CFG['k'] * CFG['s_clip']:.0%} more or less than 25% (capped at {CFG['w_max']:.0%} each). When bonds as a "
    f"whole have negative momentum, up to {CFG['cash_cap']:.0%} moves into T-bills. Trades are limited to "
    f"{CFG['to_budget']:.0%} two-way turnover per rolling 12 months."
)


# ---------------- Signal ----------------
def _par_price(yp, c, m):
    yp = np.maximum(yp, 1e-7)
    disc = (1 + yp / 2) ** (-2 * m)
    return c / yp * (1 - disc) + disc


def _dur_conv(y, m, h=1e-5):
    y = np.asarray(y, dtype=float)
    p0, pu, pd_ = _par_price(y, y, m), _par_price(y + h, y, m), _par_price(y - h, y, m)
    return -(pu - pd_) / (2 * h) / p0, (pu - 2 * p0 + pd_) / h ** 2 / p0


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def compute_signal(monthly: pd.DataFrame) -> dict:
    yld = monthly.loc[DATA_START:] / 100
    cash = yld["DGS3MO"].shift(1) / 12
    syn = {}
    for e, col in YIELD_MAP.items():
        y = yld[col]
        D, C = _dur_conv(y.values, TENOR[e])
        D, C = pd.Series(D, y.index), pd.Series(C, y.index)
        dy = y.diff()
        syn[e] = y.shift(1) / 12 - D.shift(1) * dy + 0.5 * C.shift(1) * dy ** 2
    syn = pd.DataFrame(syn)
    mom = syn.sub(cash, axis=0).rolling(CFG["lookback"], min_periods=CFG["lookback"]).sum().dropna()
    xdev = mom.sub(mom.mean(axis=1), axis=0)
    mbar = mom.mean(axis=1)
    sig_x = xdev.pow(2).mean(axis=1).expanding().mean() ** .5
    sig_m = (mbar ** 2).expanding().mean() ** .5
    s = xdev.div(sig_x, axis=0).clip(-CFG["s_clip"], CFG["s_clip"])
    s = s.sub(s.mean(axis=1), axis=0)
    cash_z = (-mbar / sig_m).clip(0, 1)
    ok = np.arange(len(mom)) >= CFG["burn_in"] - 1
    return {"mom": mom[ok], "s": s[ok], "mbar": mbar[ok], "cash_z": cash_z[ok]}


def _box_simplex(t, hi, total):
    lo_l, hi_l = t.min() - hi.max() - 1, t.max() + 1
    for _ in range(100):
        lam = (lo_l + hi_l) / 2
        if np.clip(t - lam, 0, hi).sum() > total:
            lo_l = lam
        else:
            hi_l = lam
    return np.clip(t - (lo_l + hi_l) / 2, 0, hi)


def target_weights(s: np.ndarray, cash_z: float) -> np.ndarray:
    cash = CFG["cash_cap"] * cash_z
    bonds = _box_simplex(np.full(4, .25) + CFG["k"] * s, np.full(4, CFG["w_max"]), 1.0) * (1 - cash)
    return np.append(bonds, cash)


def apply_budget(w_now: np.ndarray, tgt: np.ndarray, used_11m: float) -> tuple[np.ndarray, bool]:
    sig_left = min(CFG["to_month"], max(0.0, CFG["to_budget"] - CFG["to_reserve"] - used_11m))
    need = np.abs(tgt - w_now).sum()
    a = 1.0 if need <= sig_left + 1e-12 else sig_left / need
    return w_now + a * (tgt - w_now), a < 1


# ---------------- Data ----------------
@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "momentum_backtest.csv", index_col="ym")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


# ---------------- Charts ----------------
def momentum_chart(mom: pd.DataFrame):
    long = (mom * 100).reset_index(names="date").melt("date", var_name="ETF", value_name="m")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=_x(), y=alt.Y("m:Q", title="12m excess return (%)"),
        color=alt.Color("ETF:N", scale=alt.Scale(domain=ETFS, range=[PALETTE[e] for e in ETFS])),
        tooltip=["date:T", "ETF:N", alt.Tooltip("m:Q", format=".2f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9AA3B5").encode(y="y:Q")
    return _style(zero + lines, 280)


def turnover_chart(to12: pd.Series):
    df = (to12 * 100).to_frame("value").rename_axis("date").reset_index()
    line = alt.Chart(df).mark_line(strokeWidth=2, color=PALETTE["IEF"]).encode(
        x=_x(), y=alt.Y("value:Q", title="Rolling 12m two-way turnover (%)"),
        tooltip=["date:T", alt.Tooltip("value:Q", format=".0f")])
    lim = pd.DataFrame({"y": [CFG["to_budget"] * 100, (CFG["to_budget"] - CFG["to_reserve"]) * 100],
                        "label": ["Total limit", "Signal-trade limit"]})
    rules = alt.Chart(lim).mark_rule(strokeDash=[4, 4]).encode(
        y="y:Q", color=alt.Color("label:N", scale=alt.Scale(range=["#b23a3a", "#9AA3B5"]), title=None))
    return _style(rules + line, 260)


# ---------------- Layout ----------------
def render(show_header: bool = True):
    if show_header:
        st.header("Momentum")
        st.caption(METHOD_SUMMARY)

    sub1, sub2 = st.tabs(["This month's signal", "Historical backtest"])

    # ---------- Live signal ----------
    with sub1:
        try:
            monthly, _, source = load_yields()
            sig = compute_signal(monthly)
        except Exception as e:
            st.error("The live signal couldn't be computed. Refresh the page in a few minutes.")
            st.exception(e)
            return

        t = sig["s"].index[-1]
        tgt = target_weights(sig["s"].loc[t, ETFS].values, float(sig["cash_z"].loc[t]))

        in_sync = NEXT is not None and pd.Period(NEXT["month"], "M") == t + 1
        if in_sync:
            prev = np.array([NEXT[f"prev_{a}"] for a in ASSETS])
            w, capped = apply_budget(prev, tgt, NEXT["budget_used_11m"])
            used = NEXT["budget_used_11m"] + np.abs(w - prev).sum()
        else:
            prev, w, capped, used = None, tgt, False, None

        st.subheader("This Month's Recommended Positioning")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positioning for", (t + 1).strftime("%B %Y"))
        c2.metric("Signal as of (month-end)", t.strftime("%b %Y"))
        c3.metric("T-bill weight", f"{w[4]:.0%}")
        c4.metric(f"12m turnover used (limit {CFG['to_budget']:.0%})", f"{used:.0%}" if used is not None else "–")

        rows = []
        for i, e in enumerate(ETFS):
            rows.append({"Instrument": e, "Signal bond": f"{TENOR[e]}y CMT",
                         "12m excess return (%)": f"{sig['mom'].at[t, e] * 100:+.2f}",
                         "Tilt score": f"{sig['s'].at[t, e]:+.2f}",
                         "Target (%)": f"{tgt[i] * 100:.1f}", "Weight (%)": f"{w[i] * 100:.1f}",
                         "Change (pp)": f"{(w[i] - prev[i]) * 100:+.1f}" if prev is not None else "–"})
        rows.append({"Instrument": "3-Month T-Bill", "Signal bond": "Average of 4",
                     "12m excess return (%)": f"{sig['mbar'].at[t] * 100:+.2f}",
                     "Tilt score": f"cash z {sig['cash_z'].at[t]:.2f}",
                     "Target (%)": f"{tgt[4] * 100:.1f}", "Weight (%)": f"{w[4] * 100:.1f}",
                     "Change (pp)": f"{(w[4] - prev[4]) * 100:+.1f}" if prev is not None else "–"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        notes = [f"Yields from {source}."]
        if in_sync:
            notes.append("Weight = target after the turnover budget, starting from last month's weights drifted "
                         "with returns." + (" The budget limited this month's trade." if capped else ""))
        else:
            notes.append("Showing target weights only: the stored backtest ends earlier than the latest signal, "
                         "so re-run momentum_export.py to apply the turnover budget.")
        st.caption(" ".join(notes))

        st.subheader("12-Month Momentum, Last 10 Years")
        m10 = sig["mom"].loc[t - 119:t]
        m10.index = _ts(m10.index)
        st.altair_chart(momentum_chart(m10), width="stretch")
        st.caption("Excess return of a constant-maturity par bond at each ETF's signal tenor, summed over 12 months. "
                   "The spread between lines drives the tilt; the average below zero drives the T-bill weight.")

    # ---------- Historical backtest ----------
    with sub2:
        try:
            bt = load_backtest()
        except FileNotFoundError:
            st.info("Backtest results are missing. Run momentum_export.py and commit momentum/data/.")
            return

        cmp = bt.dropna(subset=["ret_bbg"])
        rets = pd.DataFrame({"Strategy": cmp["ret_strategy"], "Equal weight (25% each)": cmp["ret_ew"],
                             "Bloomberg US Treasury Index": cmp["ret_bbg"]})
        rf = cmp["ret_tbill"]

        st.subheader("Backtest Period")
        c1, c2, c3 = st.columns(3)
        c1.metric("Start", rets.index[0].strftime("%b %Y"))
        c2.metric("End", rets.index[-1].strftime("%b %Y"))
        c3.metric("Length", f"{len(rets)} months ({len(rets) / 12:.1f} yrs)")
        st.caption("Starts once the 12-month momentum and 12 months of its dispersion are available (yield data from "
                   "Nov 2005). Ends with the last month of Bloomberg index data. Monthly rebalancing, ETF total "
                   f"returns, transaction costs {CFG['tc'] * 1e4:.0f}bp.")

        st.subheader("Growth of $1")
        wealth = (1 + rets).cumprod()
        wealth.index = _ts(wealth.index)
        st.altair_chart(line_chart(wealth, None, fmt="$.2f", zero=False), width="stretch")

        st.subheader("Performance Summary")
        st.dataframe(style_perf_table(pd.DataFrame({c: perf_stats(rets[c], rf) for c in rets}).T), width="stretch")
        rel = pd.DataFrame({f"vs {b}": relative_stats(rets["Strategy"], rets[b])
                            for b in ["Equal weight (25% each)", "Bloomberg US Treasury Index"]}).T
        st.dataframe(rel.style.format("{:.2f}"), width="stretch")
        st.caption("Sharpe and Sortino use the 3-month T-bill as the risk-free rate.")

        st.subheader("Drawdown")
        dd = wealth / wealth.cummax() - 1
        st.altair_chart(line_chart(dd, None, fmt=".0%", height=240), width="stretch")

        st.subheader("Asset Weights")
        w_hist = bt[[f"w_{a}" for a in ASSETS]].rename(columns=lambda c: c[2:]).rename(columns={"BIL": "TBILL"})
        w_hist.index = _ts(w_hist.index)
        st.altair_chart(weights_chart(w_hist), width="stretch")

        st.subheader("Portfolio Duration")
        d = bt[["duration", "dur_ew", "dur_bbg"]].rename(columns={
            "duration": "Strategy", "dur_ew": "Equal weight (25% each)", "dur_bbg": "Bloomberg US Treasury Index"})
        d.index = _ts(d.index)
        st.altair_chart(line_chart(d, "Years (OAD)", fmt=".0f", height=280, interpolate="step-after"),
                        width="stretch")
        st.caption("No duration limit in v2; duration moves with the tilt and the T-bill weight (T-bill = 0).")

        st.subheader("Turnover Budget")
        to = bt["turnover"].iloc[1:]
        c1, c2, c3 = st.columns(3)
        c1.metric("Two-way turnover / yr", f"{to.mean() * 12:.0%}")
        c2.metric("Max rolling 12m", f"{to.rolling(12).sum().max():.0%}")
        c3.metric("Months the budget bound", f"{int(bt['capped'].sum())}")
        to12 = to.rolling(12).sum().dropna()
        to12.index = _ts(to12.index)
        st.altair_chart(turnover_chart(to12), width="stretch")
