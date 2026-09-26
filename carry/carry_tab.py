"""
Carry tab -- carry timing on US Treasury ETFs (long-only, T-bill fallback)
---------------------------------------------------------------------------
Signal
    carry_i  = matched Treasury yield (DGS2/5/10/20) - 3M T-bill (DGS3MO), month-end
    z_i      = (carry_i - rolling mean) / rolling std over the last WINDOW months
    Long if z > ENTER_Z, sell to T-bill if z < EXIT_Z, otherwise hold (hysteresis)
    Held ETFs equal-weighted; none held -> 100% 3M T-bill
Timing
    Signal at month-end t -> positioning held over month t+1

Live Signal sub-tab : yields fetched from FRED on page load (cached 6h);
                      falls back to data/yields_monthly.csv if FRED is unreachable.
Historical sub-tab  : reads data/carry_backtest.csv (exported by step 7 of the notebook).
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Settings ----------------
DATA_DIR = Path(__file__).resolve().parent / "data"      # works regardless of working directory
ETFS = ["SHY", "IEI", "IEF", "TLT"]
YIELD_MAP = {"SHY": "DGS2", "IEI": "DGS5", "IEF": "DGS10", "TLT": "DGS20"}
TBILL = "DGS3MO"
WINDOW = 72            # z-score look-back (months)
ENTER_Z = 1.0          # long above
EXIT_Z = -1.0          # sell below
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

PALETTE = {"TBILL": "#b5b2ab", "SHY": "#a4c0ec", "IEI": "#6b95dc", "IEF": "#3f6bc4", "TLT": "#22397a"}
LABELS = {"TBILL": "T-bill (cash)", "SHY": "SHY", "IEI": "IEI", "IEF": "IEF", "TLT": "TLT"}
LINE_COLORS = {"Strategy": "#3f6bc4", "Equal weight (25% each)": "#222222",
               "Bloomberg US Treasury Index": "#e8843a"}


# ---------------- Data ----------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fred_yields() -> pd.DataFrame:
    """Month-end yields from FRED (no API key needed)."""
    out = {}
    for sid in list(YIELD_MAP.values()) + [TBILL]:
        s = pd.read_csv(FRED_CSV.format(sid), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        out[sid] = s.astype(float).dropna().resample("ME").last()
    df = pd.DataFrame(out)
    df.index = df.index.to_period("M")
    return df


def load_yields() -> tuple[pd.DataFrame, str]:
    try:
        df, source = fetch_fred_yields(), "FRED (live)"
    except Exception:
        df = pd.read_csv(DATA_DIR / "yields_monthly.csv", index_col="ym")
        df.index = pd.PeriodIndex(df.index, freq="M")
        source = "stored file (FRED unreachable)"
    current = pd.Timestamp.today().to_period("M")
    return df.loc[df.index < current], source          # use completed months only


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "carry_backtest.csv", index_col="ym")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


# ---------------- Signal ----------------
def compute_signal(ylds: pd.DataFrame) -> dict:
    carry = pd.DataFrame({etf: ylds[y] - ylds[TBILL] for etf, y in YIELD_MAP.items()})
    mu = carry.rolling(WINDOW, min_periods=WINDOW).mean()
    sd = carry.rolling(WINDOW, min_periods=WINDOW).std()
    z = (carry - mu) / sd

    pos = pd.DataFrame(0, index=z.index, columns=ETFS)
    for etf in ETFS:
        state = 0
        for t, v in z[etf].items():
            if np.isnan(v):
                state = 0
            elif v > ENTER_Z:
                state = 1
            elif v < EXIT_Z:
                state = 0
            pos.at[t, etf] = state
    return {"carry": carry, "z": z, "pos": pos}


def build_live_table(sig: dict, ylds: pd.DataFrame) -> pd.DataFrame:
    pos, z, carry = sig["pos"], sig["z"], sig["carry"]
    t = pos.index[-1]
    now, prev = pos.loc[t], pos.iloc[-2]
    n_long = int(now.sum())
    rows = []
    for etf in ETFS:
        if now[etf] and not prev[etf]:
            action = "Buy (Long)"
        elif now[etf]:
            action = "Hold (Long)"
        elif prev[etf]:
            action = "Sell (to Cash)"
        else:
            action = "Stay out (Cash)"
        rows.append({
            "Instrument": etf,
            "Yield used": YIELD_MAP[etf],
            "Carry (%p)": round(carry.at[t, etf], 3),
            "Z-score": round(z.at[t, etf], 2),
            "Action": action,
            "Weight (%)": round(100 / n_long, 1) if (now[etf] and n_long) else 0.0,
        })
    rows.append({
        "Instrument": "3-Month T-Bill",
        "Yield used": f"{TBILL} = {ylds.at[t, TBILL]:.2f}%",
        "Carry (%p)": None,
        "Z-score": None,
        "Action": "Fallback (cash)" if n_long == 0 else "Not used",
        "Weight (%)": 100.0 if n_long == 0 else 0.0,
    })
    return pd.DataFrame(rows)


def build_recent_history(pos: pd.DataFrame, n_months: int = 6):
    """Last n months of positioning (most recent first) + a mask of cells that changed vs prior month.
    Row label = month the position is HELD (signal month-end + 1)."""
    recent = pos.iloc[-(n_months + 1):]
    rows, changed = [], []
    for i in range(1, len(recent)):
        now, prev = recent.iloc[i], recent.iloc[i - 1]
        n_now, n_prev = int(now.sum()), int(prev.sum())
        row = {"Held in": str(recent.index[i] + 1)}
        chg = {"Held in": False}
        for etf in ETFS:
            row[etf] = f"Long ({100 / n_now:.1f}%)" if now[etf] else "Cash (0%)"
            chg[etf] = bool(now[etf] != prev[etf])
        row["3-Month T-Bill"] = "100.0%" if n_now == 0 else "0.0%"
        chg["3-Month T-Bill"] = (n_now == 0) != (n_prev == 0)
        rows.append(row)
        changed.append(chg)
    return pd.DataFrame(rows[::-1]), pd.DataFrame(changed[::-1])


def highlight_changes(display_df: pd.DataFrame, changed_df: pd.DataFrame):
    def style_func(_):
        styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
        for col in changed_df.columns:
            styles[col] = changed_df[col].map(
                lambda c: "background-color: #FFF3B0; font-weight: bold;" if c else "")
        return styles
    return display_df.style.apply(style_func, axis=None)


# ---------------- Stats ----------------
def perf_stats(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0)
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (12 / len(r)) - 1
    ex = r - rf
    dd = wealth / wealth.cummax() - 1
    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Vol (%)": round(r.std() * np.sqrt(12) * 100, 2),
        "Sharpe": round(ex.mean() / ex.std() * np.sqrt(12), 2),
        "Sortino": round(ex.mean() / ex[ex < 0].std() * np.sqrt(12), 2),
        "Max Drawdown (%)": round(dd.min() * 100, 2),
        "Calmar": round(cagr / abs(dd.min()), 2),
    }


def relative_stats(r: pd.Series, b: pd.Series) -> dict:
    act = r - b
    te = act.std() * np.sqrt(12)
    up, dn = b > 0, b < 0
    return {
        "Excess return (%/yr)": round(act.mean() * 1200, 2),
        "Tracking error (%)": round(te * 100, 2),
        "Information ratio": round(act.mean() * 12 / te, 2),
        "t-stat": round(act.mean() / act.std() * np.sqrt(len(act)), 2),
        "Beta": round(np.cov(r, b)[0, 1] / b.var(), 2),
        "Up capture (%)": round(r[up].mean() / b[up].mean() * 100, 1),
        "Down capture (%)": round(r[dn].mean() / b[dn].mean() * 100, 1),
    }


# ---------------- Charts ----------------
def _ts(idx: pd.PeriodIndex) -> pd.DatetimeIndex:
    return idx.to_timestamp()


def line_chart(df: pd.DataFrame, y_title: str, fmt: str = ".2f", colors: dict | None = None):
    long = df.reset_index(names="date").melt("date", var_name="series", value_name="value")
    colors = colors or LINE_COLORS
    return (alt.Chart(long).mark_line(strokeWidth=1.6)
            .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt)),
                    color=alt.Color("series:N", title=None,
                                    scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                                    legend=alt.Legend(orient="top")),
                    tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=fmt)])
            .properties(height=320))


def weights_chart(w: pd.DataFrame):
    order = ETFS + ["TBILL"]                                  # SHY bottom -> TLT -> T-bill on top
    long = (w[order].rename(columns=LABELS).reset_index(names="date")
            .melt("date", var_name="asset", value_name="weight"))
    long["order"] = long["asset"].map({LABELS[a]: i for i, a in enumerate(order)})
    legend_order = [LABELS[a] for a in ["TBILL"] + ETFS]
    return (alt.Chart(long).mark_area(interpolate="step-after")
            .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("weight:Q", stack="zero", title=None,
                            axis=alt.Axis(format=".0%", tickCount=10), scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("asset:N", title=None,
                                    scale=alt.Scale(domain=legend_order,
                                                    range=[PALETTE[a] for a in ["TBILL"] + ETFS]),
                                    legend=alt.Legend(orient="top")),
                    order=alt.Order("order:Q"),
                    tooltip=["date:T", "asset:N", alt.Tooltip("weight:Q", format=".1%")])
            .properties(height=340))


def zscore_chart(z: pd.DataFrame, pos: pd.DataFrame):
    long = z.reset_index(names="date").melt("date", var_name="ETF", value_name="z")
    base = alt.Chart(long).mark_line(strokeWidth=1.3).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("z:Q", title="z-score"),
        color=alt.Color("ETF:N", scale=alt.Scale(domain=ETFS, range=[PALETTE[e] for e in ETFS]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["date:T", "ETF:N", alt.Tooltip("z:Q", format=".2f")])
    rules = alt.Chart(pd.DataFrame({"y": [ENTER_Z, EXIT_Z]})).mark_rule(
        strokeDash=[5, 4], color="grey").encode(y="y:Q")
    return (base + rules).properties(height=300)


# ---------------- Layout ----------------
def render():
    st.header("Carry \u2014 US Treasury ETFs")

    subtab1, subtab2 = st.tabs(["\U0001F534 Live Signal (This Month)", "\U0001F4CA Historical Backtest"])

    # ---------- Live signal ----------
    with subtab1:
        st.subheader("This Month's Recommended Positioning")
        st.caption(
            f"Carry = matched Treasury yield (DGS2 / DGS5 / DGS10 / DGS20) \u2212 3M T-bill (DGS3MO). "
            f"Z-score over the last {WINDOW} months. Hysteresis rule: Long if Z > {ENTER_Z}, "
            f"Sell to cash if Z < {EXIT_Z}, otherwise hold. Held ETFs are equal-weighted; "
            "if none are Long, 100% goes to the 3-month T-bill. "
            "Signal uses the last completed month-end and applies to the current month."
        )
        with st.spinner("Fetching yields and computing this month's signal..."):
            try:
                ylds, source = load_yields()
                sig = compute_signal(ylds)
                table = build_live_table(sig, ylds)
                t = sig["pos"].index[-1]
                n_long = int(sig["pos"].loc[t].sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("Positioning for", str(t + 1))
                c2.metric("Signal as of (month-end)", str(t))
                c3.metric("ETFs held", f"{n_long} / 4", "100% T-bill" if n_long == 0 else None,
                          delta_color="off")

                st.dataframe(table, width="stretch", hide_index=True)
                st.caption(f"Yield data source: {source}.")

                st.subheader("Recent History \u2014 What Were We Meant to Be Carrying")
                st.caption("Last 6 months' positioning, most recent first \u2014 highlighted cells "
                           "mark a change from the prior month (a flip in/out of that position).")
                recent_display, recent_changed = build_recent_history(sig["pos"], n_months=6)
                st.dataframe(highlight_changes(recent_display, recent_changed),
                             width="stretch", hide_index=True)

                st.subheader(f"Carry z-score ({WINDOW}m) \u2014 last 10 years")
                z_recent = sig["z"].loc[t - 119:t]
                z_recent.index = _ts(z_recent.index)
                st.altair_chart(zscore_chart(z_recent, sig["pos"]), width="stretch")
                st.caption(f"Dashed lines = entry (+{ENTER_Z}) and exit ({EXIT_Z}) thresholds.")
            except Exception as e:
                st.error("Couldn't compute the live signal. Try refreshing in a few minutes.")
                st.exception(e)

    # ---------- Historical backtest ----------
    with subtab2:
        try:
            bt = load_backtest()
        except FileNotFoundError:
            st.info("carry/data/carry_backtest.csv not found \u2014 run step 7 of the notebook and commit it.")
            return

        cmp = bt.dropna(subset=["ret_bbg"])                   # common period with the Bloomberg index
        rets = pd.DataFrame({
            "Strategy": cmp["ret_strategy"],
            "Equal weight (25% each)": cmp["ret_ew"],
            "Bloomberg US Treasury Index": cmp["ret_bbg"],
        })
        rf = cmp["ret_tbill"]

        st.subheader("Cumulative Growth of $1")
        st.caption(
            f"Strategy: {WINDOW}m z-score, +{ENTER_Z} / {EXIT_Z} thresholds, monthly rebalance, "
            f"no transaction costs. ETF total returns (NAV + distributions). "
            f"Period: {rets.index[0]} \u2013 {rets.index[-1]} ({len(rets)} months)."
        )
        wealth = (1 + rets).cumprod()
        wealth.index = _ts(wealth.index)
        st.altair_chart(line_chart(wealth, "Growth of $1"), width="stretch")

        st.subheader("Performance Summary")
        st.dataframe(pd.DataFrame({c: perf_stats(rets[c], rf) for c in rets}).T,
                     width="stretch")
        st.caption("Sharpe / Sortino use the 3M T-bill as the risk-free rate.")

        st.subheader("Relative to Benchmarks")
        st.dataframe(pd.DataFrame({
            f"vs {b}": relative_stats(rets["Strategy"], rets[b])
            for b in ["Equal weight (25% each)", "Bloomberg US Treasury Index"]
        }).T, width="stretch")

        st.subheader("Drawdown")
        dd = wealth / wealth.cummax() - 1
        st.altair_chart(line_chart(dd, "Drawdown", fmt=".0%"), width="stretch")

        st.subheader("Asset Weights")
        w = bt[[f"w_{a}" for a in ETFS + ["TBILL"]]].rename(columns=lambda c: c[2:])
        w.index = _ts(w.index)
        st.altair_chart(weights_chart(w), width="stretch")

        st.subheader("Portfolio Duration (OAD)")
        d = bt[["dur_strategy", "dur_ew", "dur_bbg"]].rename(columns={
            "dur_strategy": "Strategy", "dur_ew": "Equal weight (25% each)",
            "dur_bbg": "Bloomberg US Treasury Index"})
        d.index = _ts(d.index)
        st.altair_chart(line_chart(d, "Years", fmt=".1f"), width="stretch")
        st.caption("Weighted Bloomberg index OAD of each held ETF's index; T-bill = 0.25 years.")

        st.subheader("Turnover & Position Flips")
        pos = (bt[[f"w_{e}" for e in ETFS]] > 0).astype(int)
        pos.columns = ETFS
        chg = pos.diff().fillna(0)
        n_years = len(bt) / 12
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual turnover (one-way)", f"{bt['turnover'].mean() * 12:.0%}")
        c2.metric("Months with a trade", f"{(bt['turnover'] > 0.01).mean():.0%}")
        c3.metric("Total flips", f"{int((chg != 0).sum().sum())}")
        st.dataframe(pd.DataFrame({
            "Buys": (chg == 1).sum(),
            "Sells": (chg == -1).sum(),
            "Flips / yr": ((chg != 0).sum() / n_years).round(2),
            "% months held": (pos.mean() * 100).round(1),
        }), width="stretch")
