"""
Carry-Timing Strategy -- Carry tab module
-------------------------------------------
Everything that used to be its own standalone app.py, now wrapped in a
render() function so it can be called from inside a tab of the main
"Style Investing" site instead of running as its own separate app.

No st.set_page_config() or top-level st.title() here -- those belong
to the main entry point (style-investing/app.py) since Streamlit only
allows page config to be set once.
"""

import os

import pandas as pd
import streamlit as st
import yfinance as yf

CARRY_TESTED = ["SHY", "IEI", "IEF", "TLT"]
TTM_WINDOW = 12
ZSCORE_WINDOW = 72
ENTER_Z = 1.0
EXIT_Z = -1.0

FRED_TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

# Absolute path to this module's own data/ folder, so it works
# regardless of which directory Streamlit's main script runs from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKTEST_CSV = os.path.join(_THIS_DIR, "data", "portfolio_backtest_dynamic.csv")


# ==================== Live Signal ====================

@st.cache_data(ttl=3600 * 6)
def fetch_etf_monthly(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    hist = t.history(period="5y", interval="1d", auto_adjust=False)
    hist = hist.reset_index()
    hist["ym"] = hist["Date"].dt.to_period("M")

    monthly_price = hist.groupby("ym")["Close"].last()
    monthly_div = hist.groupby("ym")["Dividends"].sum()

    df = pd.DataFrame({"nav": monthly_price, "distribution": monthly_div}).reset_index()
    df["maturity"] = ticker
    return df


@st.cache_data(ttl=3600 * 6)
def fetch_tbill_monthly() -> pd.DataFrame:
    raw = pd.read_csv(FRED_TBILL_URL)
    raw.columns = ["date", "tbill"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["tbill"] = pd.to_numeric(raw["tbill"], errors="coerce")
    raw["ym"] = raw["date"].dt.to_period("M")
    return raw.dropna(subset=["tbill"]).groupby("ym")["tbill"].last().reset_index()


@st.cache_data(ttl=3600 * 6)
def build_live_dataset() -> pd.DataFrame:
    frames = [fetch_etf_monthly(t) for t in CARRY_TESTED]
    long_df = pd.concat(frames, ignore_index=True)
    tbill_monthly = fetch_tbill_monthly()
    long_df = long_df.merge(tbill_monthly, on="ym", how="left")
    return long_df.sort_values(["maturity", "ym"]).reset_index(drop=True)


@st.cache_data(ttl=3600 * 6)
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """TTM yield -> carry_adj -> Z-score -> signal. Uses float('nan'),
    not pd.NA, since there's no CSV round-trip here to silently
    convert it (unlike the notebook, which saves/reloads between
    every step)."""

    df = df.sort_values(["maturity", "ym"]).reset_index(drop=True)

    trailing_dist = df.groupby("maturity")["distribution"].transform(
        lambda x: x.rolling(window=TTM_WINDOW, min_periods=TTM_WINDOW).sum()
    )
    df["ttm_yield"] = (trailing_dist / df["nav"]) * 100

    def compute_carry(row):
        if pd.isna(row["ttm_yield"]) or pd.isna(row["tbill"]):
            return float("nan")
        return row["ttm_yield"] - row["tbill"]

    df["carry_adj"] = df.apply(compute_carry, axis=1)

    shifted = df.groupby("maturity")["carry_adj"].shift(1)
    df["carry_mean"] = shifted.groupby(df["maturity"]).transform(
        lambda x: x.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
    )
    df["carry_std"] = shifted.groupby(df["maturity"]).transform(
        lambda x: x.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std()
    )
    df["z"] = (df["carry_adj"] - df["carry_mean"]) / df["carry_std"]

    df["signal"] = pd.NA
    for maturity in CARRY_TESTED:
        mask = df["maturity"] == maturity
        idx = df[mask].index
        position = "Cash"
        signals = []
        for i in idx:
            z = df.loc[i, "z"]
            if pd.notna(z):
                if z > ENTER_Z:
                    position = "Long"
                elif z < EXIT_Z:
                    position = "Cash"
            signals.append(position)
        df.loc[idx, "signal"] = signals

    return df


def get_action(prior_signal, latest_signal) -> str:
    if prior_signal == "Cash" and latest_signal == "Long":
        return "Long (Buy)"
    elif prior_signal == "Long" and latest_signal == "Cash":
        return "Sell (to Cash)"
    elif latest_signal == "Long":
        return "Hold (Long)"
    else:
        return "Hold (Cash)"


def build_live_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_long = 0
    for maturity in CARRY_TESTED:
        m_df = df[df["maturity"] == maturity].sort_values("ym")
        latest, prior = m_df.iloc[-1], m_df.iloc[-2]
        action = get_action(prior["signal"], latest["signal"])
        is_long = latest["signal"] == "Long"
        if is_long:
            n_long += 1
        rows.append({
            "Instrument": maturity,
            "As of": str(latest["ym"]),
            "Carry_adj": round(latest["carry_adj"], 4) if pd.notna(latest["carry_adj"]) else None,
            "Z-score": round(latest["z"], 3) if pd.notna(latest["z"]) else None,
            "Action": action,
            "_is_long": is_long,
        })
    for row in rows:
        row["Weight (%)"] = round(100 / n_long, 1) if (row["_is_long"] and n_long > 0) else 0.0
        del row["_is_long"]

    tbill_latest = df.sort_values("ym").iloc[-1]
    rows.append({
        "Instrument": "3-Month T-Bill",
        "As of": str(tbill_latest["ym"]),
        "Carry_adj": None,
        "Z-score": None,
        "Action": "Cash reference (fallback)",
        "Weight (%)": 100.0 if n_long == 0 else 0.0,
    })
    return pd.DataFrame(rows)


def build_recent_history(df: pd.DataFrame, n_months: int = 6):
    """Trailing table: one row per month, one column per instrument,
    showing what the strategy was actually holding and at what weight.
    Also returns a parallel boolean frame marking which cells changed
    from the prior month, so a flip can be visually highlighted."""

    wide_signal = df.pivot(index="ym", columns="maturity", values="signal")
    full_index = wide_signal.index
    recent_months = full_index[-n_months:]

    display_rows = []
    changed_rows = []
    for ym in recent_months:
        idx_pos = full_index.get_loc(ym)
        longs = [m for m in CARRY_TESTED if wide_signal.loc[ym, m] == "Long"]
        n_long = len(longs)
        was_tbill_active = False
        if idx_pos > 0:
            prior_ym = full_index[idx_pos - 1]
            prior_longs = [m for m in CARRY_TESTED if wide_signal.loc[prior_ym, m] == "Long"]
            was_tbill_active = len(prior_longs) == 0

        row = {"Month": str(ym)}
        changed = {"Month": False}
        for m in CARRY_TESTED:
            sig = wide_signal.loc[ym, m]
            prior_sig = wide_signal.loc[full_index[idx_pos - 1], m] if idx_pos > 0 else None
            if sig == "Long":
                weight = round(100 / n_long, 1) if n_long > 0 else 0.0
                row[m] = f"Long ({weight}%)"
            else:
                row[m] = "Cash (0%)"
            changed[m] = (prior_sig is not None) and (sig != prior_sig)

        is_tbill_active = n_long == 0
        row["3-Month T-Bill"] = f"{100.0 if is_tbill_active else 0.0}%"
        changed["3-Month T-Bill"] = (idx_pos > 0) and (is_tbill_active != was_tbill_active)

        display_rows.append(row)
        changed_rows.append(changed)

    display_df = pd.DataFrame(display_rows).iloc[::-1].reset_index(drop=True)
    changed_df = pd.DataFrame(changed_rows).iloc[::-1].reset_index(drop=True)
    return display_df, changed_df


def highlight_changes(display_df: pd.DataFrame, changed_df: pd.DataFrame):
    """Return a pandas Styler that highlights any cell marked True in
    changed_df with a yellow background and bold text."""

    def style_func(_):
        styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
        for col in changed_df.columns:
            styles[col] = changed_df[col].map(
                lambda changed: "background-color: #FFF3B0; font-weight: bold;" if changed else ""
            )
        return styles

    return display_df.style.apply(style_func, axis=None)


# ==================== Historical Backtest (static, pre-computed) ====================

def max_drawdown(cum_series: pd.Series) -> float:
    running_max = cum_series.cummax()
    return ((cum_series - running_max) / running_max).min() * 100


def summarize_backtest(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    n_months = len(portfolio_df)
    rows = []
    for label, ret_col, cum_col in [
        ("Benchmark (25/25/25/25, static)", "benchmark_return", "cum_benchmark"),
        ("Strategy (dynamic weight, timed)", "strategy_return", "cum_strategy"),
    ]:
        cagr = portfolio_df[cum_col].iloc[-1] ** (12 / n_months) - 1
        vol = portfolio_df[ret_col].std() / 100 * (12 ** 0.5)
        sharpe = cagr / vol if vol else float("nan")
        mdd = max_drawdown(portfolio_df[cum_col])
        rows.append({
            "Portfolio": label,
            "CAGR (%)": round(cagr * 100, 2),
            "Vol (%, ann.)": round(vol * 100, 2),
            "Sharpe*": round(sharpe, 2),
            "Max Drawdown (%)": round(mdd, 2),
        })
    return pd.DataFrame(rows)


# ==================== Public entry point ====================

def render():
    """Call this from inside a st.tabs() block in the main app to
    render the entire Carry section (its own two sub-tabs)."""

    st.header("Carry \u2014 US Treasury ETFs")

    subtab1, subtab2 = st.tabs(["\U0001F534 Live Signal (This Month)", "\U0001F4CA Historical Backtest"])

    with subtab1:
        st.subheader("This Month's Recommended Positioning")
        st.caption(
            f"Fetched live from Yahoo Finance and FRED (DGS3MO) \u2014 recomputed on every "
            f"page load (cached 6h). Hysteresis rule: Long if Z > {ENTER_Z}, "
            f"Sell to cash if Z < {EXIT_Z}, otherwise hold. All 4 ETFs are carry-tested; "
            "if none are Long, 100% falls back to the 3-month T-bill."
        )
        with st.spinner("Fetching live data and computing this month's signal..."):
            try:
                live_df = build_live_dataset()
                signaled_df = compute_signals(live_df)
                live_table = build_live_table(signaled_df)
                st.dataframe(live_table, use_container_width=True, hide_index=True)
                st.success(f"Live data as of {live_table['As of'].iloc[0]}")

                st.subheader("Recent History \u2014 What Were We Meant to Be Carrying")
                st.caption(
                    "Last 6 months' positioning, most recent first \u2014 highlighted cells "
                    "mark a change from the prior month (a flip in/out of that position)."
                )
                recent_display, recent_changed = build_recent_history(signaled_df, n_months=6)
                st.dataframe(
                    highlight_changes(recent_display, recent_changed),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as e:
                st.error("Couldn't fetch live data right now. Try refreshing in a few minutes.")
                st.exception(e)

    with subtab2:
        st.subheader("Cumulative Growth of $1")
        st.caption(
            "Results from the full historical backtest (notebook pipeline, 2007\u20132026, "
            "using iShares' official monthly NAV returns). This tab shows fixed results, "
            "not live data \u2014 re-run the notebook and re-upload the CSV to update it."
        )
        try:
            portfolio_df = pd.read_csv(_BACKTEST_CSV, index_col="ym")
            chart_df = portfolio_df[["cum_benchmark", "cum_strategy"]].rename(columns={
                "cum_benchmark": "Benchmark (25/25/25/25, static)",
                "cum_strategy": "Strategy (dynamic weight, timed)",
            })
            st.line_chart(chart_df)

            st.subheader("Performance Summary")
            st.dataframe(summarize_backtest(portfolio_df), use_container_width=True, hide_index=True)
            st.caption(
                "*Sharpe = raw return / vol, no risk-free subtraction \u2014 rough comparison only. "
                f"Backtest window: {portfolio_df.index.min()} to {portfolio_df.index.max()} "
                f"({len(portfolio_df)} months)."
            )
        except FileNotFoundError:
            st.info(
                "Historical backtest data not found. Upload carry/data/portfolio_backtest_dynamic.csv "
                "(from Step 6 of the notebook pipeline) to enable this tab."
            )
