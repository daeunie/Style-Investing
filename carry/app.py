"""
Carry-Timing Strategy Dashboard -- LIVE VERSION
-------------------------------------------------
Unlike the earlier version, this one does NOT read a pre-baked CSV for
the current signal. Every time the app runs (subject to the cache TTL
below), it:
  1. Fetches live price + dividend history for SHY, IEI, IEF, TLT via
     yfinance.
  2. Fetches the live 3-month T-bill rate (FRED, DGS3MO) via FRED's
     public CSV endpoint -- no API key needed.
  3. Recomputes TTM distribution yield -> carry_adj -> rolling 24-month
     Z-score -> hysteresis signal, exactly as the backtest notebook
     does, but on freshly fetched data.
  4. Shows TODAY's actual recommended action (Long / Hold / Sell) per
     ETF, plus the resulting portfolio weights.

The historical Backtest tab still reads the pre-computed CSVs from the
notebook pipeline (data/long_data_with_signal.csv,
data/portfolio_backtest_dynamic.csv) -- that part is legitimately
historical and doesn't need to be live.

Needs ~40+ months of history fetched (12 months for the TTM yield
window + 24 months for the Z-score window) before a valid signal can
be computed -- this is handled automatically by fetching 5 years.
"""

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Carry-Timing Strategy", layout="wide")

CARRY_TESTED = ["SHY", "IEI", "IEF", "TLT"]
DURATIONS = {"SHY": 1.9, "IEI": 4.5, "IEF": 7.5, "TLT": 17.0}
TTM_WINDOW = 12
ZSCORE_WINDOW = 24
ENTER_Z = 0.5
EXIT_Z = -0.5
FETCH_YEARS = 5  # enough history for TTM (12mo) + Z-score (24mo) windows

FRED_TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"


@st.cache_data(ttl=3600 * 6)  # refresh every 6 hours -- avoids re-fetching on every click
def fetch_etf_monthly(ticker: str) -> pd.DataFrame:
    """Fetch live daily price + dividend history for one ticker,
    collapse to monthly NAV (month-end close) and monthly total
    dividends."""
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{FETCH_YEARS}y", interval="1d", auto_adjust=False)
    hist = hist.reset_index()
    hist["ym"] = hist["Date"].dt.to_period("M")

    monthly_price = hist.groupby("ym")["Close"].last()
    monthly_div = hist.groupby("ym")["Dividends"].sum()

    df = pd.DataFrame({"nav": monthly_price, "distribution": monthly_div})
    df = df.reset_index()
    df["maturity"] = ticker
    return df


@st.cache_data(ttl=3600 * 6)
def fetch_tbill_monthly() -> pd.DataFrame:
    """Fetch the live 3-month T-bill yield series from FRED's public
    CSV endpoint (no API key required), collapse to monthly last
    value."""
    raw = pd.read_csv(FRED_TBILL_URL)
    raw.columns = ["date", "tbill"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["tbill"] = pd.to_numeric(raw["tbill"], errors="coerce")  # FRED uses "." for missing
    raw["ym"] = raw["date"].dt.to_period("M")
    monthly = raw.dropna(subset=["tbill"]).groupby("ym")["tbill"].last().reset_index()
    return monthly


@st.cache_data(ttl=3600 * 6)
def build_live_dataset() -> pd.DataFrame:
    """Fetch everything live and assemble one long DataFrame, same
    shape as the notebook's long_data.csv."""
    frames = [fetch_etf_monthly(t) for t in CARRY_TESTED]
    long_df = pd.concat(frames, ignore_index=True)

    tbill_monthly = fetch_tbill_monthly()
    long_df = long_df.merge(tbill_monthly, on="ym", how="left")
    long_df = long_df.sort_values(["maturity", "ym"]).reset_index(drop=True)
    return long_df


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full carry -> TTM yield -> carry_adj -> Z-score -> signal
    pipeline on live-fetched data. Same formulas as the backtest
    notebook (Steps 2-5)."""

    df = df.sort_values(["maturity", "ym"]).reset_index(drop=True)

    # Step 2: TTM distribution yield
    trailing_dist = df.groupby("maturity")["distribution"].transform(
        lambda x: x.rolling(window=TTM_WINDOW, min_periods=TTM_WINDOW).sum()
    )
    df["ttm_yield"] = (trailing_dist / df["nav"]) * 100

    # Step 3: carry_adj
    def compute_carry(row):
        if pd.isna(row["ttm_yield"]) or pd.isna(row["tbill"]):
            return float("nan")
        return (row["ttm_yield"] - row["tbill"]) / DURATIONS[row["maturity"]]

    df["carry_adj"] = df.apply(compute_carry, axis=1)

    # Step 4: rolling Z-score (shift(1) avoids lookahead)
    shifted = df.groupby("maturity")["carry_adj"].shift(1)
    df["carry_mean"] = shifted.groupby(df["maturity"]).transform(
        lambda x: x.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
    )
    df["carry_std"] = shifted.groupby(df["maturity"]).transform(
        lambda x: x.rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std()
    )
    df["z"] = (df["carry_adj"] - df["carry_mean"]) / df["carry_std"]

    # Step 5: hysteresis signal
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
        latest = m_df.iloc[-1]
        prior = m_df.iloc[-2]
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
    tbill_weight = 100.0 if n_long == 0 else 0.0
    rows.append({
        "Instrument": "3-Month T-Bill",
        "As of": str(tbill_latest["ym"]),
        "Carry_adj": None,
        "Z-score": None,
        "Action": "Cash reference (fallback)",
        "Weight (%)": tbill_weight,
    })

    return pd.DataFrame(rows)


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


# ---------------- Layout ----------------
st.title("Carry \u2014 US Treasury ETFs")

tab1, tab2 = st.tabs(["\U0001F534 Live Signal (Today)", "\U0001F4CA Historical Backtest"])

with tab1:
    st.subheader("Today's Recommended Positioning")
    st.caption(
        f"Fetched live from Yahoo Finance and FRED (DGS3MO) \u2014 recomputed on every "
        f"page load (cached {6}h). Hysteresis rule: Long if Z > {ENTER_Z}, "
        f"Sell to cash if Z < {EXIT_Z}, otherwise hold. All 4 ETFs are carry-tested; "
        "if none are Long, 100% falls back to the 3-month T-bill."
    )

    with st.spinner("Fetching live data and computing today's signal..."):
        try:
            live_df = build_live_dataset()
            live_df = compute_signals(live_df)
            live_table = build_live_table(live_df)
            st.dataframe(live_table, use_container_width=True, hide_index=True)
            st.success(f"Live data as of {live_table['As of'].iloc[0]}")
        except Exception as e:
            st.error(
                "Couldn't fetch live data right now. This can happen if Yahoo Finance "
                "or FRED is temporarily unavailable. Try refreshing in a few minutes."
            )
            st.exception(e)

with tab2:
    st.subheader("Cumulative Growth of $1 (Historical Backtest)")
    st.caption(
        "This tab shows how the strategy performed historically \u2014 it reads the "
        "pre-computed backtest results, not live data."
    )
    try:
        portfolio_df = pd.read_csv("data/portfolio_backtest_dynamic.csv", index_col="ym")
        chart_df = portfolio_df[["cum_benchmark", "cum_strategy"]].rename(
            columns={
                "cum_benchmark": "Benchmark (25/25/25/25, static)",
                "cum_strategy": "Strategy (dynamic weight, timed)",
            }
        )
        st.line_chart(chart_df)

        st.subheader("Performance Summary")
        st.dataframe(summarize_backtest(portfolio_df), use_container_width=True, hide_index=True)
        st.caption("*Sharpe = raw return / vol, no risk-free subtraction \u2014 rough comparison only.")
    except FileNotFoundError:
        st.info("Historical backtest data not found in data/ -- upload portfolio_backtest_dynamic.csv to enable this tab.")
