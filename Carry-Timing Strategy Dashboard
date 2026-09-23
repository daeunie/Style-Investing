"""
Carry-Timing Strategy Dashboard
--------------------------------
Two tabs:
  1. Rebalancing (main view) -- for all 5 instruments (SHY, IEI, IEF,
     TLT, and the 3-month T-bill), shows the latest carry_adj, Z-score,
     recommended action (Long / Hold / Sell), and current strategy
     weight.
  2. Backtest -- cumulative growth chart + performance summary (CAGR,
     Vol, Sharpe, Max Drawdown) for the dynamic carry-timing strategy
     vs. a static 25/25/25/25 benchmark.

Data expected in ./data/:
  - long_data_with_signal.csv       (from Step 5 of the backtest pipeline)
  - portfolio_backtest_dynamic.csv  (from Step 6 of the backtest pipeline)
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Carry-Timing Strategy", layout="wide")

CARRY_TESTED = ["IEI", "IEF", "TLT"]
ENTER_Z = 0.5
EXIT_Z = -0.5


@st.cache_data
def load_data():
    signal_df = pd.read_csv("data/long_data_with_signal.csv")
    portfolio_df = pd.read_csv("data/portfolio_backtest_dynamic.csv", index_col="ym")
    return signal_df, portfolio_df


def max_drawdown(cum_series: pd.Series) -> float:
    running_max = cum_series.cummax()
    return ((cum_series - running_max) / running_max).min() * 100


def summarize(portfolio_df: pd.DataFrame) -> pd.DataFrame:
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


def get_action(prior_signal, latest_signal) -> str:
    if prior_signal == "Cash" and latest_signal == "Long":
        return "Long (Buy)"
    elif prior_signal == "Long" and latest_signal == "Cash":
        return "Sell (to Cash)"
    elif latest_signal == "Long":
        return "Hold (Long)"
    else:
        return "Hold (Cash)"


def build_rebalancing_table(signal_df: pd.DataFrame) -> pd.DataFrame:
    """One row per instrument -- SHY, IEI, IEF, TLT, and the 3-month
    T-bill -- showing carry_adj, Z-score, action, and current strategy
    weight.

    - IEI/IEF/TLT: fully carry-tested; action + weight driven by signal.
    - SHY: not carry-tested (it's only in the static benchmark, not the
      dynamic strategy) -- shown for reference with 0% strategy weight.
    - 3-month T-bill: the strategy's actual cash fallback -- gets 100%
      of the strategy weight only when no maturity is currently Long.
    """
    rows = []
    n_long = 0

    for maturity in CARRY_TESTED:
        m_df = signal_df[signal_df["maturity"] == maturity].sort_values("ym")
        latest = m_df.iloc[-1]
        prior = m_df.iloc[-2]
        action = get_action(prior["signal"], latest["signal"])
        is_long = latest["signal"] == "Long"
        if is_long:
            n_long += 1
        rows.append({
            "Instrument": maturity,
            "Date": latest["ym"],
            "Carry_adj": round(latest["carry_adj"], 4) if pd.notna(latest["carry_adj"]) else None,
            "Z-score": round(latest["z"], 3) if pd.notna(latest["z"]) else None,
            "Action": action,
            "_is_long": is_long,
        })

    # Fill in weights now that we know how many are Long
    for row in rows:
        row["Strategy Weight (%)"] = round(100 / n_long, 1) if (row["_is_long"] and n_long > 0) else 0.0
        del row["_is_long"]

    # SHY -- reference only, not carry-tested, never held in the dynamic strategy
    shy_df = signal_df[signal_df["maturity"] == "SHY"].sort_values("ym")
    shy_latest = shy_df.iloc[-1]
    rows.insert(0, {
        "Instrument": "SHY",
        "Date": shy_latest["ym"],
        "Carry_adj": None,
        "Z-score": None,
        "Action": "N/A \u2014 benchmark only",
        "Strategy Weight (%)": 0.0,
    })

    # 3-month T-bill -- the actual cash fallback in the strategy
    tbill_latest_row = signal_df.sort_values("ym").iloc[-1]
    tbill_weight = 100.0 if n_long == 0 else 0.0
    rows.append({
        "Instrument": "3-Month T-Bill",
        "Date": tbill_latest_row["ym"],
        "Carry_adj": None,
        "Z-score": None,
        "Action": "Cash reference (fallback)",
        "Strategy Weight (%)": tbill_weight,
    })

    return pd.DataFrame(rows)


# ---------------- Layout ----------------
st.title("Carry \u2014 US Treasury ETFs")

signal_df, portfolio_df = load_data()

tab1, tab2 = st.tabs(["\u2696\ufe0f Rebalancing", "\U0001F4CA Backtest"])

with tab1:
    st.subheader("Current Rebalancing Weights")
    st.caption(
        f"Hysteresis rule: Long if Z > {ENTER_Z}, Sell to cash if Z < {EXIT_Z}, "
        "otherwise hold the existing position. Weight = equal split of 100% "
        "across maturities currently Long; if none are Long, 100% falls back "
        "to the 3-month T-bill."
    )
    rebalancing_table = build_rebalancing_table(signal_df)
    st.dataframe(rebalancing_table, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Cumulative Growth of $1")
    chart_df = portfolio_df[["cum_benchmark", "cum_strategy"]].rename(
        columns={
            "cum_benchmark": "Benchmark (25/25/25/25, static)",
            "cum_strategy": "Strategy (dynamic weight, timed)",
        }
    )
    st.line_chart(chart_df)

    st.subheader("Performance Summary")
    st.dataframe(summarize(portfolio_df), use_container_width=True, hide_index=True)
    st.caption("*Sharpe = raw return / vol, no risk-free subtraction \u2014 rough comparison only.")
