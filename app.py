"""
Style Investing -- main entry point
--------------------------------------
Point Streamlit Cloud's "Main file path" at this file. It sets the page
config, applies the shared site style, and delegates each top-level tab
to that factor's own render() function.

Repo layout:
    app.py                     <- this file
    site_style.py              <- shared CSS + page header
    .streamlit/config.toml     <- theme colours
    carry/carry_tab.py         <- Carry factor
    carry/data/*.csv
    momentum/momentum_tab.py   <- Momentum factor
    value/value_tab.py         <- Value factor
"""

import streamlit as st

st.set_page_config(page_title="Style Investing", page_icon="\U0001F4C8", layout="wide")

from site_style import apply_style, hero  # noqa: E402
from carry.carry_tab import render as render_carry  # noqa: E402
from momentum.momentum_tab import render as render_momentum  # noqa: E402
from value.value_tab import render as render_value  # noqa: E402

apply_style()
hero(
    "Style Investing",
    "AQR-style factor investing \u2014 carry, momentum and value \u2014 applied to US Treasury "
    "ETFs, adapted from Brooks, Palhares &amp; Richardson (2018). KSIF Strategic Asset Allocation Team.",
)

tab_carry, tab_momentum, tab_value = st.tabs(["Carry", "Momentum", "Value"])

with tab_carry:
    render_carry()

with tab_momentum:
    render_momentum()

with tab_value:
    render_value()
