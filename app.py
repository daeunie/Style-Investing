"""
Style Investing -- main entry point
--------------------------------------
This is the file to point Streamlit Cloud's "Main file path" at
(not carry/app.py anymore). It sets up the whole site's page config
and title once, then delegates each top-level tab to that factor's own
render() function.

Repo layout this expects:
    app.py                     <- this file
    carry/carry_tab.py         <- Carry factor (done)
    carry/data/*.csv
    momentum/momentum_tab.py   <- Momentum factor (placeholder for now)
    value/value_tab.py         <- Value factor (placeholder for now)
"""

import streamlit as st

from carry.carry_tab import render as render_carry
from momentum.momentum_tab import render as render_momentum
from value.value_tab import render as render_value

st.set_page_config(page_title="Style Investing", layout="wide")

st.title("Style Investing")
st.caption(
    "AQR-style factor investing \u2014 carry, momentum, value \u2014 applied to US "
    "Treasury ETFs, adapted from Brooks, Palhares & Richardson (2018)."
)

tab_carry, tab_momentum, tab_value = st.tabs(["Carry", "Momentum", "Value"])

with tab_carry:
    render_carry()

with tab_momentum:
    render_momentum()

with tab_value:
    render_value()
