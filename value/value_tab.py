"""Value tab -- placeholder until the value strategy is built."""

import streamlit as st


def render():
    st.header("Value \u2014 US Treasury ETFs")
    st.info(
        "Not built yet. Value will use curve mispricing (actual yield vs. a "
        "fitted fair-value curve) as its signal, following the same "
        "style-investing framework as Carry."
    )
