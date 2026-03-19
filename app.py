import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.styles import load_css, nav_html, disclaimer_html

st.set_page_config(
    page_title="Sambhav - Uncertainty, Quantified.",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(load_css(), unsafe_allow_html=True)
st.markdown(nav_html(""), unsafe_allow_html=True)
st.markdown(open("pages/home.html").read(), unsafe_allow_html=True)
st.markdown(disclaimer_html(), unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stButton"] { display: none; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("__predict__"):
        st.switch_page("pages/3_Dashboard.py")
with col2:
    if st.button("__modes__"):
        st.switch_page("pages/2_Modes.py")
