import streamlit as st

st.set_page_config(
    page_title="Deteksi Jalan Rusak",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
    )

st.page_link("pages/photo.py", label="Photo")
st.page_link("pages/video.py", label="Video")
st.page_link("pages/camera.py", label="Camera")