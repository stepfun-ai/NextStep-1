import streamlit as st
from PIL import Image


def init_session_state(**kwargs):
    """Placed at the beginning of the program for initialization, to prevent rerun from forcibly overwriting."""
    for key in kwargs:
        if key not in st.session_state:
            st.session_state[key] = kwargs[key]


def reset_session_state(**kwargs):
    """Forcibly overwrite session state."""
    for key in kwargs:
        st.session_state[key] = kwargs[key]


def resize_image_h(img, max_size=256):
    w, h = img.size
    ratio = max_size / h
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img
