import streamlit as st
import os

from nextstep.service.preview_dataset import preview_dataset
from nextstep.service.preview_tar import preview_tar

# fmt: off
################################################################################
# Initialize page config & variables
################################################################################
st.set_page_config(layout="wide", initial_sidebar_state="auto")

page_names_to_funcs = {
    "preview_tar"        : preview_tar,
    "preview_dataset"    : preview_dataset,
}

demo_name = st.sidebar.selectbox(
    "Choose a preview function",
    page_names_to_funcs.keys(),
    placeholder="Choose a preview function...",
)
page_names_to_funcs[demo_name]()
# fmt: on
