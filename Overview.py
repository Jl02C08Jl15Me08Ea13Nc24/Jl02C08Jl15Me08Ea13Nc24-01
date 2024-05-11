## Main page
#--------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------
import streamlit as st
import json, os, sys
# Custom Functions
from custom_package import stfns

# Set the title of the browser tab
st.set_page_config(page_title="Sales Prediction Analysis - Overview")

#--------------------------------------------------------------------------
# Paths
#--------------------------------------------------------------------------
# File paths configuration
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)

#--------------------------------------------------------------------------
# Banner
#--------------------------------------------------------------------------
fpath_banner = FPATHS['images']['app']['app_banner.png']

# Display the image
st.image(fpath_banner)
#--------------------------------------------------------------------------
# Including a Markdown File Readme on main page
#--------------------------------------------------------------------------
# Open and display a .md file
with open("Streamlit_Overview.md") as f:
    readme = f.read()
st.markdown(readme,  unsafe_allow_html=True)