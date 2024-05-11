# Data Dictionary Page

#--------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------
import streamlit as st
import json, os, sys
# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import stfns

# Set the title of the browser tab
st.set_page_config(page_title="Sales Prediction Analysis - Data Dictionary")

#--------------------------------------------------------------------------
# Including a Markdown File with data dictionaries
#--------------------------------------------------------------------------
# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the path to the markdown file
md_file_path = os.path.join(script_dir, "dic_info.md")

# Open and display the markdown file
with open(md_file_path) as f:
    dic_info_fpath = f.read()
st.markdown(dic_info_fpath, unsafe_allow_html=True)
