# Notebooks Page

#--------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------
import streamlit as st
import os, sys

# Set the title of the browser tab
st.set_page_config(page_title="Sales Prediction Analysis - Notebooks")
#--------------------------------------------------------------------------
# HTML Notebooks with nbviewer
#--------------------------------------------------------------------------
def main():
    # List of notebook URLs and their names
    notebooks = {
        'Data Cleaning': 'https://nbviewer.org/github/Edgar-Villasenor/Sales-Prediction-Analysis/blob/main/Sales-Prediction-Analysis-Part-1-Data-Cleaning.ipynb',
        'Exploratory Data Analysis': 'https://nbviewer.org/github/Edgar-Villasenor/Sales-Prediction-Analysis/blob/main/Sales-Prediction-Analysis-Part-2-EDA.ipynb',
        'ML Models': 'https://nbviewer.org/github/Edgar-Villasenor/Sales-Prediction-Analysis/blob/main/Sales-Prediction-Analysis-Part-3-ML.ipynb',
        'Preparing for Streamlit': 'https://nbviewer.org/github/Edgar-Villasenor/Sales-Prediction-Analysis/blob/main/Sales-Prediction-Analysis-Part-4-Streamlit.ipynb',
    }

    # Create radio buttons for each notebook in the sidebar
    selected_notebook = st.sidebar.radio('Select a notebook', list(notebooks.keys()))

    # Get the URL of the selected notebook
    notebook_url = notebooks[selected_notebook]

    # Set the title of the app to the name of the selected notebook
    st.title(selected_notebook)

    # Embed the selected notebook
    st.markdown(f'<iframe src="{notebook_url}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

