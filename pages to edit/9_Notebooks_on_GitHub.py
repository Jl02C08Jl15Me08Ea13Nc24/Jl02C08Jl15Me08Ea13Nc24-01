# Notebooks Page

#--------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------
import streamlit as st
import os, sys
# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import stfns


# Set the title of the browser tab
st.set_page_config(page_title="IMDB Movie Analysis - Notebooks")
#--------------------------------------------------------------------------
# HTML Notebooks with nbviewer
#--------------------------------------------------------------------------
def main():
    # List of notebook URLs and their names
    notebooks = {
        'Data Cleaning': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-1-Data-Cleaning.ipynb',
        'Database Creation': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-2-Database-Creation.ipynb',
        'TMDB Extraction': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-3-TMDB-Extraction.ipynb',
        'Exploratory Data Analysis': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-4-Exploratory-Data-Analysis.ipynb',
        'Hypothesis Testing': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-5-Hypothesis-Testing.ipynb',
        'Sentiment Analysis': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-6-Sentiment-Analysis.ipynb',
        'ML & NLP Models': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-7-ML-NLP-Models.ipynb',
        'Preparing for Streamlit': 'https://nbviewer.org/github/Edgar-Villasenor/IMDB-Movie-Analysis-Project/blob/main/IMDB-Movie-Analysis-Part-8-Preparing-for-Streamlit%20.ipynb',
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

