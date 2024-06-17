# Data Page

#--------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------
import streamlit as st
import json, os, sys
import pandas as pd
import numpy as np
# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import deploy

# Set the title of the browser tab
st.set_page_config(page_title="Sales Prediction Analysis - Data")

#--------------------------------------------------------------------------
# Data Dictionary Section
#--------------------------------------------------------------------------
# title
st.title('Data Dictionary')
st.info("The data dictionary below lists each feature along with a brief description.")


# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the path to the markdown file
md_file_path = os.path.join(script_dir, "dic_info.md")

# Open and display the markdown file
with open(md_file_path) as f:
    dic_info_fpath = f.read()
st.markdown(dic_info_fpath, unsafe_allow_html=True)


#--------------------------------------------------------------------------
# Data Loading Section 
#--------------------------------------------------------------------------
# title
st.title('Exploring Data')
st.info("Explore data using the sidebar by selecting columns, choosing operations, and executing custom queries.")


# Load File Structure Dictionary
with open ('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Define the filepath using dictionary structure 
df_fpath = FPATHS['data']['filtered']['sales_predictions_2023_preprocessing.csv']

# Load the dataframe from the file
df = deploy.load_csv_data(df_fpath)

#--------------------------------------------------------------------------
# Data Exploration Section 
#--------------------------------------------------------------------------
# Initialize session state for the filtered data from custom queries
if 'filtered_data' not in st.session_state:
    st.session_state['filtered_data'] = None

# Sidebar for options
st.sidebar.header("Data Exploration Setup")

# Step 1: Select columns with 'All' option
all_columns = list(df.columns)
column_selection = st.sidebar.multiselect('Select columns to display', options=['All'] + all_columns, default='All')
selected_columns = all_columns if 'All' in column_selection else column_selection
selected_df = df[selected_columns] if selected_columns else df

# Step 2: Choose operation
operation = st.sidebar.selectbox("Choose operation", ["Head", "Tail", "Random Sample", "Statistics", "Custom Query"])

# Step 3: Number of rows (conditional display)
num_rows = 5
if operation in ["Head", "Tail", "Random Sample"]:
    num_rows = st.sidebar.slider("Number of Rows to View", 1, 100, 5)

# Custom Query input and execute (conditional display)
if operation == "Custom Query":
    st.sidebar.markdown("""
    **Example Queries:**
    - `item_weight > 10`
    - `item_fat_content == 'Regular' and item_mrp < 50`
    - `1987 <= outlet_establishment_year <= 2009`
    """)
    custom_query = st.sidebar.text_area("Enter custom query")
    execute_query = st.sidebar.button("Execute Query")
    if execute_query:
        try:
            st.session_state['filtered_data'] = selected_df.query(custom_query)
        except Exception as e:
            st.error(f"Error with query: {e}")

# Display results based on session state or direct data manipulation
if st.session_state['filtered_data'] is not None:
    post_query_operation = st.selectbox("Post Query Operation", ["Default", "Head", "Tail", "Random Sample", "Statistics"])
    filtered_data = st.session_state['filtered_data']
    if post_query_operation == "Head":
        st.dataframe(filtered_data.head(num_rows))
    elif post_query_operation == "Tail":
        st.dataframe(filtered_data.tail(num_rows))
    elif post_query_operation == "Random Sample":
        st.dataframe(filtered_data.sample(n=num_rows))
    elif post_query_operation == "Statistics":
        st.write("### Numerical Features Statistics")
        st.dataframe(filtered_data.describe())
        categorical_cols = filtered_data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.write("### Categorical Features Statistics")
            st.dataframe(filtered_data[categorical_cols].describe())
    else:  # Default display
        st.dataframe(filtered_data)
else:
    # Show button for other operations when Custom Query is not selected
    if operation != "Custom Query" and st.sidebar.button("Show Results"):
        if operation == "Head":
            st.dataframe(selected_df.head(num_rows))
        elif operation == "Tail":
            st.dataframe(selected_df.tail(num_rows))
        elif operation == "Random Sample":
            st.dataframe(selected_df.sample(n=num_rows))
        elif operation == "Statistics":
            st.write("### Numerical Features Statistics")
            st.dataframe(selected_df.describe())
            categorical_cols = selected_df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.write("### Categorical Features Statistics")
                st.dataframe(selected_df[categorical_cols].describe())

# Reset button
if st.sidebar.button("Reset"):
    st.session_state['filtered_data'] = None
    st.experimental_rerun()

















