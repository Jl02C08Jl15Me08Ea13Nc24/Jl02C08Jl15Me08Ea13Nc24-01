# deployment functions

# ------------------------------
# Functions for Streamlit 
import streamlit as st
import pandas as pd
import os
import joblib
from pprint import pprint

# ------------------------------
# Prepare File Structure
def build_file_paths(root_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {'.ipynb_checkpoints', '.git'}

    # Dictionary for the FPATHS structure 
    FPATHS = {
        'data': {},
        'models': {},
        'images': {}
    }

    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Remove directories in exclude_dirs
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(tuple(['.csv', '.joblib', '.html', '.png', '.gz'])):
                file_path = os.path.relpath(os.path.join(root, file), start=root_dir)
                parts = file_path.split(os.sep)
                d = FPATHS
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = file_path
    
    return FPATHS



# ------------------------------
# Create directories
def create_directories_from_paths(nested_dict):
    """OpenAI. (2023). ChatGPT [Large language model]. https://chat.openai.com 
    Recursively create directories for file paths in a nested dictionary.
    Parameters:
    nested_dict (dict): The nested dictionary containing file paths.
    """
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            create_directories_from_paths(value)
        elif isinstance(value, str):
            # If the value is a string, treat it as a file path and get the directory path
            directory_path = os.path.dirname(value)
            # If the directory path is not empty and the directory does not exist, create it
            if directory_path and not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Directory created: {directory_path}")


# ------------------------------
def test_filepaths(nested_dict):
    """Recursively test file paths in a nested dictionary.
    Parameters:
    nested_dict (dict): The nested dictionary containing file paths.
    """
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            test_filepaths(value)
        elif isinstance(value, str):
            # If the value is a string, treat it as a file path
            if not os.path.exists(value):
                print(f"Path does not exist: {value}")
            else:
                print(f"Path exists: {value}")



# ------------------------------
def correct_paths(paths):
    """
    Recursively corrects file paths in a nested dictionary.
    """
    if isinstance(paths, dict):
        for key, value in paths.items():
            paths[key] = correct_paths(value)
    elif isinstance(paths, str):
        paths = paths.replace("\\", "/")
    return paths

#-------------------------------------------------------------------------- 
# Function to load CSV data
@st.cache_data
def load_csv_data(filepath):
    return pd.read_csv(filepath)

#-------------------------------------------------------------------------- 
# Function to load DF data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)

#-------------------------------------------------------------------------- 
# Function to load ML models
@st.cache_resource
def load_ml_model(fpath):
    return joblib.load(fpath)

# ------------------------------
