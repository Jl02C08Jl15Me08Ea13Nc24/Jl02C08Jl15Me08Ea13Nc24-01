# ML Model Evaluation

#---------------------------------------------------------------------------
# Libraries
#--------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys

# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import deploy, ml


#--------------------------------------------------------------------------
# TITLES
#--------------------------------------------------------------------------
# Set the title of the browser tab
st.set_page_config(page_title="Sales Prediction Analysis - ML Model Evaluation")
# title
st.title('Machine Learning (ML) Model Performance')
# Explanation
st.info(
    """
    This section provides a detailed evaluation of the Machine Learning model's performance, utilizing a XGBoost RandomizedSearchCV Tuned Model. Selection of specific metrics for display is facilitated through checkboxes in the sidebar, enabling a customized analysis evaluating using standard metrics such as MAE, MSE, RMSE, and R² to compare their performance on both training and testing datasets
    
    Additionally, the evaluation options like SHAP and LIME are available
    """
)


#--------------------------------------------------------------------------
# DATA
#--------------------------------------------------------------------------
# Load File Structure Dictionary
with open ('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Load training and testing data 
X_train = deploy.load_Xy_data(FPATHS['data']['ml']['X_train.joblib'])
y_train = deploy.load_Xy_data(FPATHS['data']['ml']['y_train.joblib'])
X_test = deploy.load_Xy_data(FPATHS['data']['ml']['X_test.joblib'])
y_test = deploy.load_Xy_data(FPATHS['data']['ml']['y_test.joblib'])
preprocessor = deploy.load_Xy_data(FPATHS['data']['ml']['preprocessor.joblib'])
feature_names = deploy.load_Xy_data(FPATHS['data']['ml']['feature_names.joblib'])

#--------------------------------------------------------------------------
# ML MODEL (only the best performing model)
#--------------------------------------------------------------------------
# Load the best model (XGBoost RandomizedSearchCV Tuned Model)
best_model = deploy.load_ml_model(FPATHS['models']['ml']['xgboost_rscv_tuned_model.joblib'])

#--------------------------------------------------------------------------
# Model Evaluation
#--------------------------------------------------------------------------
ml.single_model_evaluation(
    model=best_model,     
    preprocessor=preprocessor,
    X_train=X_train,      
    y_train=y_train,
    X_test=X_test,         
    y_test=y_test,
    features_imp=True,     
    perms=True,            
    metric_decimals=3,
    perm_decimals=4,
    feature_imp_decimals=4,
    error_decimals=4,
    model_name='XGBoost RandomizedSearchCV Tuned Model'
)
