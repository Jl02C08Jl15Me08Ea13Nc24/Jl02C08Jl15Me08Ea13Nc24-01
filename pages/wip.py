import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys

# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import deploy, st_ml


# Streamlit UI setup
st.set_page_config(page_title="Sales Prediction Analysis - ML Model Evaluation")
st.title('Machine Learning (ML) Model Performance')
# Explanation
st.info(
    """
    This section provides
    """
)

# Load File Structure Dictionary
with open ('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Verify data loading
try:
    X_train = deploy.load_Xy_data(FPATHS['data']['ml']['X_train.joblib'])
    y_train = deploy.load_Xy_data(FPATHS['data']['ml']['y_train.joblib'])
    X_test = deploy.load_Xy_data(FPATHS['data']['ml']['X_test.joblib'])
    y_test = deploy.load_Xy_data(FPATHS['data']['ml']['y_test.joblib'])
    preprocessor = deploy.load_Xy_data(FPATHS['data']['ml']['preprocessor.joblib'])
    feature_names = deploy.load_Xy_data(FPATHS['data']['ml']['feature_names.joblib'])
    st.write("Data loaded successfully!")
    #st.write("Sample X_train data:", X_train.head())
except Exception as e:
    st.error(f"Failed to load data: {e}")

# Verify model loading
try:
    best_model = deploy.load_ml_model(FPATHS['models']['ml']['xgboost_rscv_tuned_model.joblib'])
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Streamlit UI setup
st.sidebar.header("Model Evaluation Options")


# Sidebar setup for execution control
execute_evaluation = st.sidebar.button("Evaluate Model")
show_shap = st.sidebar.button("Perform SHAP Analysis")
show_lime = st.sidebar.button("Perform LIME Analysis")

if execute_evaluation:
    with st.spinner('Evaluating model... Please wait.'):
        # Model Performance Metrics
        st.header('Model Performance Metrics')
        metrics_df = st_ml.st_model_metrics(best_model, X_train, y_train, X_test, y_test,
                                            model_name='XGBoost RandomizedSearchCV Tuned Model',
                                            metric_decimals=3)
        st.dataframe(metrics_df)

        # Absolute Errors
        st.header('Absolute Errors')
        error_results, error_plot = st_ml.st_error_analysis(best_model,
                                                            X_train, y_train, X_test, y_test,
                                                            model_name='XGBoost RandomizedSearchCV Tuned Model',
                                                            error_decimals=4)
        st.dataframe(error_results)
        st.plotly_chart(error_plot, use_container_width=True)

        # Feature Importances
        feature_importances_results, feature_importances_plot = st_ml.st_feature_importances(
            best_model, 
            feature_names, 
            model_name='XGBoost RandomizedSearchCV Tuned Model',
            feature_imp_decimals=4)
        if feature_importances_plot is not None:
            st.header('Feature Importances')
            st.dataframe(feature_importances_results)
            st.plotly_chart(feature_importances_plot, use_container_width=True)

        # Permutation Importances
        st.header('Permutation Importances')
        perm_importances_results, perm_importances_plot = st_ml.st_permutation_importances(
            best_model, 
            X_train, y_train, 
            model_name='XGBoost RandomizedSearchCV Tuned Model',
            perm_decimals=4)
        st.dataframe(perm_importances_results)
        st.plotly_chart(perm_importances_plot, use_container_width=True)

        # Coefficients
        coeffs_results, coeffs_plot = st_ml.st_coefficients(best_model,
                                                            feature_names,
                                                            model_name='XGBoost RandomizedSearchCV Tuned Model',
                                                            coeff_decimals=3)
        if coeffs_plot is not None:
            st.header('Coefficients')
            st.dataframe(coeffs_results)
            st.plotly_chart(coeffs_plot, use_container_width=True)


# ------------------------------



import matplotlib.pyplot as plt
import shap

# SHAP
if show_shap:
    with st.spinner('Evaluating model... Please wait.'):
        model_name = "XGBoost RandomizedSearchCV Tuned Model"  
    
        shap_plot, shap_values, shap_df, X_train_preprocessed = st_ml.st_shap_analysis(model=best_model,
                                                                                 preprocessor=preprocessor,
                                                                                 X_train=X_train,
                                                                                 feature_names=feature_names,
                                                                                 model_name=model_name)
        
        # Display SHAP force plot
        st.subheader(f'{model_name} - SHAP Force Plot')
        st.components.v1.html(shap_plot, height=300)
        
        # Display SHAP Summary plot
        st.subheader(f'{model_name} - SHAP Summary Plot')
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, 
                          X_train_preprocessed, 
                          feature_names=feature_names, 
                          plot_type="bar")
        st.pyplot(fig)
        
        # Display DataFrame
        st.subheader(f'{model_name} - Mean Absolute SHAP Values')
        st.dataframe(shap_df.style.format({'Mean Absolute SHAP Value': "{:.3f}"}))



# ------------------------------



