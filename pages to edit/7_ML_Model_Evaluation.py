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
from custom_package import stfns


#--------------------------------------------------------------------------
# TITLES
#--------------------------------------------------------------------------
# Set the title of the browser tab
st.set_page_config(page_title="IMDB Movie Analysis - ML Model Evaluation")
# title
st.title('Machine Learning (ML) Model Performance')
# Explanation
st.info(
    """
    This section provides a detailed evaluation of the Machine Learning model's performance, utilizing a Logistic Regression Model. Selection of specific metrics for display is facilitated through checkboxes in the sidebar, enabling a customized analysis that encompasses classification metrics across training and test datasets. Insights into the model's accuracy, precision, recall, and F1-score, among other key performance indicators, are presented. 
    
    Additionally, the option to unveil model parameters via the sidebar enriches the understanding of the configurations that contribute to the model's predictions. Activation of the 'Evaluate Model' button in the sidebar initiates this comprehensive evaluation, aimed at illuminating the model's predictive strengths and areas for enhancement.
    """
)


#--------------------------------------------------------------------------
# DATA
#--------------------------------------------------------------------------
# Load File Structure Dictionary
with open ('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Load training
train_data_fpath = FPATHS['data']['ml']['training_data']
X_train, y_train = stfns.load_Xy_data(train_data_fpath)

# Load testing data
test_data_fpath = FPATHS['data']['ml']['testing_data']
X_test, y_test = stfns.load_Xy_data(test_data_fpath)

# Load target lookup dictionary 
target_lookup_fpath = FPATHS['data']['ml']['target_lookup_dict']
target_lookup = stfns.load_lookup(target_lookup_fpath)

# Load labelEncoder
label_encoder_fpath = FPATHS['data']['ml']['label_encoder']
encoder = stfns.load_encoder(label_encoder_fpath)


#--------------------------------------------------------------------------
# ML MODEL (only the best performing model)
#--------------------------------------------------------------------------
# Load Logistic Regression model 
logistic_regression_model_fpath = FPATHS['models']['logistic_regression']['logistic_regression_model_1']
logistic_regression_model = stfns.load_ml_model(logistic_regression_model_fpath)


#--------------------------------------------------------------------------
# Classification Metrics
#--------------------------------------------------------------------------
# Options to display training and test data
show_train = st.sidebar.checkbox("Training Data Classification Metrics", value=True)
show_test = st.sidebar.checkbox("Test Data \n Classification Metrics", value=True)
show_model_params =st.sidebar.checkbox("Show model params.", value=False)

if st.sidebar.button("Evaluate Model"):
    with st.spinner("Please wait while the model is evaluated..."):
        # Display the type of model
        st.write('Logistic Regression Model')
        if show_train:
            # Display training data results
            y_pred_train = logistic_regression_model.predict(X_train) 
            report_str, fig = stfns.classification_metrics_streamlit(y_train, y_pred_train, label='Training Data')
            st.text("\n\n")
        if show_test:
            # Display test data results
            y_pred_test = logistic_regression_model.predict(X_test)  
            report_str, fig = stfns.classification_metrics_streamlit(y_test, y_pred_test, cmap='Reds', label='Test Data')
            st.text("\n\n")
            
        if show_model_params:
            try:
                model_params = logistic_regression_model.get_params()
            except AttributeError:  
                model_params = 'Model parameters not accessible'
            st.markdown("#### Model Parameters:")
            st.write(model_params)
else:
    st.empty()

