# ------------------------------
# Libraries
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import deploy, st_ml

#-------------------------------
# TITLES
# ------------------------------
# Streamlit UI setup
st.set_page_config(page_title="Sales Prediction Analysis - ML Model Evaluation")
st.title('Machine Learning (ML) Model Performance')
#--------------------------------------------------------------------------
# Overall ML model Interpretation
#--------------------------------------------------------------------------
with st.expander("📌 **Click here to view overall ML model insights**"):
        
    st.markdown("""
### Overview and Future Directions

Throughout the evaluation of a series of machine learning models and detailed analyses like SHAP and LIME, it has become apparent that while specific features significantly impact sales predictions, the overall model performances have been modest, with R² values consistently below 0.6. This points to a fundamental limitation in the dataset or the ability of the features to fully capture the complexities of sales dynamics.

- **Data Limitations**: The apparent lack of variance or absence of critical features that significantly influence sales suggests that essential explanatory variables might be missing. This deficiency limits the models' ability to effectively learn and predict the underlying patterns.

- **Model Complexity**: Despite employing advanced techniques such as Gradient Boosting Machines and extensive hyperparameter tuning, a breakthrough in predictive performance was not achieved. This indicates that the challenges inherent in the dataset extend beyond model sophistication and may relate more to the approach to modeling or the nature of the data itself. The analysis was confined solely to machine learning models, potentially limiting the exploration of alternative analytical techniques.

- **Noise in the Data**: The high variability in error rates, especially during testing, indicates significant noise within the data, which may obscure true patterns and complicate accurate modeling of the target variable.

- **Feature Impact and Insights**: Insights from SHAP and LIME analyses have reinforced the importance of features like `item_mrp`, `outlet_type`, and `outlet_establishment_year`. These features significantly impact sales predictions, but their moderate overall influence highlights the complexity and possible interactions among variables not fully captured in the models.

**Future Analysis Considerations**:

- **Data Augmentation**: Incorporating additional variables, could enrich the dataset and provide a more solid foundation for modeling sales dynamics.

- **Feature Engineering**: Developing features that capture interactions between existing variables or incorporating polynomial features to model non-linear relationships could yield new insights into the complex patterns within the data.

- **Exploration of Alternative Analytical Approaches**: The observed complexities could benefit from applying alternative analytical methods, such as deep learning or ensemble models, which might capture intricate patterns and interactions more effectively.

**Concluding Observations**:

The objective of this project was to improve sales predictions through a well-structured analytical framework. While the models provided valuable insights, their modest predictive accuracy points to the need for broader data collection, advanced feature engineering, and potentially a reevaluation of the analytical methods employed. Future efforts should focus on these areas to refine the predictions and achieve a deeper understanding of the sales dynamics.
""")
    
# ------------------------------
# Data Loading
# ------------------------------
# Load File Structure Dictionary
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Load data and model
try:
    X_train = deploy.load_Xy_data(FPATHS['data']['ml']['X_train.joblib'])
    y_train = deploy.load_Xy_data(FPATHS['data']['ml']['y_train.joblib'])
    X_test = deploy.load_Xy_data(FPATHS['data']['ml']['X_test.joblib'])
    y_test = deploy.load_Xy_data(FPATHS['data']['ml']['y_test.joblib'])
    preprocessor = deploy.load_Xy_data(FPATHS['data']['ml']['preprocessor.joblib'])
    feature_names = deploy.load_Xy_data(FPATHS['data']['ml']['feature_names.joblib'])
    best_model = deploy.load_ml_model(FPATHS['models']['ml']['xgboost_rscv_tuned_model.joblib'])
    data_loaded = True
    st.success("Data loaded successfully!\nYou can now use the sidebar to explore the model evaluation, including performance metrics, absolute errors, feature importances, and permutation importances of the best model. For SHAP force plots, summary plots, and a table of mean absolute SHAP values, select 'SHAP Analysis'. For an examination of LIME explainer results, choose 'LIME Analysis'.")

except Exception as e:
    st.error(f"Failed to load data or model: {e}")
    data_loaded = False
# ------------------------------


if data_loaded:
    # ------------------------------
    # Sidebar
    st.sidebar.header("Model Evaluation Options")
    execute_evaluation = st.sidebar.button("Evaluate Model")
    show_shap = st.sidebar.button("Run SHAP Analysis")
    show_lime = st.sidebar.button("Run LIME Analysis")
    
    # ------------------------------
    # Evaluation and Analysis
    # ------------------------------
    # Model Performance
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
    # LIME
    if show_lime:
        with st.spinner('Performing LIME analysis... Please wait.'):
            model_name = "XGBoost RandomizedSearchCV Tuned Model"
            
            lime_exp, lime_df, lime_plot = st_ml.st_lime_analysis(
                model=best_model,
                preprocessor=preprocessor,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                model_name=model_name
            )
            
            # Display LIME explanation
            st.subheader(f'{model_name} - LIME Explanation for Single Instance')
            st.components.v1.html(lime_exp.as_html(), height=800)
            
            # Display DataFrame
            st.subheader(f'{model_name} - LIME Feature Importance')
            st.dataframe(lime_df.style.format({'Importance': "{:.3f}"}))
            
            # Display bar plot
            st.plotly_chart(lime_plot, use_container_width=True)
else:
    st.error("Cannot perform analysis. Data or model not loaded correctly.")
# ------------------------------
