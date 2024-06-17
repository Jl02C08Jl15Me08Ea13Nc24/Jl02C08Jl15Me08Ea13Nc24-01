# Predictions and LIME 

#---------------------------------------------------------------------------
# Libraries
#--------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import json, os, sys
import tensorflow as tf
import streamlit.components.v1 as components

# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import stfns


#--------------------------------------------------------------------------
# TITLES
#--------------------------------------------------------------------------
# Set the title of the browser tab
st.set_page_config(page_title="IMDB Movie Analysis - Predictive Text Analysis")
# Set title
st.title('Predictive Text Analysis with LIME Explanations')
# Explanation about LIME
st.info(
    """
    This section predicts the sentiment of movie reviews and provides insights into how the prediction was made using a method known as LIME (Local Interpretable Model-agnostic Explanations). LIME helps to understand why the model made a certain prediction by highlighting the most influential words in the review.
    """
)


#--------------------------------------------------------------------------
# DATA
#--------------------------------------------------------------------------
# Load File Structure Dictionary
with open ('config/filepaths.json') as f:
    FPATHS = json.load(f)
    
# Load target lookup dictionary 
target_lookup_fpath = FPATHS['data']['ml']['target_lookup_dict']
target_lookup = stfns.load_lookup(target_lookup_fpath)

# Load labelEncoder
label_encoder_fpath = FPATHS['data']['ml']['label_encoder']
encoder = stfns.load_encoder(label_encoder_fpath)

# Create Lime Explainer
explainer = stfns.get_explainer(class_names = encoder.classes_)


#--------------------------------------------------------------------------
# MODELS (only the best performing models)
#--------------------------------------------------------------------------
# Load Logistic Regression model 
logistic_regression_model_fpath = FPATHS['models']['logistic_regression']['logistic_regression_gridsearch_model_1']
logistic_regression_model = stfns.load_ml_model(logistic_regression_model_fpath)

# Load Bi-directional model 
bi_directional_model_fpath = FPATHS['models']['bi_directional']['bi_di_gru2_model']
bi_directional_model = stfns.load_tf_model(bi_directional_model_fpath)


#--------------------------------------------------------------------------
# Text Input Box 
#--------------------------------------------------------------------------
# Step 1: Enter Text
st.markdown("##### 1. Enter Text to Predict")
X_to_pred = st.text_input("", value="I feel that...")
# Add space
st.write("")


# -------------------------------------
# Model Selection (models dictionary)
# -------------------------------------
# Step 2: Select a Model
st.markdown("##### 2. Select a Model to Evaluate the Text")
models = {
    "Machine Learning (ML) Model": logistic_regression_model,
    "Natural Language Processing (NLP) Model": bi_directional_model,
}

model_name = st.radio(
    "",
    list(models.keys())
)

selected_model = models[model_name]
# Add space
st.write("")


# -------------------------------------
# Predictions
# -------------------------------------
# Step 3: Generate Prediction and Explanation
st.markdown("##### 3. Get Prediction and Local Interpretable Model-agnostic Explanations (LIME)")

# Trigger prediction with a button
if st.button("Generate Prediction and LIME Explanations"):
    if model_name == "Natural Language Processing (NLP) Model":
        # For NLP model
        pred_class_name = stfns.predict_decode_deep(X_to_pred, selected_model, target_lookup)
        # Determine the number of stars and color based on the predicted category
        stars = "🌟🌟🌟🌟🌟" if pred_class_name == "High_rating" else "🌟"
        color = "#1f77b4" if pred_class_name == "High_rating" else "#ff7f0e"
        st.markdown(f"###### The Neural Network Predicted a review of category: <span style='color:{color}; font-weight:bold;'>{pred_class_name} {stars}</span>", unsafe_allow_html=True)

        with st.spinner("Please wait while the LIME explanations are being generated..."):
            def predict_proba(texts):
                return selected_model.predict(texts)  

            # Generate and display LIME explanation
            html_explanation = stfns.explain_instance(explainer, X_to_pred, predict_proba)
            components.html(html_explanation, height=400)

    else:
        # For ML model
        pred_class = stfns.make_prediction(X_to_pred, model_to_pred=selected_model, lookup_dict=target_lookup)
        # Determine the number of stars and color based on the predicted category
        stars = "🌟🌟🌟🌟🌟" if pred_class == "High_rating" else "🌟"
        color = "#1f77b4" if pred_class == "High_rating" else "#ff7f0e"
        st.markdown(f"###### The ML Model Predicted a review of category: <span style='color:{color}; font-weight:bold;'>{pred_class} {stars}</span>", unsafe_allow_html=True)
        with st.spinner("Please wait while the LIME explanations are being generated..."):
            # Get the Explanation as HTML and display using the .html component.
            html_explanation = stfns.explain_instance(explainer, X_to_pred, selected_model.predict_proba)
            components.html(html_explanation, height=400)

else: 
    st.empty()
