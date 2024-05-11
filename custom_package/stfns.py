# Functions for Streamlit 
import streamlit as st
import pandas as pd
import os
import joblib
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer


#-------------------------------------------------------------------------- 
# Function to load CSV data
@st.cache_data
def load_csv_data(filepath):
    return pd.read_csv(filepath)

#-------------------------------------------------------------------------- 
# Function to load Parquet data
@st.cache_data
def load_parquet_data(filepath):
    return pd.read_parquet(filepath)

#-------------------------------------------------------------------------- 
# Function to load CSV or Parquet data
@st.cache_data
def load_csv_parquet_data(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Parquet.")

#-------------------------------------------------------------------------- 
# Function to load DF data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)
        
#-------------------------------------------------------------------------- 
# Function to load target lookup
@st.cache_data
def load_lookup(fpath):
    return joblib.load(fpath)

#-------------------------------------------------------------------------- 
# Function to load ecoder
@st.cache_resource
def load_encoder(fpath):
    return joblib.load(fpath)

#-------------------------------------------------------------------------- 
# Function to load tf dataset
@st.cache_resource
def load_tf_dataset(fpath):
    return tf.data.Dataset.load(fpath)

#-------------------------------------------------------------------------- 
# Function to load tf network
@st.cache_resource
def load_network(fpath):
    return tf.keras.models.load_model(fpath)

#-------------------------------------------------------------------------- 
# Function to load TensorFlow models
@st.cache_resource
def load_tf_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model at {model_path} loaded successfully.")
        # Print the model summary
        print("Model Summary:")
        model.summary()
        return model
    except Exception as e:
        print(f"Failed to load model at {model_path}. Error: {e}")

#-------------------------------------------------------------------------- 
# Function to load ML models
@st.cache_resource
def load_ml_model(fpath):
    return joblib.load(fpath)

#-------------------------------------------------------------------------- 
# Function to load either joblib or TensorFlow models
@st.cache_resource
def load_model(fpath):
    if fpath.endswith('.joblib'):
        return joblib.load(fpath)
    else:
        return tf.keras.models.load_model(fpath)

#-------------------------------------------------------------------------- 
# Function to Predict
def predict_decode_deep(X_to_pred, network, lookup_dict):
    if isinstance(X_to_pred, str):
        X = [X_to_pred]
    else:
        X = X_to_pred
    pred_probs = network.predict(X)
    pred_class = convert_y_to_sklearn_classes(pred_probs)
    # Decode label
    class_name = lookup_dict[pred_class[0]]
    return class_name

#-------------------------------------------------------------------------- 
# Function to decode the prediction
def make_prediction(X_to_pred, model_to_pred, lookup_dict):
    # Get Prediction
    pred_class = model_to_pred.predict([X_to_pred])[0]
    print(f"Predicted class: {pred_class}")
    # Decode label
    pred_class = lookup_dict[pred_class]
    return pred_class

#-------------------------------------------------------------------------- 
# Function to load Explainer
@st.cache_resource
def get_explainer(class_names = None):
    lime_explainer = LimeTextExplainer(class_names=class_names)
    return lime_explainer

#-------------------------------------------------------------------------- 
# Function to explain intance    
def explain_instance(explainer, X_to_pred, predict_func):
    explanation = explainer.explain_instance(X_to_pred, predict_func)
    return explanation.as_html(predict_proba=False)

#-------------------------------------------------------------------------- 
# Function to load scattertext
@st.cache_data
def load_scattertext(fpath):
    with open(fpath) as f:
        explorer = f.read()
        return explorer

#-------------------------------------------------------------------------- 
# Function to display classification metrics and confusion matrix within Streamlit
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
def classification_metrics_streamlit(y_true, y_pred, label='',
                                     figsize=(3, 3),
                                     normalize='true',
                                     cmap='Blues',
                                     colorbar=False,
                                     values_format=".2f",
                                     class_names=None,
                                     title_fontsize=6, 
                                     label_fontsize=4, 
                                     ticks_fontsize=4,  
                                     matrix_text_fontsize=6): 
    """Display classification metrics and confusion matrix in Streamlit."""
    # Get the classification report
    report = classification_report(y_true, y_pred, target_names=class_names)

    # Save header and report
    header = "-"*70
    final_report = "\n".join([header, f" Classification Metrics: {label}", header, report, "\n"])

    # Display the classification report in Streamlit
    st.text(final_report)
        
    # CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    # Plot the confusion matrix - raw counts
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                         normalize=None,
                                                         cmap='gist_gray_r',
                                                         display_labels=class_names,
                                                         values_format="d",
                                                         colorbar=colorbar,
                                                         ax=axes[0])
    axes[0].set_title("Raw Counts", fontsize=title_fontsize)
    axes[0].set_xlabel("Predicted label", fontsize=label_fontsize)
    axes[0].set_ylabel("True label", fontsize=label_fontsize)
    axes[0].tick_params(axis='x', labelsize=ticks_fontsize)
    axes[0].tick_params(axis='y', labelsize=ticks_fontsize)

    # Adjust font size for the numbers in the raw counts confusion matrix
    for text in cm_display.text_.ravel():
        text.set_fontsize(matrix_text_fontsize)

    # Plot the confusion matrix - normalized
    cm_display_norm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                              normalize=normalize,
                                                              cmap=cmap,
                                                              display_labels=class_names,
                                                              values_format=values_format,
                                                              colorbar=colorbar,
                                                              ax=axes[1])
    axes[1].set_title("Normalized Confusion Matrix", fontsize=title_fontsize)
    axes[1].set_xlabel("Predicted label", fontsize=label_fontsize)
    axes[1].set_ylabel("True label", fontsize=label_fontsize)
    axes[1].tick_params(axis='x', labelsize=ticks_fontsize)
    axes[1].tick_params(axis='y', labelsize=ticks_fontsize)

    # Adjust font size for the numbers in the normalized confusion matrix
    for text in cm_display_norm.text_.ravel():
        text.set_fontsize(matrix_text_fontsize)

    # Adjust layout
    fig.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)  

    return final_report, fig


#--------------------------------------------------------------------------     
def get_true_pred_labels(model,ds):

    y_true = []
    y_pred_probs = []
    
    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():
        
        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)
        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)
    ## Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    return y_true, y_pred_probs

#-------------------------------------------------------------------------- 
def convert_y_to_sklearn_classes(y, verbose=False):
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y
        
    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)
    
    else:
        if verbose:
            print("y is 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)


#-------------------------------------------------------------------------- 
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
def classification_metrics_streamlit_tensorflow(model, data=None, label='Training Data',
                                                figsize=(3, 3), normalize='true',
                                                output_dict=False,
                                                cmap='Blues',
                                                values_format=".2f", 
                                                class_names=None,
                                                colorbar=False,
                                                title_fontsize=6, 
                                                label_fontsize=4, 
                                                ticks_fontsize=4,  
                                                matrix_text_fontsize=6):
    # Check if data is a dataset
    if hasattr(data, 'map'):
        # If it IS a Dataset:
        # extract y_true and y_pred with helper function
        y_true, y_pred = get_true_pred_labels(model, data)
    else:
        # Get predictions for the data
        y_pred = model.predict(data)

    # Pass both y-vars through helper compatibility function
    y_true = convert_y_to_sklearn_classes(y_true)
    y_pred = convert_y_to_sklearn_classes(y_pred)
    
    # Call the helper function to obtain classification metrics for the data
    report, fig = classification_metrics_streamlit(y_true, y_pred, 
                                                   figsize=figsize,
                                                   colorbar=colorbar, cmap=cmap, 
                                                   values_format=values_format, label=label,
                                                   class_names=class_names,
                                                   title_fontsize=title_fontsize, 
                                                   label_fontsize=label_fontsize, 
                                                   ticks_fontsize=ticks_fontsize,  
                                                   matrix_text_fontsize=matrix_text_fontsize)
    # Adjust layout
    fig.tight_layout()
    
    return report, fig


#-------------------------------------------------------------------------- 
import pandas as pd
import nltk
def get_ngram_measures_finder(tokens, 
                              ngrams=2, 
                              measure='raw_freq', 
                              top_n=None, 
                              min_freq = 1,
                              words_colname='Words'):
    if ngrams == 4:
        MeasuresClass = nltk.collocations.QuadgramAssocMeasures
        FinderClass = nltk.collocations.QuadgramCollocationFinder
    elif ngrams == 3: 
        MeasuresClass = nltk.collocations.TrigramAssocMeasures
        FinderClass = nltk.collocations.TrigramCollocationFinder
    else:
        MeasuresClass = nltk.collocations.BigramAssocMeasures
        FinderClass = nltk.collocations.BigramCollocationFinder

    measures = MeasuresClass()
    
   
    finder = FinderClass.from_words(tokens)
    finder.apply_freq_filter(min_freq)
    if measure=='pmi':
        scored_ngrams = finder.score_ngrams(measures.pmi)
    else:
        measure='raw_freq'
        scored_ngrams = finder.score_ngrams(measures.raw_freq)

    df_ngrams = pd.DataFrame(scored_ngrams, columns=[words_colname, measure.replace("_",' ').title()])
    if top_n is not None:
        return df_ngrams.head(top_n)
    else:
        return df_ngrams



# Function to to specify the group names
#-------------------------------------------------------------------------- 
@st.cache_data
def compare_group_ngrams(group1_tokens, group2_tokens, group1_name, group2_name, ngrams, measure, min_freq, top_n):

    # Get Ngram df for group 1
    df_group1 = get_ngram_measures_finder(group1_tokens,ngrams=ngrams, measure=measure, min_freq=min_freq, 
                                        top_n=top_n)
    ## Rename group1 columns for streamlit compatibility
    new_group1_cols = [f"{group1_name} - {col}" for col in df_group1.columns]
    df_group1.columns = new_group1_cols
    
        
    # Get Ngram df for group 2
    df_group2 = get_ngram_measures_finder(group2_tokens,ngrams=ngrams, measure=measure, min_freq=min_freq, 
                                        top_n=top_n)
    ## Rename group1 columns for streamlit compatibility
    new_group2_cols = [f"{group2_name} - {col}" for col in df_group2.columns]
    df_group2.columns = new_group2_cols

    
    # Combine low and high reviews score dfs and add a group name as multi-index
    df_compare_ngrams = pd.concat(
        [df_group1, df_group2],
        axis=1)
    
    return df_compare_ngrams

#-------------------------------------------------------------------------- 

        
#-------------------------------------------------------------------------- 
# Function footer links
def add_sidebar_links():
    links_html = """
    <ul>
    <li><a href="https://github.com/Edgar-Villasenor/IMDB-Movie-Analysis-Projectm">Project Repo</a></li>
      <li><a href="https://www.linkedin.com/in/edgarvillasenor/" rel="nofollow noreferrer">
        <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
      </a> </li>
      <li><a href="https://github.com/Edgar-Villasenor" rel="nofollow noreferrer">
        <img src="https://i.stack.imgur.com/tskMh.png" alt="github"> Github
      </a></li>
    </ul>
    """
    st.sidebar.markdown(links_html, unsafe_allow_html=True)

