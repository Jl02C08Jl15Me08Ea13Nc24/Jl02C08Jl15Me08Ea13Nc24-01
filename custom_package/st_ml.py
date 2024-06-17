# ------------------------------
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def st_model_metrics(model, 
                     X_train, y_train, X_test=None, y_test=None,
                     model_name='', 
                     metric_decimals=3):
    # Generate predictions
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'Data': model_name + ' Training Data',
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': mean_squared_error(y_train, y_train_pred, squared=False),
        'R^2': r2_score(y_train, y_train_pred)
    }

    metrics = [train_metrics]
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_metrics = {
            'Data': model_name + ' Test Data',
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': mean_squared_error(y_test, y_test_pred, squared=False),
            'R^2': r2_score(y_test, y_test_pred)
        }
        metrics.append(test_metrics)

    model_metrics_results_df = pd.DataFrame(metrics)
    return model_metrics_results_df.set_index('Data').style.format("{:." + str(metric_decimals) + "f}")  


# ------------------------------
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

def st_error_analysis(model,
                      X_train, y_train, X_test, y_test, 
                      model_name='',
                      error_decimals=3,
                      train_color='#5799c6', test_color='#d4a132'):
    
    # Generate predictions for training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate absolute and percentage errors for training data
    abs_errors_train = np.abs(y_train - y_train_pred)
    pct_errors_train = abs_errors_train / y_train * 100
    
    # Calculate absolute and percentage errors for testing data
    abs_errors_test = np.abs(y_test - y_test_pred)
    pct_errors_test = abs_errors_test / y_test * 100

    # Prepare summary statistics for training and testing
    summary_stats = {
        'Data': [f'{model_name} Training Data', f'{model_name} Testing Data'],
        'Mean Absolute Error': [np.mean(abs_errors_train), np.mean(abs_errors_test)],
        'Median Absolute Error': [np.median(abs_errors_train), np.median(abs_errors_test)],
        'Standard Deviation of Absolute Errors': [np.std(abs_errors_train), np.std(abs_errors_test)],
        'Mean Percentage Error': [np.mean(pct_errors_train), np.mean(pct_errors_test)],
        'Median Percentage Error': [np.median(pct_errors_train), np.median(pct_errors_test)],
        'Standard Deviation of Percentage Errors': [np.std(pct_errors_train), np.std(pct_errors_test)]
    }
    
    error_analysis_result_df = pd.DataFrame(summary_stats)
    error_analysis_result_df.set_index('Data', inplace=True)

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=abs_errors_train, 
                               name='Train Abs Errors', 
                               opacity=0.75, 
                               marker_color=train_color))
    fig.add_trace(go.Histogram(x=abs_errors_test, 
                               name='Test Abs Errors', 
                               opacity=0.75,
                               marker_color=test_color))
    fig.update_layout(title=f"{model_name} - Distribution of Absolute Errors",
                      xaxis_title="Absolute Errors",
                      yaxis_title="Frequency",
                      barmode='overlay')

    # Format DataFrame for Streamlit display
    styled_df = error_analysis_result_df.style.format("{:." + str(error_decimals) + "f}")

    return styled_df, fig

# ------------------------------
from sklearn.pipeline import Pipeline
import pandas as pd
import plotly.express as px

def st_feature_importances(model, feature_names, model_name='', feature_imp_decimals=3):
    # Check if the model is a pipeline and extract the estimator if it is
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]  

    if hasattr(model, 'feature_importances_'):
        # Retrieve feature importances
        importances = model.feature_importances_
        importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importances_df.sort_values(by='Importance', ascending=False, inplace=True)

        # Plotly visualization
        fig = px.bar(importances_df, x='Importance', y='Feature', 
                     title=f"{model_name} - Feature Importances",
                     labels={'Importance': 'Importance', 'Feature': 'Feature'},
                     orientation='h')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          xaxis_title="Importance",
                          yaxis_title="Feature")

        # Style the DataFrame for display
        styled_df = importances_df.style.format({'Importance': "{:." + str(feature_imp_decimals) + "f}"})
        
        return styled_df, fig
    else:
        # Return None if feature importances are not supported
        return None, None


# ------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.inspection import permutation_importance

def st_permutation_importances(model,
                               X_train, y_train, 
                               model_name='', 
                               n_repeats=30, 
                               random_state=42, 
                               perm_decimals=3):
    # Calculate permutation importances
    result = permutation_importance(model,
                                    X_train, y_train,
                                    n_repeats=n_repeats, 
                                    random_state=random_state)
    importances = result.importances_mean

    # Prepare DataFrame
    importances_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    })
    importances_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Plotly visualization
    fig = px.bar(importances_df, x='Importance', y='Feature',
                 title=f"{model_name} - Permutation Importances",
                 labels={'Importance': 'Mean Importance (Decrease in Model Score)', 'Feature': 'Feature'},
                 orientation='h')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      xaxis_title="Mean Decrease in Model Score",
                      yaxis_title="Feature")

    # Style DataFrame for display
    styled_df = importances_df.style.format({'Importance': "{:." + str(perm_decimals) + "f}"})

    return styled_df, fig

# ------------------------------
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline

def st_coefficients(model, feature_names, model_name='', coeff_decimals=3):
    # Check if the model is a pipeline and extract the estimator if it is
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]  

    # Check if model has 'coef_' attribute
    if hasattr(model, 'coef_'):
        # Retrieve coefficients
        coeffs = model.coef_
        if coeffs.ndim > 1:
            coeffs = coeffs.flatten()  

        coeffs_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coeffs
        })
        coeffs_df['Absolute Coefficient'] = coeffs_df['Coefficient'].abs()
        coeffs_df.sort_values(by='Absolute Coefficient', ascending=False, inplace=True)

        # Plotly visualization
        fig = px.bar(coeffs_df, x='Coefficient', y='Feature',
                     title=f"{model_name} - Feature Coefficients",
                     labels={'Coefficient': 'Coefficient', 'Feature': 'Feature'},
                     orientation='h')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                          xaxis_title="Coefficient Value",
                          yaxis_title="Feature")

        # Style DataFrame for display
        styled_df = coeffs_df.drop(columns='Absolute Coefficient').style.format({'Coefficient': "{:." + str(coeff_decimals) + "f}"})

        return styled_df, fig
    else:
        # Return None if coefficients are not supported
        return None, None

        
# ------------------------------

import streamlit as st
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def st_shap_analysis(model, preprocessor, X_train, feature_names, model_name=''):
    # Preprocess the training data as per the model's requirements
    X_train_preprocessed = preprocessor.transform(X_train)

    # Check if the model is a pipeline and extract the estimator
    if isinstance(model, Pipeline):
        # Assuming the estimator is the last step in the pipeline
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # Initialize SHAP Explainer using the estimator
    explainer = shap.Explainer(estimator, X_train_preprocessed)
    shap_values = explainer.shap_values(X_train_preprocessed)

    # SHAP Force Plot for an example prediction
    shap.initjs()
    force_plot_html = shap.force_plot(explainer.expected_value, 
                                      shap_values[0], 
                                      X_train_preprocessed[0], 
                                      feature_names=feature_names, 
                                      show=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"

    # Generate SHAP Summary Plot
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, 
                      X_train_preprocessed, 
                      feature_names=feature_names, 
                      plot_type="bar", 
                      show=False)
    plt.title(f"{model_name} - SHAP Summary Plot")  # Setting title for the summary plot
    plt.tight_layout()
    plt.close()

    # Calculate Mean Absolute SHAP Values for DataFrame display
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Absolute SHAP Value': mean_abs_shap_values
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

    return shap_html, shap_values, shap_df, X_train_preprocessed



# ------------------------------



# ------------------------------