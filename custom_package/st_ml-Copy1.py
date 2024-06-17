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
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline

def OLDst_feature_importances(model, feature_names, model_name='', feature_imp_decimals=3):
    # Check if the model is a pipeline and extract the estimator if it is
    if isinstance(model, Pipeline):
        model = model[-1] 

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
        styled_df = importances_df.style.format({'Importance': "{:." + str(feature_imp_decimals) + "f}"}).hide_index()

        
        return styled_df, fig
    else:
        raise ValueError("The model does not have feature importances or is not compatible.")

        
# ------------------------------
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline

def st_feature_importances(model, feature_names, model_name='', feature_imp_decimals=3):
    # Check if the model is a pipeline and extract the estimator if it is
    if isinstance(model, Pipeline):
        model = model[-1] 

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

        # Style the DataFrame for display and hide the index
        styled_df = importances_df.style.format({'Importance': "{:." + str(feature_imp_decimals) + "f}"}).hide_index()
        
        return styled_df, fig
    else:
        raise ValueError("The model does not have feature importances or is not compatible.")






# ------------------------------