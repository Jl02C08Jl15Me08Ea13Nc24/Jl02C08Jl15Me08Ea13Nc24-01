# ml funtions
# ------------------------------
# Imports

import re
import pandas as pd
import numpy as np
import joblib
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import shap
import lime
import lime.lime_tabular
from IPython.display import display, HTML, Image
from IPython.display import display_html
from html2image import Html2Image

from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import kurtosis
from scipy.stats import skew

import graphviz
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer




# ------------------------------
# Set custom palette globally
def set_custom_palette():
    """Sets custom color and pattern palette"""
    # Define custom color palette
    custom_colors = ["#cdf7ff", "#b2dcdd", "#8fbfcc", "#789fb9", "#667c9c"]
    
    # Define custom hatch pattern palette
    custom_hatches = ["//", "xx", "||", "..", "++"]

    return custom_colors, custom_hatches
    
    
# ------------------------------
# Function: Dynamic Formatter
def dynamic_formatter(x, pos):
    # Appropriate unit is determined and value is formatted
    if x >= 1_000_000_000_000:
        new_x = x / 1_000_000_000_000
        return f"{new_x:.1f}T"
    elif x >= 1_000_000_000:
        new_x = x / 1_000_000_000
        return f"{new_x:.1f}B"
    elif x >= 1_000_000:
        new_x = x / 1_000_000
        return f"{new_x:.1f}M"
    elif x >= 1_000:
        new_x = x / 1_000
        return f"{new_x:.1f}K"
    else:
        return f"{x:.1f}"
        
        
# ------------------------------
def detect_outliers(df, feature):
    """Detects outliers"""
    # Calculate the IQR of the feature
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    # Determine the outliers
    outliers = ((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))
    
    # Return the number of outliers
    return outliers.sum()


# ------------------------------
def imputation_strategy_info(df, feature, target_feature):
    # Temporary data
    temp_df = df.copy()
    # Count
    total_count = temp_df[feature].count()
    # Missing Values
    nulls_count = temp_df[feature].isna().sum()
    null_percentage = (nulls_count / total_count) * 100 if total_count > 0 else 0 

    # Check datatype
    if temp_df[feature].dtype in ['int64', 'float64']:
        # Outliers
        outliers_count = detect_outliers(temp_df, feature)
        outliers_percentage = (outliers_count / total_count) * 100 if total_count > 0 else 0  
        
        # Skewness and kurtosis
        skewness = round(skew(temp_df[feature].dropna()), 3)
        kurtosis_val = round(kurtosis(temp_df[feature].dropna()), 3)
        
        # Suggest an imputation strategy based on the skewness and presence of outliers
        if nulls_count == 0:
            imputation_strategy = None
        elif skewness > 0.5 or outliers_count > 0:
            imputation_strategy = 'median'
        else:
            imputation_strategy = 'mean'
            
        stats = {
            'dtype': temp_df[feature].dtype,
            'count': total_count,
            'nulls': nulls_count,
            'nulls%': round(null_percentage, 2),
            'outliers': outliers_count,
            'outliers%': round(outliers_percentage, 2),
            'median': round(temp_df[feature].median(), 4),
            'mean': round(temp_df[feature].mean(), 4),
            'mode': temp_df[feature].mode()[0] if not temp_df[feature].mode().empty else None,  
            'skewness': skewness,
            'kurtosis': kurtosis_val,
            'suggested_imputation': imputation_strategy
        }
    else:
       # For categorical features
        unique_categories = temp_df[feature].nunique()
        
        # Calculate most frequent category before replacing missing values
        most_frequent_category = temp_df[feature].mode()[0] if not temp_df[feature].mode().empty else None
        
        # Replace missing values with 'Missing'
        temp_df[feature].fillna('Missing', inplace=True)
        
        # Suggest mode imputation based on the original most frequent category
        imputation_strategy = most_frequent_category  if nulls_count > 0 else None
        stats = {
            'dtype': temp_df[feature].dtype,
            'count': total_count,
            'nulls': nulls_count,
            'nulls%': round(null_percentage, 2),
            'unique_categories': unique_categories,
            'most_frequent_category': most_frequent_category,  
            'suggested_imputation': imputation_strategy
        }
    # Create a DataFrame from the dictionary
    imputation_strategy_info_df = pd.DataFrame(stats, index=[0])
    imputation_strategy_info_df.index = ['']
    
    # Display the DataFrame
    display(imputation_strategy_info_df)


# ------------------------------
def suggest_imputation_strategy(df, 
                                feature, 
                                target_feature='item_outlet_sales',
                                binwidth=None, bbox_to_anchor=(0.79, 0.9), 
                                sort_categories=True, sort_categories_vs=True,
                                replace_nulls_with=None, replacement_value=None, 
                                rotation=None, vs_rotation=None, 
                                save_filename_imputation=None):

    # Set custom palette
    custom_colors, custom_hatches = set_custom_palette()
    sns.set_palette(custom_colors)
    # Temporary data
    temp_df = df.copy()
    
    # Plotting distribution based on data type
    # Numerical: plot histogram and boxplot
    if pd.api.types.is_numeric_dtype(temp_df[feature]):
        sns.set(style="whitegrid")
        feature_mean = round(temp_df[feature].mean(), 2)
        feature_median = round(temp_df[feature].median(), 2)
        fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [6, 1]})
        sns.histplot(data=temp_df, x=feature, binwidth=binwidth, ax=axes[0], kde=True, color=custom_colors[4])
        axes[0].axvline(feature_mean, color='red')
        axes[0].axvline(feature_median, color='green')
        axes[0].set_xlabel("")
        axes[0].set_xticks([])
        sns.boxplot(data=temp_df, x=feature, ax=axes[1], color=custom_colors[4])
        mean_line = mlines.Line2D([], [], color='red', label='Mean')
        median_line = mlines.Line2D([], [], color='green', label='Median')
        fig.legend(handles=[mean_line, median_line], loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=2)
        
    # Categorical: plot countplot   
    else:
        # Only modify the DataFrame copy used for plotting to preserve original null values
        plot_df = temp_df.copy()
        plot_df[feature].fillna('Missing', inplace=True)
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 3))
        if sort_categories:
            order = sorted(plot_df[feature].unique())
        else:
            order = plot_df[feature].value_counts().index
        
        if len(order) > 5:
            palette = [custom_colors[2]] * len(order)  
        else:
            palette = custom_colors
            
        sns.countplot(data=plot_df, x=feature, ax=ax, order=order, palette=palette)
        if rotation:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
            
        total = len(plot_df[feature])
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', xytext=(0, 2), textcoords='offset points')
            ax.annotate(f'({height/total:.1%})', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', xytext=(0, -10), textcoords='offset points')
        ax.yaxis.set_major_formatter(dynamic_formatter)
    
    fig.suptitle(f"Distribution of {feature} for Imputation Strategy", fontweight='bold')
    if save_filename_imputation:
        plt.savefig(save_filename_imputation)
    plt.show()

    # Print basic information by passing the original DataFrame
    return imputation_strategy_info(temp_df, feature, target_feature)


# ------------------------------
#import pandas as pd
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#from IPython.display import display, HTML
def calculate_vif(df, return_html=False):
    df_with_const = df.copy()
    df_with_const['const'] = 1
    
    vif_data = pd.DataFrame({
        "Feature": df.columns,
        "VIF": [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1] - 1)]
    })
    
    vif_data['Multicollinearity'] = pd.cut(vif_data['VIF'], 
                                           bins=[0, 1, 5, 10, float('inf')],
                                           labels=['None', 'Low', 'Moderate', 'High'],
                                           include_lowest=True, right=True)

    vif_data['Impact'] = vif_data['Multicollinearity'].map({
        'None': 'None',
        'Low': 'Low',
        'Moderate': 'Moderate',
        'High': 'High'
    })

    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    vif_summary = """
    VIF = 1: No multicollinearity. The predictor is not linearly related to other predictors.
    1 < VIF < 5: Low multicollinearity. Generally not a concern.
    5 ≤ VIF < 10: Moderate multicollinearity. May influence regression results, might need attention.
    VIF ≥ 10: High multicollinearity. Likely necessitates action (e.g., removing variables, applying regularization).
    """

    if return_html:
        from IPython.display import HTML
        return HTML(vif_data.to_html(index=False)), vif_summary
    else:
        return vif_data, vif_summary


# ------------------------------
# Pipeline Diagram
#import graphviz
#from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer

def create_pipeline_diagram(pipeline, diagram_filename='pipeline_diagram'):
    dot = graphviz.Digraph(format='png')
    dot.attr('node', shape='box', style='filled', fillcolor='#f0f8ff', fontname='Arial', fontsize='10')
    dot.attr(size='10,10')  
    
    # Main pipeline node
    dot.node('Pipeline', fontname='Arial Bold', fontsize='15') 
    
    # Traverse through the pipeline steps
    for step_name, step_proc in pipeline.steps:
        if isinstance(step_proc, ColumnTransformer):
            add_column_transformer(dot, 'Pipeline', step_proc, step_name)
        else:
            # For models or simple transformers
            model_label = f"{step_name.title()}\n{step_proc.__class__.__name__} ({get_params_label(step_proc)})"
            dot.node(step_name, model_label)
            dot.edge('Pipeline', step_name)

    dot.render(diagram_filename)
    return dot

def add_column_transformer(dot, parent_name, col_transformer, transformer_name):
    # Add the ColumnTransformer node
    dot.node(transformer_name, transformer_name.title(), fontname='Arial Bold', fontsize='12')
    dot.edge(parent_name, transformer_name)

    # Adding transformers in the ColumnTransformer
    for name, transformer, columns in col_transformer.transformers:
        transformer_label = f"{name.title()} Pipeline \n \n" + get_transformer_label(transformer)
        transformer_node = f"{transformer_name}_{name}"
        dot.node(transformer_node, transformer_label)
        dot.edge(transformer_name, transformer_node)

        # Add a node for features processed by this transformer
        features_label = "Features\\n\n" + "\n".join(columns)
        features_node = f"{transformer_node}_features"
        dot.node(features_node, features_label, 
                 shape='box', style='filled', fillcolor='#d4ebff',
                 fontcolor='black')
        dot.edge(transformer_node, features_node)

    # Handle the remainder separately
    if col_transformer.remainder != 'drop':
        remainder_label = f"Remainder\n{col_transformer.remainder}"
        dot.node(f"{transformer_name}_remainder", remainder_label)
        dot.edge(transformer_name, f"{transformer_name}_remainder")

def get_transformer_label(transformer):
    """Generate labels for transformers with their parameters."""
    if isinstance(transformer, Pipeline):
        return "\n \n".join([f"{step[1].__class__.__name__} ({get_params_label(step[1])})" for step in transformer.steps])
    else:
        return f"{transformer.__class__.__name__} ({get_params_label(transformer)})"

def get_params_label(estimator):
    """Helper to format parameters of the estimators."""
    params = estimator.get_params()
    param_list = []
    for key, value in params.items():
        if key == 'categories':
            if isinstance(value, list) and all(isinstance(i, list) for i in value):
                categories_str = "\ncategories=[" + ",\n ".join(["[" + ", ".join(map(str, c)) + "]" for c in value]) + "]"
            else:
                categories_str = f"\ncategories={value}" 
            param_list.append(categories_str)
        elif key in ['strategy', 'fill_value', 'drop', 'handle_unknown']:
            param_list.append(f"\n{key}: {value}")
    return ", ".join(param_list)  



# ------------------------------
#import pandas as pd
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from IPython.display import display
#from joblib import dump
def model_metrics(model, X_train, y_train, X_test=None, y_test=None,
                  model_name='', model_filename=None,
                  metric_decimals=3):
    # Generate predictions
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'Label': model_name + ' Training Data',
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': mean_squared_error(y_train, y_train_pred, squared=False),
        'R^2': r2_score(y_train, y_train_pred)
    }

    metrics = [train_metrics]

    # Check if test data is provided
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_metrics = {
            'Label': model_name + ' Test Data',
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': mean_squared_error(y_test, y_test_pred, squared=False),
            'R^2': r2_score(y_test, y_test_pred)
        }
        metrics.append(test_metrics)

    # Format and display results as a DataFrame
    results_df = pd.DataFrame(metrics)
    results_df = results_df.set_index('Label')
    results_df.index.name = None
    display(results_df.style.set_caption(model_name + ' - Training vs. Testing Performance').format("{:." + str(metric_decimals) + "f}"))

    # Saving model
    if model_filename:
        dump(model, model_filename)


# ------------------------------
#import numpy as np
#import pandas as pd
#from IPython.display import display
#import matplotlib.pyplot as plt
#import seaborn as sns

def error_analysis(model, X_train, y_train, X_test=None, y_test=None, 
                   model_name='', error_decimals=3, 
                   abs_error_xlim=None, perc_error_xlim=None,
                   save_filename_abs_errors=None):
    # Generate predictions for training data
    y_train_pred = model.predict(X_train)
    # Calculate absolute errors for training data
    abs_errors_train = np.abs(y_train - y_train_pred)
    # Calculate percentage errors for training data
    pct_errors_train = abs_errors_train / y_train * 100
    # Calculate various statistics for the training data
    train_summary = {
        'Training Data': {
            'Mean Absolute Error': np.mean(abs_errors_train),
            'Median Absolute Error': np.median(abs_errors_train),
            'Standard Deviation of Absolute Errors': np.std(abs_errors_train),
            'Min Absolute Error': np.min(abs_errors_train),
            'Max Absolute Error': np.max(abs_errors_train),
        },
        '- % -': {
            'Mean Absolute Error': np.mean(pct_errors_train),
            'Median Absolute Error': np.median(pct_errors_train),
            'Standard Deviation of Absolute Errors': np.std(pct_errors_train),
            'Min Absolute Error': np.min(pct_errors_train),
            'Max Absolute Error': np.max(pct_errors_train),
        }
    }
    # Create a DataFrame for the training data summary
    train_df = pd.DataFrame(train_summary)
    # Check if test data is provided
    if X_test is not None and y_test is not None:
        # Generate predictions for test data
        y_test_pred = model.predict(X_test)
        # Calculate absolute errors for test data
        abs_errors_test = np.abs(y_test - y_test_pred)
        # Calculate percentage errors for test data
        pct_errors_test = abs_errors_test / y_test * 100
        # Calculate various statistics for the test data
        test_summary = {
            'Test Data': {
                'Mean Absolute Error': np.mean(abs_errors_test),
                'Median Absolute Error': np.median(abs_errors_test),
                'Standard Deviation of Absolute Errors': np.std(abs_errors_test),
                'Min Absolute Error': np.min(abs_errors_test),
                'Max Absolute Error': np.max(abs_errors_test),
            },
            '- % -': {
                'Mean Absolute Error': np.mean(pct_errors_test),
                'Median Absolute Error': np.median(pct_errors_test),
                'Standard Deviation of Absolute Errors': np.std(pct_errors_test),
                'Min Absolute Error': np.min(pct_errors_test),
                'Max Absolute Error': np.max(pct_errors_test),
            }
        }
        # Create a DataFrame for the test data summary
        test_df = pd.DataFrame(test_summary)
        # Concatenate the training and test DataFrames
        summary_df = pd.concat([train_df, test_df], axis=1)
        # Display the DataFrame with the specified number of decimal places
        display(summary_df.style.set_caption(model_name + ' - Training vs. Testing Absolute Errors').format("{:." + str(error_decimals) + "f}"))

    else:
        # If no test data is provided, just display the training data summary
        display(train_df.style.set_caption(model_name + ' - Training Absolute Errors').format("{:." + str(error_decimals) + "f}"))

    print('\n')
    
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))

    # Plot histogram of absolute errors on the left
    sns.histplot(abs_errors_train, bins=30, kde=False, ax=axs[0], label='Train') 
    axs[0].set_xlabel('Absolute Errors')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'{model_name} - Frequency and Density of Absolute Errors')
    if abs_error_xlim:  
        axs[0].set_xlim(right=abs_error_xlim)

    # Plot density plot of percentage errors on the right
    sns.kdeplot(pct_errors_train, ax=axs[1], label='Train') 
    axs[1].set_xlabel('Percentage Errors')
    axs[1].set_ylabel('Density')
    #axs[1].set_title(f'{model_name} - Percentage Errors Density')
    if perc_error_xlim: 
        axs[1].set_xlim(right=perc_error_xlim)

    # Check if test data is provided
    if X_test is not None and y_test is not None:
        # Plot histogram of absolute errors on the left
        sns.histplot(abs_errors_test, bins=30, kde=False, ax=axs[0], color='orange', label='Test')  
        #axs[0].set_title(f'{model_name} - Frequency of Train & Test Absolute Errors')
        if abs_error_xlim: 
            axs[0].set_xlim(right=abs_error_xlim)

        # Plot density plot of percentage errors on the right
        sns.kdeplot(pct_errors_test, ax=axs[1], color='orange', label='Test') 
        #axs[1].set_title(f'{model_name} - Density of Train & Test Absolute Errors')
        if perc_error_xlim:  
            axs[1].set_xlim(right=perc_error_xlim)

    # Add legends
    axs[0].legend()
    axs[1].legend()
    # Display the plots
    plt.tight_layout
    # Adjust the space around the plot
    plt.subplots_adjust(bottom=0.17)
    # Save plot if a filename is provided
    if save_filename_abs_errors:
        plt.savefig(save_filename_abs_errors)
    plt.show();


# ------------------------------
#from sklearn.inspection import permutation_importance
#import pandas as pd
#from IPython.display import display
def df_permutation_importances(model, X, y,
                               n_repeats=30, random_state=42, 
                               model_name='', 
                               display_df=False, perm_decimals=3):
    # Calculate permutation importances
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    importances = result.importances_mean

    # Create a DataFrame
    permutations_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    permutations_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Display the DataFrame with formatted decimal places
    if display_df:
        formatted_df = permutations_df.style.format({'Importance': f"{{:.{perm_decimals}f}}"})
        display(formatted_df.set_caption(f'{model_name} - Permutation Importances').hide())

    return permutations_df


# ------------------------------
#import matplotlib.pyplot as plt
#import seaborn as sns
def plot_permutation_importances(permutations_df, top_n=20,
                                 model_name='', save_filename_permutations=None, 
                                 perm_decimals=3):
    """Plots the permutation importances of a model."""
    
    # Sort and select top_n importances for plotting
    top_n = min(top_n, len(permutations_df))
    top_perm_importances = permutations_df.sort_values('Importance', ascending=False).head(top_n)

    # Plotting
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x='Importance', y='Feature', data=top_perm_importances, palette="viridis")
    plt.title(f'{model_name} - Top {top_n} Permutation Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    # Add annotations to each bar with controlled decimal precision
    for p in ax.patches:
        ax.annotate(format(p.get_width(), f'.{perm_decimals}f'), 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', 
                    xytext=(5, 0), textcoords='offset points')
        
    # Adjust the space around the plot
    plt.subplots_adjust(left=0.25, bottom=0.15)
    
    # Save the plot if a filename is provided
    if save_filename_permutations:
        plt.savefig(save_filename_permutations)
        
    plt.show();


# ------------------------------
#import pandas as pd
#from IPython.display import display
def df_coeffs(model, feature_names, 
              model_name='', 
              display_df=False, 
              coeff_decimals=3):
    # Check if the model is a pipeline
    if isinstance(model, Pipeline):
        # Extract the LinearRegression model from the pipeline
        model = model.named_steps['linearregression']
        
    if hasattr(model, 'coef_'):
        coeffs = pd.Series(model.coef_, index=feature_names)
        coeffs_df = pd.DataFrame(coeffs).reset_index()
        coeffs_df.columns = ['Feature', 'Coefficient']
        coeffs_df['abs_coefficient'] = coeffs_df['Coefficient'].abs()
        coeffs_df = coeffs_df.sort_values('abs_coefficient', ascending=False).drop(columns='abs_coefficient')

        # Display the DataFrame
        if display_df:
            formatted_df = coeffs_df.style.format({'Coefficient': f"{{:.{coeff_decimals}f}}"})
            display(formatted_df.hide().set_caption(f'{model_name} - Feature Coefficients'))

        # Return the DataFrame without formatting the 'Coefficient' column
        return coeffs_df
    else:
        raise ValueError("Model does not have coefficients or is not compatible.")


# ------------------------------
#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
def plot_coeffs(coeffs_df, 
                top_n=20, 
                model_name='', save_filename_coeff=None, 
                coeff_decimals=3):
    """Plots the coefficients of a model provided as a DataFrame."""
    sns.set(style="whitegrid")

    # Check the DataFrame isn't empty and properly formatted
    if coeffs_df.empty or 'Coefficient' not in coeffs_df.columns:
        raise ValueError("DataFrame of coefficients is empty or improperly formatted.")

    # Convert coefficients to float for plotting
    coeffs_df['Coefficient'] = pd.to_numeric(coeffs_df['Coefficient'], errors='coerce')

    # If top_n is greater than the number of coefficients, set it to the number of coefficients
    top_n = min(top_n, len(coeffs_df))

    # Select the top_n coefficients
    top_coeffs = coeffs_df.head(top_n)

    # Create a bar plot of the coefficients
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x=top_coeffs['Coefficient'].abs(), y=top_coeffs['Feature'], palette="Blues_r")

    # Color bars based on the sign of the coefficients
    for patch, value in zip(ax.patches, top_coeffs['Coefficient']):
        patch.set_facecolor('#e5a28a' if value < 0 else '#9ab4ef')

    # Annotate correlation values with controlled decimal precision
    for i, (value, feature) in enumerate(zip(top_coeffs['Coefficient'], top_coeffs['Feature'])):
        ax.text(abs(value) + 0.05 * abs(value), i, f'{value:.{coeff_decimals}f}', color='black', ha="left", va="center")

    # Set the title and labels
    plt.title(f'{model_name} - Top {top_n} Feature Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')

    # Adjust the space around the plot
    plt.subplots_adjust(left=0.25, bottom=0.15)

    # Save plot if a filename is provided
    if save_filename_coeff:
        plt.savefig(save_filename_coeff)
    
    # Show the plot
    plt.show();


# ------------------------------
#import pandas as pd
#from IPython.display import display
#from sklearn.pipeline import Pipeline

def df_feature_importances(model, feature_names, display_df=False, feature_imp_decimals=3):
    # Check if the model is a pipeline and access the last step (estimator)
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]  # Accessing the last step which is the estimator
    else:
        estimator = model

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importances_df.sort_values(by='Importance', ascending=False, inplace=True)

        # Display the DataFrame with formatted decimal places
        if display_df:
            formatted_df = importances_df.style.format({'Importance': f"{{:.{feature_imp_decimals}f}}"})
            display(formatted_df.set_caption(f'{estimator.__class__.__name__} - Feature Importances').hide())

        return importances_df
    else:
        raise ValueError("The model does not have feature importances or is not compatible.")


# ------------------------------
#matplotlib.pyplot as plt
#import seaborn as sns
def plot_feature_importances(importances_df, top_n=20, 
                             model_name='', save_filename_feature_imp=None,
                             feature_imp_decimals=3):
    """Plots the feature importances of a model provided as a DataFrame."""
    # Check the DataFrame isn't empty and properly formatted
    if importances_df.empty or 'Importance' not in importances_df.columns:
        raise ValueError("DataFrame of feature importances is empty or improperly formatted.")

    # Select top_n features
    top_n = min(top_n, len(importances_df))
    top_features = importances_df.sort_values('Importance', ascending=False).head(top_n)
   
    # Plotting
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x='Importance', y='Feature', data=top_features, palette="viridis")
    plt.title(f'{model_name} - Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    # Add annotations to each bar with controlled decimal precision
    for p in ax.patches:
        ax.annotate(format(p.get_width(), f'.{feature_imp_decimals}f'), 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='left', va='center', 
                    xytext=(5, 0), textcoords='offset points')

    # Adjust the space around the plot
    plt.subplots_adjust(left=0.25, bottom=0.15)
    
    # Save the plot if a filename is provided
    if save_filename_feature_imp:
        plt.savefig(save_filename_feature_imp)
    plt.show();


# ------------------------------
def get_model_from_pipeline(pipeline):
    if hasattr(pipeline, 'steps') and len(pipeline.steps) > 0:
        _, model = pipeline.steps[-1]
        return model
    else:
        return pipeline  


# ------------------------------
# dataframes side by side
def display_side_by_side(titled_dataframes):
    html_str = '<div style="display:flex;flex-direction:row;align-items:flex-start;justify-content:center;gap:20px;width:100%;">'
    for title, df in titled_dataframes:
        # Each DataFrame and title is wrapped in a <div> for better control
        html_str += f'<div><h3 style="text-align: center">{title}</h3>{df.to_html(index=False)}</div>'
    html_str += '</div>'
    display_html(html_str, raw=True)


def single_model_evaluation(model, preprocessor, X_train, y_train, X_test=None, y_test=None, 
                            top_n=20,
                            model_name='', 
                            model_filename=None, 
                            coeffs=False, perms=False, features_imp=False,
                            display_features_df=True,
                            metric_decimals=3,coeff_decimals=3,perm_decimals=3,feature_imp_decimals=3,error_decimals=3,
                            abs_error_xlim=None, perc_error_xlim=None,
                            save_filename_coeff=None, 
                            save_filename_permutations=None,
                            save_filename_feature_imp=None,
                            save_filename_abs_errors=None,
                           ):
    
    # Model performance metrics
    #print(f"Evaluating {model_name} - Model Metrics:")
    model_metrics(model, X_train, y_train, X_test, y_test, 
                  model_name=model_name, 
                  model_filename=model_filename,
                  metric_decimals=metric_decimals)
    print('\n')  
  
    # Absolute Error analysis
    #print(f"Evaluating {model_name} - Absolute Errors:")
    error_analysis(model, X_train, y_train, X_test, y_test, 
                   model_name=model_name,
                   error_decimals=error_decimals, 
                   abs_error_xlim=abs_error_xlim,
                   perc_error_xlim=perc_error_xlim,
                   save_filename_abs_errors=save_filename_abs_errors)
    print('\n') 
    
    # Coefficient analysis for linear models
    if coeffs :
        #print(f"Evaluating {model_name} - Coefficient Analysis:")
        feature_names_after_preprocessing = preprocessor.get_feature_names_out()
        coeffs_df = df_coeffs(model, feature_names_after_preprocessing, 
                              model_name=model_name, 
                              display_df=False, 
                              coeff_decimals=coeff_decimals)
        plot_coeffs(coeffs_df, top_n=top_n, 
                    model_name=model_name, 
                    save_filename_coeff=save_filename_coeff,
                    coeff_decimals=coeff_decimals)
        print('\n')

    # Permutation importances for models that support them
    if perms:
        #print(f"Evaluating {model_name} - Permutation Importances:")
        permutations_df = df_permutation_importances(model, 
                                                     X_train, y_train, 
                                                     model_name=model_name, 
                                                     display_df=False, 
                                                     perm_decimals=perm_decimals)
        plot_permutation_importances(permutations_df, 
                                     top_n=top_n, 
                                     model_name=model_name, 
                                     save_filename_permutations=save_filename_permutations,
                                     perm_decimals=perm_decimals)
        print('\n')

    # Feature importances for tree-based models
    if features_imp:
        #print(f"Evaluating {model_name} - Feature Importances:")
        # Get the feature names after transformation
        feature_names_after_preprocessing = preprocessor.get_feature_names_out()
        importances_df = df_feature_importances(model, feature_names_after_preprocessing, 
                                                display_df=False, 
                                                feature_imp_decimals=feature_imp_decimals)
        plot_feature_importances(importances_df, top_n=top_n, 
                                 model_name=model_name, 
                                 save_filename_feature_imp=save_filename_feature_imp,
                                 feature_imp_decimals=feature_imp_decimals)
        print('\n')

    # Display DataFrames side by side
    if (features_imp and perms):
        display_side_by_side([("Feature Importances", importances_df), 
                              ("Permutation Importances", permutations_df)])
    
    if (coeffs and perms):
        display_side_by_side([("Coefficients", coeffs_df), 
                              ("Permutation Importances", permutations_df)])


# ------------------------------
def evaluate_multiple_models(best_models, preprocessor, X_train, y_train, X_test, y_test,
                             coeffs=False, perms=False, features_imp=False,
                             metric_decimals=3, coeff_decimals=3, perm_decimals=3, feature_imp_decimals=3, error_decimals=3,
                             abs_error_xlim=None, perc_error_xlim=None
                            ):

    model_counter = 1  # Initialize a counter to create unique names for saving outputs

    for name, model in best_models:
        # Define unique identifiers for this model's output
        unique_model_name = f"{name} Tuned Model {model_counter}"
        model_filename = f"models/ml/{name.lower().replace(' ', '_')}_tuned_model_{model_counter}.joblib"
        save_filename_coeff = f"images/ml/{name.lower().replace(' ', '_')}_tuned_model_{model_counter}_coefficients.png"
        save_filename_permutations = f"images/ml/{name.lower().replace(' ', '_')}_tuned_model_{model_counter}_permutations.png"
        save_filename_feature_imp = f"images/ml/{name.lower().replace(' ', '_')}_tuned_model_{model_counter}_features_imp.png"
        save_filename_abs_errors = f"images/ml/{name.lower().replace(' ', '_')}_tuned_model_{model_counter}_abs_error.png"

        print('\n')
        print(f"Evaluating {unique_model_name}")

        # Evaluate the model with the specific settings
        single_model_evaluation(
            model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            top_n=20,
            coeffs=coeffs,
            perms=perms,
            features_imp=features_imp,
            model_name=unique_model_name,
            model_filename=model_filename,
            display_features_df=True,
            metric_decimals=metric_decimals, 
            coeff_decimals=coeff_decimals, 
            perm_decimals=perm_decimals, 
            feature_imp_decimals=feature_imp_decimals,
            error_decimals=error_decimals,
            abs_error_xlim=abs_error_xlim, perc_error_xlim=perc_error_xlim,
            save_filename_coeff=save_filename_coeff,
            save_filename_permutations=save_filename_permutations,
            save_filename_feature_imp=save_filename_feature_imp,
            save_filename_abs_errors= save_filename_abs_errors,
        )

        model_counter += 1


# ------------------------------
#import shap
#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np
#import joblib
#from IPython.display import display, HTML, Image

def generate_shap_explainer(model, model_name, preprocessor, X_train, shap_decimals=2, save_filename_shap_hbar=None, save_filename_shap_forceplot=None):
    # Preprocess the training data
    X_train_preprocessed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)

    # Extract the model from the pipeline if it's part of one
    if 'xgb_regressor' in model.named_steps:
        xgb_model_from_pipeline = model.named_steps['xgb_regressor']
    else:
        xgb_model_from_pipeline = model
    
    explainer = shap.Explainer(xgb_model_from_pipeline, X_train_preprocessed_df)
    shap_values = explainer(X_train_preprocessed_df)

    # SHAP Force Plot
    print(f'{model_name} - SHAP Force Plot')
    plt.figure(figsize=(12, 4))  
    shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_train_preprocessed_df.iloc[0,:],
                    show=False, matplotlib=True)
    plt.title(f'{model_name} - SHAP Force Plot') 
    if save_filename_shap_forceplot:
        plt.savefig(save_filename_shap_forceplot)
        plt.close()
        display(Image(save_filename_shap_forceplot)) 
    else:
        plt.show()
    plt.clf()
    print('\n')

    # SHAP Summary Bar Plot
    shap.summary_plot(shap_values.values, X_train_preprocessed_df, plot_type="bar", plot_size=(10, 4), show=False)
    plt.title(f'{model_name} - SHAP Summary Plot')
    if save_filename_shap_hbar:
        plt.savefig(save_filename_shap_hbar)
    plt.show()
    print('\n')

    # Mean Absolute SHAP Values DataFrame
    mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
    mean_abs_shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Absolute SHAP Value': mean_abs_shap_values
    }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

    formatted_mean_abs_shap_df = mean_abs_shap_df.style.format({'Mean Absolute SHAP Value': f"{{:.{shap_decimals}f}}"})
    display(formatted_mean_abs_shap_df.hide().set_caption(f'{model_name} - SHAP Values'))

    return formatted_mean_abs_shap_df

    
# ------------------------------
#from IPython.display import display, HTML, Image
#import pandas as pd
#from html2image import Html2Image
def generate_lime_explainer(X_train_processed_path, preprocessor_path, model_path, i, model_name='Model'):
    # Importing required libraries here
    import joblib
    import lime.lime_tabular

    # Load the data and preprocessors
    X_train_processed = joblib.load(X_train_processed_path)
    preprocessor = joblib.load(preprocessor_path)

    # Load the best model
    best_model = joblib.load(model_path)

    # Prediction function to integrate the model with LIME
    def model_predict(data):
        return best_model.named_steps[best_model.steps[-1][0]].predict(data)

    # Create a Lime Tabular Explainer using preprocessed data as training data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_processed.to_numpy(),  
        feature_names=preprocessor.get_feature_names_out(),  
        class_names=['Sales'],
        mode='regression'
    )

    # Choose an instance to explain
    instance_to_explain = X_train_processed.iloc[i].to_numpy().reshape(1, -1) 

    print(f'{model_name} - LIME Explainer')
    print('\n')

    # Generate LIME explanation
    exp = explainer.explain_instance(instance_to_explain.ravel(), model_predict, num_features=5)

    # Save the explanation as an HTML file
    with open('images/ml/lime_explanation.html', 'w', encoding='utf-8') as f:
        f.write(exp.as_html())

    # Screenshot, load and display the HTML file as PNG
    hti = Html2Image(output_path='images/ml/') 
    hti.screenshot(html_file='images/ml/lime_explanation.html', save_as='lime_explanation.png', size=(950, 250))  
    display(Image(filename='images/ml/lime_explanation.png')) 
    
    # Extracting explanation data for the DataFrame
    exp_data = exp.as_list() 
    left_conditions = []
    features = []
    values = []
    weights = []
    for item in exp_data:
        parts = re.match(r"(([\d\.-]+)? < )?(.+) <= ([\d\.-]+)", item[0])
        if parts:
            left_conditions.append(parts.group(1) if parts.group(1) else None)
            features.append(parts.group(3).strip())
            values.append(f"<= {parts.group(4).strip()}")
            weights.append(item[1])
        else:
            left_conditions.append(None)
            features.append(item[0])
            values.append(None)
            weights.append(item[1])

    lime_df = pd.DataFrame({
        'Left Condition': left_conditions,
        'Feature': features,
        'Value': values,
        'Weight': weights
    })

    # Additional Data
    intercept = exp.intercept[0]
    predicted_value = np.round(model_predict(instance_to_explain)[0], 2)

    model_info_df = pd.DataFrame({
        "Description": ["Intercept", "Predicted value"],
        "Value": [intercept, predicted_value]
    })

    # Display the dataframes
    display(lime_df.style.hide().set_caption(f'{model_name} - LIME Explainer'))
    display(model_info_df.style.hide().format({'Value': "{:.2f}"}).set_caption('Model Information'))

# ------------------------------

