# eda funtions
# ------------------------------
# Imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticks
import matplotlib.colors as mcolors
from scipy.stats import kurtosis
from scipy.stats import skew


# ------------------------------
# Function: Format Ticks
def format_ticks(x, pos):
    # Ticks are formatted
    return '${:,.0f}'.format(x)

# ------------------------------
# Function: Formatter
def formatter(value):
    # Value is formatted
    return "${:,.2f}".format(value)

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
# Function: Dynamic Formatter Dollars
def dynamic_formatter_dollars(x, pos):
    # Appropriate unit is determined and value is formatted
    if x >= 1_000_000_000_000:
        new_x = x / 1_000_000_000_000
        return f"${new_x:.1f}T"
    elif x >= 1_000_000_000:
        new_x = x / 1_000_000_000
        return f"${new_x:.1f}B"
    elif x >= 1_000_000:
        new_x = x / 1_000_000
        return f"${new_x:.1f}M"
    elif x >= 1_000:
        new_x = x / 1_000
        return f"${new_x:.1f}K"
    else:
        return f"${x:.1f}"


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
def category_statistics(df, feature, target_feature):
    """"Calculates statistics for each category of a given feature."""
    # Get unique categories in the specified feature
    unique_categories = df[feature].unique()

    # Initialize an empty dictionary to store statistics
    category_stats = {}

    # Compute statistics for each category
    for category in unique_categories:
        subset = df[df[feature] == category]
        total_count = subset[target_feature].count()
        
        outliers_count = detect_outliers(subset, target_feature)
        outliers_percentage = (outliers_count / total_count) * 100 if total_count > 0 else 0     
   
        category_stats[category] = {
            'count': subset[target_feature].count(),
            'outliers': outliers_count,
            'outliers%': round(outliers_percentage, 2),
            'min': round(subset[target_feature].min(), 2),
            'Q1': round(subset[target_feature].quantile(0.25), 2),
            'median': round(subset[target_feature].median(), 2),
            'Q3': round(subset[target_feature].quantile(0.75), 2),
            'max': round(subset[target_feature].max(), 2),
            'mean': round(subset[target_feature].mean(), 2),
            'std': round(subset[target_feature].std(), 2), 
            'mode': subset[target_feature].mode()[0] if not subset[target_feature].mode().empty else None,  
            'range': round(subset[target_feature].max() - subset[target_feature].min(), 2)

        }

    # Create a DataFrame from the dictionary
    category_stats_df = pd.DataFrame(category_stats)

    return category_stats_df


# ------------------------------
def numerical_statistics(df, feature):
    """Calculates statistics for the numerical feature"""
    total_count = df[feature].count()
    
    nulls_count = df[feature].isna().sum()
    null_percentage = (nulls_count / total_count) * 100 if total_count > 0 else 0 
    
    outliers_count = detect_outliers(df, feature)
    outliers_percentage = (outliers_count / total_count) * 100 if total_count > 0 else 0 

    stats = {
        'count': total_count,
        'nulls': nulls_count,
        'nulls%': round(null_percentage, 2),
        'outliers': outliers_count,
        'outliers%': round(outliers_percentage, 2),
        'min': round(df[feature].min(), 4),
        'Q1': round(df[feature].quantile(0.25), 4),
        'median': round(df[feature].median(), 4),
        'Q3': round(df[feature].quantile(0.75), 4),
        'max': round(df[feature].max(), 4),
        'mean': round(df[feature].mean(), 4),
        'std': round(df[feature].std(), 4), 
        'mode': df[feature].mode()[0] if not df[feature].mode().empty else None,  
        'range': round(df[feature].max() - df[feature].min(), 4),
        'skewness': round(skew(df[feature].dropna()), 3),
        'kurtosis': round(kurtosis(df[feature].dropna()), 3)
    }

    # Create a DataFrame from the dictionary
    numeric_stats_df = pd.DataFrame(stats, index=[0])

    return numeric_stats_df


# ------------------------------
def dollar_category_statistics(df, feature, target_feature):
    """Calculates dollar-based statistics for each category of a given feature."""
    # Get unique categories in the specified feature
    unique_categories = df[feature].unique()

    # Initialize an empty dictionary to store statistics
    dollar_category_stats = {}

    # Compute statistics for each category
    for category in unique_categories:
        subset = df[df[feature] == category]
        total_count = subset[target_feature].count()
        
        outliers_count = detect_outliers(subset, target_feature)
        outliers_percentage = (outliers_count / total_count) * 100 if total_count > 0 else 0  

        
        dollar_category_stats[category] = {
            'count': total_count,
            'outliers': outliers_count,
            'outliers%': round(outliers_percentage, 2),
            'min': f"${round(subset[target_feature].min(), 2)}",
            'Q1': f"${round(subset[target_feature].quantile(0.25), 2)}",
            'median': f"${round(subset[target_feature].median(), 2)}",
            'Q3': f"${round(subset[target_feature].quantile(0.75), 2)}",
            'max': f"${round(subset[target_feature].max(), 2)}",
            'mean': f"${round(subset[target_feature].mean(), 2)}",
            'std': f"${round(subset[target_feature].std(), 2)}",
            'mode': f"${subset[target_feature].mode()[0]}" if not subset[target_feature].mode().empty else None,
            'range': f"${round(subset[target_feature].max() - subset[target_feature].min(), 2)}"
        }

    # Create a DataFrame from the dictionary
    dollar_category_stats_df = pd.DataFrame(dollar_category_stats)

    return dollar_category_stats_df


# ------------------------------
def check_kurtosis(df, feature):
    """Calculates and interprets the kurtosis based on its value."""
    
    kurt = kurtosis(df[feature].dropna())
    kurt = round(kurt, 3)
    if kurt == 0:
        return "Mesokurtic", kurt, "The distribution is similar to the normal distribution."
    elif kurt > 0:
        return "Leptokurtic", kurt, "The distribution has heavier tails and a sharper peak than the normal distribution."
    elif kurt < 0:
        return "Platykurtic", kurt, "The distribution has lighter tails and a flatter peak than the normal distribution."


# ------------------------------
def check_skewness(df, feature):
    """Calculates and interprets the skewness based on its value."""
    
    skewness = round(skew(df[feature].dropna()), 3)
    if skewness > 0:
        return "Positively skewed", skewness, "The distribution is positively skewed, meaning that the tail on the right side is longer or fatter than the left side."
    elif skewness < 0:
        return "Negatively skewed", skewness, "The distribution is negatively skewed, meaning that the tail on the left side is longer or fatter than the right side."
    else:
        return "Symmetric", skewness, "The distribution is symmetric."


# ------------------------------
def check_constant(df, feature):
    """Checks if a feature is constant or quasi-constant."""
    
    counts = df[feature].value_counts(normalize=True)
    nunique = df[feature].nunique()
    if counts.iloc[0] > 0.95:
        return "Quasi-constant", f"The feature is quasi-constant, meaning that more than 95% of the values are the same."
    elif nunique == 1:
        return "Constant", f"The feature is constant, meaning that all values are the same."
    else:
        return "Varied", f"The feature is not constant or quasi-constant, meaning that the values are varied as it has {nunique} unique values."


# ------------------------------
def check_cardinality(df, feature):
    """Checks if a feature has high cardinality."""
    
    nunique = df[feature].nunique()
    unique_values = ', '.join(sorted(df[feature].astype(str).unique()))
    if nunique > 20:
        return "High cardinality", f"The feature has high cardinality, meaning that it has more than 20 unique values."
    else:
        return "Low cardinality", f"The feature does not have high cardinality. The unique values are: ({unique_values})."


# ------------------------------
def analyze_feature(df, 
                    feature, 
                    target_feature= 'item_outlet_sales',
                    binwidth=None, sort_categories=True, sort_categories_vs=True,
                    replace_nulls_with=None, replacement_value=None, 
                    rotation=None, vs_rotation=None, 
                    save_filename_dist=None, save_filename_vs_target=None, save_filename_correlation=None, save_filename_corrbar=None):
    """Analyzes a feature by handling missing values, plotting distributions, and providing basic information. Also plots the feature against a target feature."""
    
    # Set custom palette
    custom_colors, custom_hatches = set_custom_palette()
    sns.set_palette(custom_colors)
    
    # Temporary data
    temp_df = df.copy()
    original_null_count = temp_df[feature].isna().sum()

    # Handle missing values and replacements
    replacement_info = f"- Missing values: {original_null_count}."
    if replace_nulls_with:
        if replace_nulls_with == 'mean' and pd.api.types.is_numeric_dtype(temp_df[feature]):
            replacement_value = temp_df[feature].mean()
        elif replace_nulls_with == 'median' and pd.api.types.is_numeric_dtype(temp_df[feature]):
            replacement_value = temp_df[feature].median()
        # Note: mode() might return multiple values if there's a tie, it will take the first one
        elif replace_nulls_with == 'mode' and pd.api.types.is_numeric_dtype(temp_df[feature]):
            replacement_value = temp_df[feature].mode()[0]
        temp_df[feature].fillna(replacement_value, inplace=True)
        replacement_info += f" Replaced with {replace_nulls_with}: {replacement_value}."
    else:
        replacement_info += " No replacement strategy implemented."

    # Plotting distribution based on data type
    # Numerical: plot histogram and boxplot
    if pd.api.types.is_numeric_dtype(temp_df[feature]):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [6, 1]})
        sns.histplot(data=temp_df, x=feature, binwidth=binwidth, ax=axes[0], kde=True, color=custom_colors[4])
        sns.boxplot(data=temp_df, x=feature, ax=axes[1], color=custom_colors[4])
        axes[0].set_xlabel("")
        axes[0].set_xticks([])
        
    # Categorical: plot countplot   
    else:
        fig, ax = plt.subplots(figsize=(10, 3))
        if sort_categories:
            order = sorted(temp_df[feature].unique())
        else:
            order = temp_df[feature].value_counts().index

        # Check if there are more than 5 categories
        if len(order) > 5:
            palette = [custom_colors[2]] * len(order)  
        else:
            palette = custom_colors 
            
        sns.countplot(data=temp_df, x=feature, ax=ax, order=order, palette=palette)
        if rotation:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
            
        total = len(temp_df[feature])
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', xytext=(0, 2), textcoords='offset points')
            ax.annotate(f'({height/total:.1%})', (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', xytext=(0, -10), textcoords='offset points')
        ax.yaxis.set_major_formatter(dynamic_formatter)
    
    fig.suptitle(f"Distribution of {feature}", fontweight='bold')
    if save_filename_dist:
        plt.savefig(save_filename_dist)
    plt.show()

    # Print basic information 
    print(f"- Data type: {temp_df[feature].dtype}")
    print(replacement_info)
    nunique = temp_df[feature].nunique()
    
    # Information if the feature is categorical
    if temp_df[feature].dtype == 'object':     

        # Check for constant or quasi-constant features
        const_type, interpretation = check_constant(temp_df, feature)
        print(f"- {interpretation}")
        
        # Check for high cardinality
        card_type, interpretation = check_cardinality(temp_df, feature)
        print(f"- {interpretation}")

        
    # Information if the feature is numerical:
    if pd.api.types.is_numeric_dtype(temp_df[feature]):
        
        # Check for constant or quasi-constant features
        const_type, interpretation = check_constant(temp_df, feature)
        print(f"- {interpretation}")

        # Check for kurtosis
        kurt_type, kurt, interpretation = check_kurtosis(temp_df, feature)
        print(f"- {kurt_type} Kurtosis of {kurt}.")
        print(f"  {interpretation}")
        
        # Check for skewness
        skew_type, skewness, interpretation = check_skewness(temp_df, feature)
        print(f"- {skew_type}: {skewness}.")
        print(f"  {interpretation}")
        
        # Check for outliers
        outlier_count = detect_outliers(temp_df, feature)
        print(f"- Outliers: {outlier_count}")

        # Get statistics for numerical features
        print(f"- Statistics:")
        numeric_stats_df = numerical_statistics(temp_df, feature)
        numeric_stats_df.index = ['']
        display(numeric_stats_df)
        print('\n')

        # Note:
        if feature == target_feature:
            correlation_barplot(temp_df, feature, target_feature, save_filename_corrbar=save_filename_corrbar)
            return

        # Plot numerical feature against target feature
        numeric_vs_target(temp_df, feature, target_feature, save_filename_vs_target=save_filename_vs_target)
        
        # Plot the correlation bar
        plot_correlation_bar(df, feature, target_feature, save_filename_correlation=save_filename_correlation)

        # Print the correlation value and interpretation
        r, interpretation = calculate_and_interpret_correlation(df, feature, target_feature)
        print(f"- {interpretation} of {r}.")

        
    # Information if the feature is categorical:  
    else:
        # Note:
        if feature == target_feature:
            correlation_barplot(temp_df, feature, target_feature, save_filename_corrbar=save_filename_corrbar)
            return
            
        # Plot categorical feature against target feature plot
        print('\n')
        categorical_vs_target(temp_df, feature, target_feature, 
                              vs_rotation=vs_rotation, sort_categories_vs=sort_categories_vs,
                              save_filename_vs_target=save_filename_vs_target)
        
        # Get statistics for categorical features
        print(f"- Statistics for Each Category:")
        dollar_category_stats_df = dollar_category_statistics(temp_df, feature, target_feature)
        display(dollar_category_stats_df.T)


# ------------------------------
def calculate_and_interpret_correlation(df, feature, target_feature):
    """Calculates and interprets the correlation between two features."""
    
    # Calculate the correlation
    corr = df[[feature, target_feature]].corr().round(3)
    r = corr.loc[feature, target_feature]

    def interpret_correlation(r):
        """Interprets the strength of a correlation based on its value."""
        if r == 0:
            return "No correlation"
        elif 0 < r < 0.3:
            return "Weak positive correlation"
        elif 0.3 <= r < 0.7:
            return "Moderate positive correlation"
        elif 0.7 <= r < 1:
            return "Strong positive correlation"
        elif r == 1:
            return "Perfect positive correlation"
        elif -0.3 < r < 0:
            return "Weak negative correlation"
        elif -0.7 < r <= -0.3:
            return "Moderate negative correlation"
        elif -1 < r <= -0.7:
            return "Strong negative correlation"
        elif r == -1:
            return "Perfect negative correlation"

    # Interpret the correlation
    interpretation = interpret_correlation(r)

    return r, interpretation


# ------------------------------
def numeric_vs_target(df, feature, target_feature,
                           figsize=(10,4), color=None, marker='o', linestyle='-',
                           save_filename_vs_target=None):
    """Plots a seaborn regplot with Pearson's correlation (r)"""
    
    # Set custom style
    sns.set(style="whitegrid")
    
    # Set custom palette
    custom_colors, custom_hatches = set_custom_palette()
    sns.set_palette(custom_colors)

    # Calculate and interpret the correlation
    r, interpretation = calculate_and_interpret_correlation(df, feature, target_feature)
   
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [6, 1]}, sharey=True)
 
    # Scatter plot
    scatter_kws={'ec':'white','lw':1,'alpha':0.8}
    sns.regplot(data=df, x=feature, y=target_feature, ax=axs[0], color=custom_colors[4], 
                line_kws={'color': 'red', 'linestyle': linestyle}, scatter_kws=scatter_kws)

    # Add the title with the correlation
    axs[0].set_title(f"{feature} vs. {target_feature} (r = {r}): {interpretation}", fontweight='bold')
    
    # Set the formatter for y-axis tick values
    axs[0].yaxis.set_major_formatter(dynamic_formatter_dollars)

    # Box plot
    sns.boxplot(y=target_feature, data=df, ax=axs[1], color=custom_colors[4], orient='v')
    axs[1].set_title('')  
    axs[1].set_ylabel('') 
    axs[1].tick_params(left=False)
    axs[1].set_xlabel(target_feature) 

    # Adjust the layout
    plt.tight_layout()

    # Save plot
    if save_filename_vs_target:
        plt.savefig(save_filename_vs_target)

    # Show the plot
    plt.show();


# ------------------------------ 
def plot_correlation_bar(df, feature, target_feature, save_filename_correlation=None):
    """Plots a correlation bar for the given correlation coefficient."""
    
    # Calculate the correlation
    r, _ = calculate_and_interpret_correlation(df, feature, target_feature)

    fig, ax = plt.subplots(figsize=(10, 0.2))

    # Create a gradient bar from -1 to 1
    gradient = np.linspace(-1, 1.5, 256)  
    gradient = np.vstack((gradient, gradient))  
    cmap = sns.color_palette("coolwarm_r", as_cmap=True) 
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=(-1, 1.5, -1, 1.5))

    # Add a marker for the correlation coefficient
    ax.plot([r, r], [-1, 1], color='blue', linewidth=2)

    # Add an annotation (arrow) at the top
    ax.annotate('', xy=(r, 1), xytext=(r, 1.5), arrowprops=dict(facecolor='blue', headwidth=40))

    # Add the correlation coefficient above the arrow
    ax.text(r, 4, f'r = {r:.3f}', ha='center', va='bottom', fontsize=9, color='blue')

    # Set the limits and labels of the x-axis
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1.5])  
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'], fontsize=9)  

    # Add the labels below the x-axis ticks
    ax.text(-1, -7, 'Strong negative correlation', ha='center', fontsize=8)
    ax.text(-0.5, -2, '', ha='center', fontsize=8)
    ax.text(0, -7, 'No correlation', ha='center', fontsize=8)
    ax.text(0.5, -2, '', ha='center', fontsize=8)
    ax.text(1, -7, 'Strong positive correlation', ha='center', fontsize=8)

    # Remove the y-axis
    ax.yaxis.set_visible(False)

    # Save plot
    if save_filename_correlation:
        plt.savefig(save_filename_correlation, bbox_inches='tight')

    plt.show();


# ------------------------------    
def categorical_vs_target(df, feature, target_feature, vs_rotation=None, sort_categories_vs=True, save_filename_vs_target=None):
    """Creates a boxplot of a categorical feature against a numeric target feature."""

    # Set custom style
    sns.set(style="whitegrid")
    
    # Set custom palette
    custom_colors, custom_hatches = set_custom_palette()
    sns.set_palette(custom_colors)
    
    plt.figure(figsize=(10,3))
    
    if sort_categories_vs:
        order = sorted(df[feature].unique(), key=lambda x: x if x.isdigit() else x)
    else:
        order = None 

    # Check if there are more than 5 categories
    if df[feature].nunique() > 5:
        color = custom_colors[2] 
        sns.boxplot(x=feature, y=target_feature, data=df, order=order, color=color) 
    else:
        sns.boxplot(x=feature, y=target_feature, data=df, order=order, palette=custom_colors)  
        
    plt.xticks(rotation=vs_rotation)
    plt.title(f"{feature} vs. {target_feature}", fontweight='bold')
    
    # Set the formatter for y-axis tick values
    plt.gca().yaxis.set_major_formatter(dynamic_formatter_dollars)
    
    # Save plot
    if save_filename_vs_target:
        plt.savefig(save_filename_vs_target)
        
    plt.show();


# ------------------------------   
def correlation_heatmap(df, y_rotation=None, x_rotation=None, save_filename_heatmap=None):
    """Creates a heatmap of the correlations between all pairs of numerical features in the DataFrame."""
    # Set the custom style
    sns.set(style="whitegrid")

    # Set custom palette
    custom_colors, custom_hatches = set_custom_palette()
    sns.set_palette(custom_colors)
    
    # Find correlations
    corr = df.corr(numeric_only=True)
    corr.round(3)
    
    # Set figure size
    plt.figure(figsize=(10, 3))

    # Create a heatmap 
    ax = sns.heatmap(corr, cmap=custom_colors, annot=True, annot_kws={"size": 10},
                     cbar_kws={"shrink": 0.9}, linewidths=0.5, linecolor='white')
    ax.set_title("Numerical Features Correlations")
    
    # Tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation, horizontalalignment='center', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=y_rotation, horizontalalignment='right', fontsize=10)
    
    # Save plot
    if save_filename_heatmap:
        plt.savefig(save_filename_heatmap)
        
    plt.show();


# ------------------------------   
def correlation_barplot(df, feature, target_feature, save_filename_corrbar=None):
    """Creates a bar plot of the correlations between the target feature and all other numerical features in the DataFrame."""
    # Calculate correlations with the target feature
    correlations = df.corr(numeric_only=True)[target_feature].sort_values()
    
    # Remove the correlation of the feature with itself
    if feature in correlations:
        correlations = correlations.drop(feature)
    
    
    # Create a bar plot
    plt.figure(figsize=(10, 3))
    ax = sns.barplot(x=correlations, y=correlations.index, palette="coolwarm_r")
    
    # Annotate correlation values
    for p in ax.patches:
        ax.annotate(format(p.get_width(), '.3f'), 
                    (p.get_width(), p.get_y() + p.get_height() / 2.0), 
                    ha = 'left', va = 'center', 
                    xytext = (1, 1), 
                    textcoords = 'offset points')
    
    # Set labels
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Numerical Features')
    ax.set_title(f'Correlation Coefficients of Numerical Features with {feature}')

    # Add padding to the x-axis limits
    xmin, xmax = plt.xlim()
    plt.xlim(xmin - 0.02, xmax + 0.02)
    
    # Save plot
    if save_filename_corrbar:
        plt.savefig(save_filename_corrbar)

    plt.show();

    # Create a DataFrame of correlations and display it
    correlations_df = pd.DataFrame(correlations).rename(columns={target_feature: 'Correlation with ' + target_feature})
    display(correlations_df)


# ------------------------------ 
def outliers_histogram_and_boxplot(df, feature, binwidth=None, save_filename=None, text_position=(0.95, 0.15)):
    # Set style
    sns.set(style="whitegrid")

    # Set custom palette
    custom_colors, custom_hatches = set_custom_palette()
    sns.set_palette(custom_colors)

    # Set up figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [6, 1]})

    # Detect outliers
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR)))
    outliers_count = outliers.sum()
    outliers_percentage = outliers_count / len(df) * 100

    # Create Histogram
    ax1 = sns.histplot(data=df, x=feature, binwidth=binwidth, alpha=1, ax=axes[0], hue=outliers, palette={True: 'red', False: custom_colors[4]}, legend=False)
    ax1.set_title(f"{feature} Outliers")
    ax1.set_xlabel("")
    ax1.set_xticks([])
    ax1.set_ylabel("count")

    # Create Box Plot
    ax2 = sns.boxplot(data=df, x=feature, ax=axes[1], color=custom_colors[4])
    ax2.set_ylabel("")

    # Add a text box with the count and percentage of outliers
    textstr = f'Outliers: {outliers_count} ({outliers_percentage:.2f}% of All Observations)'
    props = dict(boxstyle='round', facecolor='black', alpha=0.9, edgecolor='red')
    
    # Move the text box automatically where it does not cover the histogram
    ax1.text(*text_position, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props, color='white')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if save_filename:
        plt.savefig(save_filename)

    # Show the plot
    plt.show();
