# Data Preparation Funtions
# -----------------------------------------
import pandas as pd


# -----------------------------------------
def calculate_outliers(column):
    """
    Calculate the number and percentage of outliers in a DataFrame column based on the IQR method.

    Parameters:
    - column (pd.Series): The DataFrame column to calculate outliers for.

    Returns:
    - tuple: Number of outliers, Percentage of outliers
    """
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    outliers_mask = ((column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR)))
    num_outliers = outliers_mask.sum()
    percent_outliers = (num_outliers / len(column)) * 100
    return num_outliers, percent_outliers

def summarize_data(df):
    """
    Summarize the descriptive statistics and outlier information for numeric columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to summarize.

    Returns:
    - pd.DataFrame: A summary DataFrame with descriptive statistics and outlier info.
    """
    summary_data = []
    for col in df.select_dtypes(include='number').columns:
        description = df[col].describe(percentiles=[.25, .5, .75])
        num_outliers, percent_outliers = calculate_outliers(df[col])
        summary_data.append({
            'Column': col,
            'Min': description['min'],
            '25%': description['25%'],
            '50%': description['50%'],  
            '75%': description['75%'],
            'Max': description['max'],
            'Mean': description['mean'],
            'STD': description['std'],
            'Outliers': num_outliers,
            'Outlier %': percent_outliers
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df[['Column', 'Min', '25%', '50%', '75%', 'Max', 'Mean', 'STD', 'Outliers', 'Outlier %']]
    return summary_df
