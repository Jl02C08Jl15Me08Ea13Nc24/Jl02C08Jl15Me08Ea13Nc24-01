# EDA Page
#--------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------
import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import deploy

#--------------------------------------------------------------------------
# Streamlit Page Configuration
#--------------------------------------------------------------------------
st.set_page_config(page_title="Sales Prediction Analysis - EDA")
st.title('Exploratory Data Analysis (EDA)')
#--------------------------------------------------------------------------
# Overall EDA Interpretation
#--------------------------------------------------------------------------
with st.expander("📌 **Click here to view overall EDA insights**"):
    st.markdown("""

    ### Overview

    The Exploratory Data Analysis has highlighted several important aspects of the dataset, focusing on the distribution of features, their individual and collective relationships with the target variable `item_outlet_sales`, and inherent patterns that may affect machine learning model performance. This examination has revealed both expected and detailed insights that will guide the subsequent phases of data preprocessing, feature engineering, and model selection.

    ### Feature Analysis Insights

    - **Item Weight and Sales Relationship**: The weak positive correlation observed between `item_weight` and `item_outlet_sales` indicates that the weight of an item has minimal influence on its sales. This insight suggests that `item_weight` may have limited predictive power in modeling efforts, although its role should not be completely ignored without further feature importance analysis.

    - **Item Fat Content's Influence**: The distinction between 'Low Fat' and 'Regular' items and their respective sales distributions offers a view into consumer preferences or purchasing behavior. However, the lack of a strong correlation with sales indicates that while item fat content may affect sales, it is likely one of several factors influencing purchasing decisions.

    - **Visibility vs. Sales Paradox**: The negative correlation between `item_visibility` and `item_outlet_sales` contradicts the expectation that higher visibility should lead to higher sales. This unexpected finding might suggest that other factors, such as qualitative aspects of item placement or the interplay with item type and brand recognition, play a significant role.

    - **MRP's Strong Positive Correlation with Sales**: The moderate to strong correlation between `item_mrp` (Maximum Retail Price) and sales emphasizes the critical role of pricing strategy in driving sales volume. This relationship highlights the significance of considering item pricing as a key feature in predictive modeling.

    - **Establishment Year's Subtle Dynamics**: The variation in sales across different `outlet_establishment_year` categories, along with the statistical breakdown, suggests that the age or historical presence of an outlet could subtly influence its sales performance. These insights necessitate further investigation into temporal factors and their interaction with other store attributes.

    - **Outlet Size and Sales**: The subtle differences in sales distributions across 'Large', 'Medium', 'Small', and 'Missing' outlet sizes reveal the potential impact of outlet size on sales. This insight indicates the value of exploring outlet size, potentially in combination with location type and outlet type, to uncover synergistic effects on sales.

    - **Location Type and Outlet Type's Role**: The distinct sales patterns observed across different `outlet_location_type` and `outlet_type` categories highlight the importance of geographical and operational characteristics in shaping sales outcomes. These factors open up avenues for deeper exploration into how location and outlet operational model interact with consumer demographics and purchasing behavior.

    ### Concluding Observations

    The EDA has provided valuable insights into the dataset's structure, the interrelationships among features, and their collective impact on sales outcomes. While some findings conform to expectations, others provide unexpected revelations that challenge assumptions and invite further exploration. The insights gained form a foundational understanding that will inform the development of predictive models, emphasizing the need for detailed feature engineering and model complexity to capture the underlying patterns observed.

    The subsequent phases will utilize these insights to refine data preprocessing steps, enhance feature engineering, and carefully select and tune machine learning models to predict sales effectively. This structured approach, informed by thorough exploratory analysis, aims to uncover the subtleties within the data, guiding the predictive modeling process towards accurate and insightful outcomes.
    """)


#--------------------------------------------------------------------------
# Data Loading
#--------------------------------------------------------------------------
# Load File Structure Dictionary
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

# Load data
try:
    data = deploy.load_csv_data(FPATHS['data']['filtered']['sales_predictions_2023_eda.csv'])
    data_loaded = True
    st.success("Data loaded successfully! You can now explore the dataset and understand its characteristics. Use the options below to analyze different features of the dataset.")

except Exception as e:
    st.error(f"Failed to load data: {e}\nPlease try again by refreshing the page.")
    data_loaded = False

if data_loaded:
    #--------------------------------------------------------------------------
    # Feature Selection
    #--------------------------------------------------------------------------
    feature = st.selectbox('Select Feature to Analyze', data.columns)
    
    #--------------------------------------------------------------------------
    # Missing value strategies 
    #--------------------------------------------------------------------------
    # Treat 'outlet_establishment_year' as categorical
    if feature == 'outlet_establishment_year':
        feature_type = 'object'  
    else:
        feature_type = data[feature].dtype
    
    missing_values = data[feature].isnull().sum()
    
    if missing_values > 0:
        # Categorical
        if feature_type == 'object': 
            strategies = ['None', 'Missing', 'Mode']
        # Numerical
        else:  
            strategies = ['None', 'Mean', 'Median']
        
        missing_value_strategy = st.selectbox('Select Missing Value Strategy', strategies)
    else:
        missing_value_strategy = 'None'

    #--------------------------------------------------------------------------
    # plot_distribution function 
    #--------------------------------------------------------------------------
    def plot_distribution(data, feature):
        plot_color = '#636EFA' 
        
        if data[feature].dtype == 'object' and feature != 'outlet_establishment_year':
            color_sequence = px.colors.qualitative.T10
        else:
            color_sequence = [plot_color]

        if data[feature].dtype == 'object' or feature == 'outlet_establishment_year':
            fig = px.bar(data[feature].value_counts().reset_index(), x='index', y=feature,
                         labels={'index': feature, feature: 'Count'},
                         color='index' if feature != 'outlet_establishment_year' else None,
                         color_discrete_sequence=color_sequence)
            fig.update_layout(title=f'Bar Plot of {feature}', xaxis_title=feature, yaxis_title='Count',
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            if 'item_outlet_sales' in data.columns:
                fig2 = px.box(data, x=feature, y='item_outlet_sales', labels={'item_outlet_sales': 'Sales'},
                              color=feature if feature != 'outlet_establishment_year' else None,
                              color_discrete_sequence=color_sequence)
                fig2.update_layout(title=f'Boxplot of Sales by {feature}', xaxis_title=feature, yaxis_title='Sales',
                                   showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            fig = px.histogram(data, x=feature, marginal="box", hover_data=data.columns,
                               color_discrete_sequence=[plot_color])
            fig.update_traces(marker_line_color='white', marker_line_width=1)
            fig.update_layout(title=f'Distribution and Boxplot of {feature}', xaxis_title=feature, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        return plot_color

    #--------------------------------------------------------------------------
    # Handling Missing Values Based on Selected Strategy
    #--------------------------------------------------------------------------
    if st.button('Analyze'):
        original_missing_values = data[feature].isnull().sum()
        if missing_value_strategy == 'Mean':
            data[feature] = data[feature].fillna(data[feature].mean())
        elif missing_value_strategy == 'Median':
            data[feature] = data[feature].fillna(data[feature].median())
        elif missing_value_strategy == 'Mode':
            mode = data[feature].mode()[0]
            data[feature] = data[feature].fillna(mode)
        elif missing_value_strategy == 'Missing':
            data[feature] = data[feature].fillna('Missing')
    
        plot_color = plot_distribution(data, feature)


        # Correlation Analysis 
        if data[feature].dtype != 'object' and feature != 'outlet_establishment_year' and feature != 'item_outlet_sales' and 'item_outlet_sales' in data.columns:
            correlation = data[[feature, 'item_outlet_sales']].corr().iloc[0, 1]
            fig_corr = px.scatter(data, x=feature, y='item_outlet_sales',
                                  trendline='ols', trendline_color_override='red',
                                  color_discrete_sequence=[plot_color])
            fig_corr.update_traces(marker=dict(line=dict(color='white', width=1)))  
            fig_corr.update_layout(title=f"Correlation between {feature} and item_outlet_sales",
                                   xaxis_title=feature,
                                   yaxis_title='Item Outlet Sales')
            st.plotly_chart(fig_corr)

        # Correlation bar plot for target feature
        if feature == 'item_outlet_sales':
            numeric_features = data.select_dtypes(include=[np.number]).columns
            correlations = data[numeric_features].corr()['item_outlet_sales'].sort_values(ascending=False)
            correlations = correlations.drop('item_outlet_sales')
            
            # Colors based on correlation values
            colors = ['#ff6666' if c < 0 else '#636EFA' for c in correlations.values]
            
            fig_corr_bar = go.Figure(go.Bar(
                x=correlations.index,
                y=correlations.values,
                marker_color=colors
            ))
            
            fig_corr_bar.update_layout(
                title='Correlation of Numerical Features with Item Outlet Sales',
                xaxis_title='Features',
                yaxis_title='Correlation Coefficient',
                yaxis=dict(range=[-1, 1])  
            )
            
            st.plotly_chart(fig_corr_bar)

        #--------------------------------------------------------------------------
        # Data Insights
        #--------------------------------------------------------------------------
        st.subheader(f"Data Insights for {feature}")
        data_type = data[feature].dtype
        current_missing_values = data[feature].isnull().sum()
        unique_values = data[feature].nunique()
        st.text(f"- Data type: {data_type}")
        
        if original_missing_values > 0 and current_missing_values == 0:
            st.text(f"- Missing values: {current_missing_values} (Missing Value Strategy implemented: \"{missing_value_strategy}\")")
        else:
            st.text(f"- Missing values: {current_missing_values}")
        
        st.text(f"- Unique values: {unique_values}")
    
        if data[feature].dtype == 'object' or feature == 'outlet_establishment_year':
            categories = sorted(data[feature].unique(), key=lambda x: str(x))
            categories_str = ', '.join(map(str, categories))
            st.text(f"- Categories: {categories_str}")
            
            # Check for quasi-constant or constant
            value_counts = data[feature].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.98:
                st.text("- This feature is quasi-constant or constant.")
            
            # Check for high cardinality
            if unique_values > 10:
                st.text("- This feature has high cardinality.")
    
        if data[feature].dtype != 'object' and feature != 'outlet_establishment_year':
            skewness = data[feature].skew()
            kurtosis = data[feature].kurtosis()
            
            # Calculate outliers
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[feature] < lower_bound) | (data[feature] > upper_bound)).sum()
            
            st.text(f"- Skewness: {skewness:.3f}")
            if skewness > 0:
                st.text("  Direction: Right-skewed (positively skewed)")
            elif skewness < 0:
                st.text("  Direction: Left-skewed (negatively skewed)")
            else:
                st.text("  Direction: Symmetrical")
            
            st.text(f"- Kurtosis: {kurtosis:.3f}")
            if kurtosis > 3:
                st.text("  Type: Leptokurtic (heavy-tailed)")
            elif kurtosis < 3:
                st.text("  Type: Platykurtic (light-tailed)")
            else:
                st.text("  Type: Mesokurtic (normal distribution)")
            
            st.text(f"- Outliers: {outliers}")
            
            if feature != 'item_outlet_sales':
                correlation = data[[feature, 'item_outlet_sales']].corr().iloc[0, 1]
                st.text(f"- Correlation with 'item_outlet_sales': {correlation:.3f}")


