# EDA

#--------------------------------------------------------------------------
# IMPORTS
#--------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib, json, os, sys
import plotly.express as px
import plotly.graph_objects as go

# Custom Functions
sys.path.append(os.path.abspath("/"))
from custom_package import stfns

#--------------------------------------------------------------------------
# TITLES
#--------------------------------------------------------------------------
# Set the title of the browser tab
st.set_page_config(page_title="IMDB Movie Analysis - EDA")

#--------------------------------------------------------------------------
# DATA
#--------------------------------------------------------------------------
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

df_eda = stfns.load_Xy_data(FPATHS['data']['filtered']['eda_plots'])
df_genres = stfns.load_csv_data(FPATHS['data']['filtered']['clean_genres_revenue'])

#--------------------------------------------------------------------------
# PLOTS SECTION
#--------------------------------------------------------------------------
# Manually defined colors 
def get_MPAA_color_map():
    return {
        'G': '#003f5c',
        'PG': '#58508d',
        'PG-13': '#bc5090',
        'R': '#ff6361'
    }

def get_binary_color_map():
    return {
        False: '#bc5090', 
        True: '#003f5c'
    }
    
def get_movie_length_color_map():
    return {
        'Long Movies': '#bc5090',  
        'Short Movies': '#003f5c'
    }
    
#--------------------------------------------------------------------------
def display_financial_data():
    # Calculate boolean for valid financial information
    df_eda['has_valid_fin_info'] = (df_eda['budget'] > 0) | (df_eda['revenue'] > 0)

    # Create figure for valid financial information
    fig = px.histogram(df_eda, x='has_valid_fin_info', title='Movies with Valid Financial Information',
                       labels={'has_valid_fin_info': 'Has Valid Financial Information'}, 
                       text_auto=True, color='has_valid_fin_info',
                       color_discrete_map=get_binary_color_map())
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, yaxis_tickformat=',.2s')
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_budget_distribution():
    # Filter to include movies with specific MPAA ratings and budget/revenue > 0
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='budget', marginal='box',
                       title='Budget Distribution',
                       labels={'budget': 'Budget'},
                       opacity=0.6, nbins=50)
                       
    fig.update_xaxes(title="Budget")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_mpaa_distribution():
    # Filter to include movies with specific MPAA ratings and budget/revenue > 0
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                                   (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    
    # Calculate the count for each MPAA rating
    mpaa_counts = df_filtered['certification'].value_counts().reset_index()
    mpaa_counts.columns = ['certification', 'count']
    
    # Ensure the order of MPAA ratings
    mpaa_order = ['G', 'PG', 'PG-13', 'R']
    mpaa_counts = mpaa_counts.set_index('certification').reindex(mpaa_order).reset_index()
    
    # Create bar plot for MPAA ratings
    fig = px.bar(mpaa_counts, x='certification', y='count',
                 title='Number of Movies per MPAA Rating',
                 labels={'count': 'Count', 'certification': 'MPAA Rating'},
                 text_auto=True,
                 color='certification',
                 color_discrete_map=get_MPAA_color_map())
    
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_title="MPAA Rating", 
                      yaxis_title="Count",
                      showlegend=False)
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_average_budget_per_mpaa():
    # Filter to include movies with specific MPAA ratings and budget/revenue > 0
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]

    # Calculate average budget
    avg_budget = df_filtered.groupby('certification', as_index=False)['budget'].mean()

    # Create bar plot for average budget
    fig = px.bar(avg_budget, x='certification', y='budget',
                 title='Average Budget per MPAA Rating',
                 labels={'budget': 'Average Budget', 'certification': 'MPAA Rating'},
                 text_auto=True,
                 color='certification',
                 color_discrete_map=get_MPAA_color_map())

    fig.update_layout(xaxis_title="MPAA Rating",
                      yaxis_title="Average Budget",
                      yaxis_tickprefix='$', yaxis_tickformat=',.2s',
                      showlegend=False)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)
    
#--------------------------------------------------------------------------
def display_average_revenue_per_mpaa():
    # Filter to include movies with specific MPAA ratings and budget/revenue > 0
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]

    # Calculate average revenue
    avg_revenue = df_filtered.groupby('certification', as_index=False)['revenue'].mean()

    # Create bar plot for average revenue
    fig = px.bar(avg_revenue, x='certification', y='revenue',
                 title='Average Revenue per MPAA Rating',
                 labels={'revenue': 'Average Revenue', 'certification': 'MPAA Rating'},
                 text_auto=True,
                 color='certification',
                 color_discrete_map=get_MPAA_color_map())

    fig.update_layout(xaxis_title="MPAA Rating",
                      yaxis_title="Average Revenue",
                      yaxis_tickprefix='$', yaxis_tickformat=',.2s',
                      showlegend=False)
    fig.update_traces( textposition='outside')
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_average_revenue_by_movie_length():
    # Filter to include movies with specific MPAA ratings and budget/revenue > 0
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]

    # Categorizing movies based on runtime for 'Long Movies' and 'Short Movies'
    df_filtered['Movie_Length'] = pd.np.where(df_filtered['runtime'] > 150, 'Long Movies',
                                              pd.np.where(df_filtered['runtime'] <= 90, 'Short Movies', pd.NA))

    # Dropping movies not categorized as either 'Long Movies' or 'Short Movies'
    df_filtered = df_filtered.dropna(subset=['Movie_Length'])

    # Calculate average revenue for 'Long Movies' and 'Short Movies'
    avg_revenue_length = df_filtered.groupby('Movie_Length')['revenue'].mean().reset_index()
    
    # Create bar plot for average revenue by movie length
    fig = px.bar(avg_revenue_length, x='Movie_Length', y='revenue',
                 title='Average Revenue by Movie Length',
                 labels={'revenue': 'Average Revenue', 'Movie_Length': ''},
                 text_auto=True,
                 color='Movie_Length',
                 color_discrete_map=get_movie_length_color_map())

    fig.update_layout(xaxis_title="",
                      yaxis_title="Average Revenue",
                      yaxis_tickprefix='$', yaxis_tickformat=',.2s',
                      showlegend=False)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_average_revenue_by_length_and_rating():
    # Filter to include movies with specific MPAA ratings and budget/revenue > 0
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]

    # Categorizing movies based on runtime for 'Long Movies' and 'Short Movies'
    df_filtered['Movie_Length'] = 'Short Movies'  # Default
    df_filtered.loc[df_filtered['runtime'] > 150, 'Movie_Length'] = 'Long Movies'

    # Aggregate data to get the average revenue by MPAA rating and movie length
    plot_df = df_filtered.groupby(['certification', 'Movie_Length'], as_index=False)['revenue'].mean()

    # Create bar plot for average revenue by lenght and rating
    fig = px.bar(plot_df, x='certification', y='revenue', color='Movie_Length',
                 title='Average Revenue by Movie Length and MPAA Rating',
                 labels={'revenue': 'Average Revenue', 'certification': 'MPAA Rating', 'Movie_Length': 'Movie Length'},
                 text_auto=True,
                 color_discrete_map=get_movie_length_color_map(),
                 barmode='group',
                 category_orders={'certification': ['G', 'PG', 'PG-13', 'R']})  

    fig.update_layout(xaxis_title="MPAA Rating",
                      yaxis_title="Average Revenue",
                      yaxis_tickprefix='$', yaxis_tickformat=',.2s',
                      legend_title_text='Movie Length')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_average_revenue_by_genre():
    # Provide additional instruction for a better user experience
    st.info("""Hover for details or click fullscreen icon for larger view.""")
    
    # Calculate average revenue by genre
    avg_revenue = df_genres.groupby('genre_name')['revenue'].mean().reset_index()

    # Sorting genres by average revenue for plotting
    avg_revenue = avg_revenue.sort_values(by='revenue', ascending=True)

    # Create bar plot for average revenue by genre
    fig = px.bar(avg_revenue, y='genre_name', x='revenue', orientation='h',
                 title='Average Revenue by Genre',
                 labels={'revenue': 'Average Revenue', 'genre_name': 'Genre'},
                 text_auto=True)

    fig.update_layout(xaxis_title="Average Revenue",
                      yaxis_title="Genre",
                      xaxis_tickprefix='$', xaxis_tickformat=',.2s',
                      height=480)

    fig.update_traces(textposition='outside',
                      width=0.5)  
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_revenue_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='revenue', marginal='box',
                       title='Revenue Distribution',
                       labels={'revenue': 'Revenue'},
                       opacity=0.6)
    fig.update_xaxes(title="Revenue")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_popularity_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='popularity', marginal='box',
                       title='Popularity Distribution',
                       labels={'popularity': 'Popularity'},
                       opacity=0.6)
    fig.update_xaxes(title="Popularity", range=[0, 120])
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)


def display_popularity_MPAA_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='popularity', color='certification', marginal='box',
                       title='Popularity Distribution',
                       labels={'popularity': 'Popularity','certification': 'MPAA Rating'},
                       opacity=0.6,
                       color_discrete_map=get_MPAA_color_map())
    fig.update_xaxes(title="Popularity", range=[0, 120])
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
def display_runtime_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='runtime', marginal='box',
                       title='Runtime Distribution',
                       labels={'runtime': 'Runtime (minutes)'},
                       opacity=0.6, nbins=50)
    fig.update_xaxes(title="Runtime (minutes)")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

def display_runtime_MPAA_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='runtime', color='certification', marginal='box',
                       title='Runtime Distribution',
                       labels={'runtime': 'Runtime (minutes)', 'certification': 'MPAA Rating'},
                       opacity=0.6, nbins=50,
                       color_discrete_map=get_MPAA_color_map())
    fig.update_xaxes(title="Runtime (minutes)")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)
    
#--------------------------------------------------------------------------
def display_vote_average_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='vote_average', marginal='box',
                       title='Vote Average Distribution',
                       labels={'vote_average': 'Vote Average'},
                       opacity=0.6, nbins=50)
    fig.update_xaxes(title="Vote Average")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)


def display_vote_average_MPAA_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='vote_average', color='certification', marginal='box',
                       title='Vote Average Distribution',
                       labels={'vote_average': 'Vote Average', 'certification': 'MPAA Rating'},
                       opacity=0.6, nbins=50,
                       color_discrete_map=get_MPAA_color_map())
    fig.update_xaxes(title="Vote Average")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)
    
#--------------------------------------------------------------------------
def display_vote_count_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='vote_count', marginal='box',
                       title='Vote Count Distribution',
                       labels={'vote_count': 'Vote Count'},
                       opacity=0.6, nbins=50)
    fig.update_xaxes(title="Vote Count")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)


def display_vote_count_MPAA_distribution():
    df_filtered = df_eda[(df_eda['budget'] > 0) & (df_eda['revenue'] > 0) & 
                         (df_eda['certification'].isin(['G', 'PG', 'PG-13', 'R']))]
    fig = px.histogram(df_filtered, x='vote_count', color='certification', marginal='box',
                       title='Vote Count Distribution',
                       labels={'vote_count': 'Vote Count', 'certification': 'MPAA Rating'},
                       opacity=0.6, nbins=50,
                       color_discrete_map=get_MPAA_color_map())
    fig.update_xaxes(title="Vote Count")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

#--------------------------------------------------------------------------
# DROPDOWN MENU
#--------------------------------------------------------------------------
def main():
    # Display only one header for the financial visualizations
    st.header("Exploratory Data Analysis for Financial Visualizations")

    question = st.selectbox("Select a question to explore:", 
                            ["How many movies have valid financial data (either budget or revenue greater than 0)?",
                             "What is the distribution of movies across different MPAA ratings (G/PG/PG-13/R)?",
                             "What is the average budget for each MPAA Rating?",
                             "Budget Distribution",
                             "What is the average revenue for each MPAA Rating?",
                             "What is the average revenue of movies based on their length?",
                             "What is the average revenue of movies based on their length for each MPAA Rating?",
                             "What is the average revenue of movies based on their genre?",
                             "Revenue Distribution",
                             "Popularity Distribution", 
                             "Popularity Distribution by MPAA rating",
                             "Runtime Distribution",
                             "Runtime Distribution by MPAA rating",
                             "Vote Average Distribution", 
                             "Vote Average Distribution by MPAA rating", 
                             "Vote Count Distribution",
                             "Vote Count Distribution by MPAA rating"                             
                            ])

    if question == "How many movies have valid financial data (either budget or revenue greater than 0)?":
        display_financial_data()
    elif question == "What is the distribution of movies across different MPAA ratings (G/PG/PG-13/R)?":
        display_mpaa_distribution()
        
    elif question == "What is the average budget for each MPAA Rating?":
        display_average_budget_per_mpaa()
    elif question == "Budget Distribution":
        display_budget_distribution()
        
    elif question == "What is the average revenue for each MPAA Rating?":
        display_average_revenue_per_mpaa()
    elif question == "What is the average revenue of movies based on their length?":
        display_average_revenue_by_movie_length()
    elif question == "What is the average revenue of movies based on their length for each MPAA Rating?":
        with st.spinner("Please wait while the chart is being generated..."):
            display_average_revenue_by_length_and_rating()
    elif question == "What is the average revenue of movies based on their genre?":
        display_average_revenue_by_genre()
    elif question == "Revenue Distribution":
        display_revenue_distribution()
        
    elif question == "Popularity Distribution":
        display_popularity_distribution()
    elif question == "Popularity Distribution by MPAA rating":
        display_popularity_MPAA_distribution()
    
    elif question == "Runtime Distribution":
        display_runtime_distribution()
    elif question == "Runtime Distribution by MPAA rating":
        display_runtime_MPAA_distribution()
        
    elif question == "Vote Average Distribution":
        display_vote_average_distribution()
    elif question == "Vote Average Distribution by MPAA rating":
        display_vote_average_MPAA_distribution()
        
    elif question == "Vote Count Distribution":
        display_vote_count_distribution()
    elif question == "Vote Count Distribution by MPAA rating":
        display_vote_count_MPAA_distribution()

if __name__ == "__main__":
    main()
