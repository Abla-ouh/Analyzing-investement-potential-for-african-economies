import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from pmdarima import auto_arima
import praw
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.figure_factory as ff
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json


# Set page config
st.set_page_config(page_title="African Economic Dashboard", layout="wide")

# Title of the app
st.title("African Economic Indicators and Sectors Dashboard")

# Sidebar for selection
st.sidebar.header("Data Selection")

# Function to normalize the 'Year' column across datasets
def normalize_year_column(df):
    """Renames the year/année column to 'Year'."""
    for col in df.columns:
        if col.lower() in ['year', 'année']:
            df.rename(columns={col: 'Year'}, inplace=True)
            break
    return df

# Function to load data for indicators
def load_data_indicator(indicator):
    """Loads the dataset based on the selected indicator."""
    if indicator == 'GDP':
        df = pd.read_csv('africa_gdp_data_2000_2023.csv')
    elif indicator == 'Inflation':
        df = pd.read_csv('Inflation_Cleaned_data.csv')
    elif indicator == 'Exports':
        df = pd.read_csv('africa_Exports_data_with_regression.csv')
    elif indicator == 'Imports':
        df = pd.read_csv('africa_Imports_data_with_regression.csv')
    elif indicator == 'Labor Force':
        df = pd.read_excel('lab.xlsx')
    elif indicator == 'FDI':
        df = pd.read_excel('fdi.xlsx')
    elif indicator == 'Political Stability':
        df = pd.read_excel('stb_politic.xlsx')
    elif indicator == 'Urbanization':
        df = pd.read_excel('urb1.xlsx')
    elif indicator == 'Population Growth':
        df = pd.read_excel('grow_pop.xlsx')

    return normalize_year_column(df)

# Function to load data for sectors
def load_data_sector(sector):
    """Loads the dataset based on the selected sector."""
    if sector == 'Technology':
        df = pd.read_excel('techno.xlsx')
    elif sector == 'Energy':
        df = pd.read_excel('ene1.xlsx')
    elif sector == 'Agriculture':
        df = pd.read_excel('agri1.xlsx')

    return normalize_year_column(df)

@st.cache_data
def load_geojson():
    """Load and cache the GeoJSON data"""
    with open('world-countries.json') as f:
        return json.load(f)


# Function for forecasting
# def forecast_sector_data(df, country, periods=2):
#     """
#     Forecasts sector data for a specific country using ARIMA
    
#     Args:
#         df: DataFrame with 'Year' and country columns
#         country: String name of the country to forecast
#         periods: Number of years to forecast (default 2)
    
#     Returns:
#         Tuple of (forecast values, forecast years)
#     """
#     # Prepare data
#     country_data = df[['Year', country]].dropna()
#     country_data.set_index('Year', inplace=True)
    
#     # Fit ARIMA model
#     model = auto_arima(
#         country_data[country],
#         seasonal=False,
#         d=1,
#         suppress_warnings=True,
#         stepwise=True
#     )
    
#     # Generate forecast
#     forecast = model.predict(n_periods=periods)
#     future_years = [country_data.index.max() + i for i in range(1, periods + 1)]
    
#     return forecast, future_years

# Sidebar for selecting type (Indicators or Sectors)
data_type = st.sidebar.radio("Select Data Type:", options=['Indicators', 'Sectors'], index=0)

# Sidebar for selecting indicators or sectors based on the type
if data_type == 'Indicators':
    data_option = st.sidebar.selectbox(
        "Select an Indicator:",
        options=['GDP', 'Inflation', 'Exports', 'Imports', 'Labor Force', 'FDI', 
                'Political Stability', 'Urbanization', 'Population Growth'],
        index=0
    )
    data = load_data_indicator(data_option)
else:
    data_option = st.sidebar.selectbox(
        "Select a Sector:",
        options=['Technology', 'Energy', 'Agriculture'],
        index=0
    )
    data = load_data_sector(data_option)

def forecast_sector_data(df, country, periods=2):
    """
    Forecasts time series data (indicator or sector) for a specific country using ARIMA.
    
    Args:
        df: DataFrame with 'Year' and country columns.
        country: Name of the country column.
        periods: Number of years to forecast (default: 2).
    
    Returns:
        Tuple of forecasted values and forecasted years.
    """
    country_data = df[['Year', country]].dropna()
    country_data.set_index('Year', inplace=True)
    
    if len(country_data) < 3:
        return [], []  # Not enough data to forecast

    try:
        model = auto_arima(
            country_data[country],
            seasonal=False,
            d=1,
            suppress_warnings=True,
            stepwise=True
        )
        forecast = model.predict(n_periods=periods)
        future_years = [country_data.index.max() + i for i in range(1, periods + 1)]
        return forecast, future_years
    except Exception as e:
        print(f"ARIMA failed for {country}: {e}")
        return [], []



def perform_clustering_analysis(all_dfs, countries_list, target_year=2023):
    """
    Performs clustering analysis on the economic indicators
    """
    # Find common countries between all datasets
    common_countries = set(countries_list)
    for df in all_dfs:
        available_countries = set(df.columns) - {'Year'}
        common_countries &= available_countries
    common_countries = sorted(common_countries)
    
    # Prepare data for clustering
    data_combined = {}
    indicators = ['GDP', 'Inflation', 'Exports', 'Imports', 'Labor Force', 'FDI', 
                 'Political Stability', 'Urbanization', 'Technology', 'Energy', 
                 'Agriculture', 'Population Growth']
    
    # Load data for each indicator
    for name, df in zip(indicators, all_dfs):
        try:
            df_filtered = df[df['Year'] == target_year]
            if not df_filtered.empty:
                data_combined[name] = df_filtered[common_countries].iloc[0].values
        except Exception as e:
            st.warning(f"Error processing {name}: {str(e)}")
    
    # Convert to DataFrame
    data_combined_df = pd.DataFrame(data_combined)
    data_combined_df.index = common_countries
    
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_combined_df)
    
    # Apply PCA
    pca = PCA(n_components=6)
    data_reduced = pca.fit_transform(scaled_data)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(data_reduced)
    
    # Add cluster labels to the data
    data_combined_df['Cluster'] = labels
    
    return data_reduced, labels, data_combined_df, common_countries


# List of African countries
countries = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 
            'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 
            'Congo', 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 
            'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 
            'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 
            'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 
            'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe',
            'Senegal', 'Seychelles', 'Sierra Leone', 'South Africa', 'Sudan', 
            'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe']

# Sidebar to select countries
selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    options=countries,
    default=countries[:5]  # Set default selection to first 5 countries
)

# Filter the data based on selected countries
valid_countries = [country for country in selected_countries if country in data.columns]

if len(valid_countries) < len(selected_countries):
    missing_countries = [country for country in selected_countries if country not in data.columns]
    for missing in missing_countries:
        st.warning(f"Data for {missing} is not available for {data_option}")

filtered_data = data[['Year'] + valid_countries]

# Display the dataset summary in an expander
with st.expander(f"Data Summary for {data_option}"):
    st.write(filtered_data)

# Main visualizations
st.subheader("Visualizations")

if 'Year' in filtered_data.columns:
    # Create figure
    fig = go.Figure()
    
    # Plot actual data and forecasts for each country
    for country in valid_countries:
        # Get actual data
        actual_data = filtered_data[['Year', country]]
        
        #if data_type == 'Sectors':  # Only show forecasts for sectors
        if data_type in ['Sectors', 'Indicators']:  # Enable forecasts for both

            try:
                # Generate forecast
                forecast_values, forecast_years = forecast_sector_data(filtered_data, country)
                
                # Create continuous line by including last actual point
                last_actual_year = actual_data['Year'].max()
                last_actual_value = actual_data[actual_data['Year'] == last_actual_year][country].values[0]
                
                forecast_x = [last_actual_year] + list(forecast_years)
                forecast_y = [last_actual_value] + list(forecast_values)
                
                # Add actual data line
                fig.add_trace(
                    go.Scatter(
                        x=actual_data['Year'],
                        y=actual_data[country],
                        name=f"{country} (Actual)",
                        mode='lines',
                    )
                )
                
                # Add forecast line (dotted)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_x,
                        y=forecast_y,
                        name=f"{country} (Forecast)",
                        mode='lines',
                        line=dict(dash='dot'),
                        line_color=fig.data[-1].line.color,  # Match color with actual data
                    )
                )
                
            except Exception as e:
                # If forecast fails, just plot actual data
                fig.add_trace(
                    go.Scatter(
                        x=actual_data['Year'],
                        y=actual_data[country],
                        name=country,
                        mode='lines',
                    )
                )
        # else:
        #     # For indicators, just plot actual data
        #     fig.add_trace(
        #         go.Scatter(
        #             x=actual_data['Year'],
        #             y=actual_data[country],
        #             name=country,
        #             mode='lines',
        #         )
        #     )
    
    # Update layout
    fig.update_layout(
        title=f"{data_option} Trends and Forecasts",
        xaxis_title="Year",
        yaxis_title=data_option,
        hovermode='x unified',
        showlegend=True
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


def standardize_country_names(df):
    """Standardize country names to match GeoJSON format"""
    name_mapping = {
        'Egypt, Arab Rep.': 'Egypt',
        'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
        'Congo, Rep.': 'Republic of Congo',
        'Cabo Verde': 'Cape Verde',
        'Gambia, The': 'Gambia',
        'Sao Tome and Principe': 'São Tomé and Príncipe',
        'Tanzania': 'United Republic of Tanzania'
    }
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Replace country names
    if 'Country' in df_copy.columns:
        df_copy['Country'] = df_copy['Country'].replace(name_mapping)
    
    return df_copy

# Choropleth Map for Geographic Visualization# Choropleth Map for Geographic Visualization
if 'Year' in data.columns and any(country in data.columns for country in countries):
    # Use the latest year data for the map
    latest_year_data = data[data['Year'] == data['Year'].max()]
    
    # Melt the data for plotting
    latest_year_data_melted = latest_year_data.melt(
        id_vars=['Year'], 
        var_name='Country', 
        value_name='Value'
    )

    geojson_data = load_geojson()
    
    # Define color scale based on indicator
    if data_option == 'Inflation':
        color_scale = [[0, "green"], [0.3, "yellow"], [0.6, "orange"], [1, "red"]]
    else:
        color_scale = [[0, "red"], [0.3, "orange"], [0.6, "yellow"], [1, "green"]]

    latest_year_data_melted = standardize_country_names(latest_year_data_melted)
    
    # Create choropleth map
    map_fig = px.choropleth(
        latest_year_data_melted,
        geojson=geojson_data,
        locations='Country',
        color='Value',
        title=f"{data_option} Across African Countries in {latest_year_data_melted['Year'].values[0]}",
        color_continuous_scale=color_scale,
        featureidkey="properties.name"
    )
    
    
    
    # Display the map
    st.plotly_chart(map_fig, use_container_width=True)







if data_type == 'Indicators':
    st.markdown("---")
    st.subheader("Country Clustering Analysis")
    
    # Load all datasets
    all_dfs = [
        load_data_indicator('GDP'),
        load_data_indicator('Inflation'),
        load_data_indicator('Exports'),
        load_data_indicator('Imports'),
        load_data_indicator('Labor Force'),
        load_data_indicator('FDI'),
        load_data_indicator('Political Stability'),
        load_data_indicator('Urbanization'),
        load_data_sector('Technology'),
        load_data_sector('Energy'),
        load_data_sector('Agriculture')
    ]
    
    # Perform clustering
    with st.spinner('Performing clustering analysis...'):
        data_reduced, labels, clustered_data, common_countries = perform_clustering_analysis(
            all_dfs, countries
        )
        
        # Create 3D scatter plot using plotly
        scatter_fig = go.Figure(data=[
            go.Scatter3d(
                x=data_reduced[:, 0],
                y=data_reduced[:, 1],
                z=data_reduced[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=labels,
                    colorscale='viridis',
                    opacity=1
                ),
                text=common_countries,  # Add country names for hover
                hoverinfo='text',
                hovertemplate="Country: %{text}<br>" +
                            "Cluster: %{marker.color}<br>" +
                            "<extra></extra>"
            )
        ])
        
        # Update layout
        scatter_fig.update_layout(
            title="Country Clusters Based on Economic Indicators",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            width=800,
            height=800
        )
        
        # Display the plot
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Display cluster information
        st.subheader("Cluster Analysis Results")
        
        # Create a more readable cluster DataFrame
        cluster_df = clustered_data[['Cluster']].copy()
        cluster_df['Cluster'] = cluster_df['Cluster'].map({
            0: 'Cluster 1',
            1: 'Cluster 2',
            2: 'Cluster 3',
            3: 'Cluster 4'
        })
        
        # Display countries in each cluster
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Countries by Cluster:")
            for cluster in range(4):
                cluster_countries = cluster_df[cluster_df['Cluster'] == f'Cluster {cluster + 1}'].index.tolist()
                st.write(f"**Cluster {cluster + 1}:** {', '.join(cluster_countries)}")
        
        with col2:
            # Show cluster sizes
            cluster_sizes = cluster_df['Cluster'].value_counts()
            fig_pie = px.pie(
                values=cluster_sizes.values,
                names=cluster_sizes.index,
                title="Distribution of Countries Across Clusters",
                color=cluster_sizes.index,  # Set color by cluster
                color_discrete_sequence=['#FF4B4B', '#ff7f0e', '#ffbb78', '#fbbf3b', '#ffff00']
            )
            st.plotly_chart(fig_pie)
        
        # Download button for cluster results
        csv = cluster_df.to_csv(index=True)
        st.download_button(
            label="Download Clustering Results",
            data=csv,
            file_name="clustering_results.csv",
            mime="text/csv"
        )

# Footer with additional information
st.markdown("---")
st.markdown("""
    **Dashboard Notes:**
    - Forecasts are available only for sector data
    - Forecast values are shown with dotted lines and markers
    - Data may not be available for all countries
    - Use the sidebar to customize your view
""")

# Add cache decorator to main functions for better performance
load_data_indicator = st.cache_data(load_data_indicator)
load_data_sector = st.cache_data(load_data_sector)