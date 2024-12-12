import streamlit as st
import pandas as pd
import math
from pathlib import Path
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.


def plot_candlestick_chartX():
    # Get today's date and calculate the date 3 months ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Use 90 days for better data coverage

    # Fetch data with sufficient history
    df = yf.download('CTSH', 
                     start=start_date.strftime('%Y-%m-%d'),
                     end=end_date.strftime('%Y-%m-%d'),
                     interval='1d')  # Ensure daily data
                     
    if df.empty:
        st.error("No data retrieved. Please check the stock symbol or the date range.")
        return
    
    # Filter out rows with zero volume (market closed)
    df = df[df['Volume'] > 0]
    
    # Debugging Information
    st.write("Debug: DataFrame Contents")
    st.dataframe(df)  # Interactive table view

    # Create figure
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',  # Customize colors
        decreasing_line_color='red'
    )])

    # Update layout for better visibility
    fig.update_layout(
        title='AAPL Stock Price',
        yaxis_title='Stock Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,  # Disable range slider
        height=600,  # Adjust height
        width=800,   # Adjust width
        template='plotly_dark'  # Use dark theme for better visibility
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Run the function
plot_candlestick_chartX()

def plot_candlestick_chart():
    # Get today's date and calculate date 6 months ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    
    # Fetch data with sufficient history
    df = yf.download('AAPL', 
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'))
    
    st.text("Inside plot_candlestick_chart")
    # Display DataFrame for debugging
    st.write("Debug: DataFrame Contents")
    st.dataframe(df)  # Interactive table view
    
    # Create figure
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',  # customize colors
        decreasing_line_color='red'
    )])
    
    # Update layout for better visibility
    fig.update_layout(
        title='AAPL Stock Price',
        yaxis_title='Stock Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,  # disable rangeslider
        height=600,  # increase height
        width=800    # increase width
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Run the function
plot_candlestick_chart()

def plot_candlestick_plotly(df):
    st.text("Inside plot_candlestick_plotly")
    # Display DataFrame for debugging
    st.write("Debug: DataFrame Contents")
    st.dataframe(df)  # Interactive table view


    # Create candlestick chart with volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=('Candlestick', 'Volume'),
                       row_width=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC'),
                                row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, 
                        y=df['Volume'],
                        name='Volume'),
                        row=2, col=1)

    fig.update_layout(
        title='Stock Price Analysis',
        yaxis_title='Stock Price',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
# Example usage
symbol = "TSLA"
data = yf.download(symbol, start="2024-12-01")
plot_candlestick_plotly(data)


@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
