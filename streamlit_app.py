import streamlit as st
import pandas as pd
import math
from pathlib import Path
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.
def plot_line_chart():
    # Add input fields in the sidebar
    st.sidebar.header('Chart Parameters')
    symbol = st.sidebar.text_input('Enter Stock Symbol:', 'AAPL')
    
    # Dropdown for interval selection
    interval_options = {
        'Daily': '1d',
        'Weekly': '1wk',
        'Monthly': '1mo',
        'Hourly': '1h',
        '15 Minutes': '15m'
    }
    selected_interval = st.sidebar.selectbox(
        'Select Time Interval:',
        list(interval_options.keys())
    )
    interval = interval_options[selected_interval]

    # Add days input with a slider
    days = st.sidebar.slider(
        'Select Number of Days:',
        min_value=7,
        max_value=365,
        value=90,
        step=1,
        help='Choose the number of days of historical data to fetch'
    )

    # Define date range based on user input
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Fetch data using user inputs
    df = yf.download(symbol,
                     start=start_date.strftime('%Y-%m-%d'),
                     end=end_date.strftime('%Y-%m-%d'),
                     interval=interval)

    if df.empty:
        st.error(f"No data retrieved for {symbol}. Please check the symbol or try a different interval.")
        return

    # Display date range info
    st.info(f"Showing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    
    # Rest of your plotting code remains the same...
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['High'], label='High', color='green', linestyle='--')
    plt.plot(df.index, df['Low'], label='Low', color='red', linestyle='--')
    plt.plot(df.index, df['Open'], label='Open', color='blue', linestyle='-')
    plt.plot(df.index, df['Close'], label='Close', color='black', linestyle='-')
    plt.plot(df.index, ma100, label='MA100', color='orange')
    plt.plot(df.index, ma200, label='MA200', color='darkgreen')
    plt.title(f'{symbol} Stock Trends (High, Low, Open, Close)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)

# Run the function
plot_line_chart()

def plot_candlestick_chart():
    # Define date range for the last 3 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Fetch data from Yahoo Finance
    df = yf.download('AAPL',
                     start=start_date.strftime('%Y-%m-%d'),
                     end=end_date.strftime('%Y-%m-%d'),
                     interval='1d')

    # Check if data is empty
    if df.empty:
        st.error("No data retrieved. Please adjust the date range or check the stock symbol.")
        return

    # Ensure valid columns and filter out non-trading days
    df = df[df['Volume'] > 0]
    if df.empty:
        st.error("No trading data found in the selected range.")
        return

    # Convert index to datetime for Plotly compatibility
    df.index = pd.to_datetime(df.index)

    # Debugging information
    st.write("DataFrame for Debugging:")
    st.dataframe(df[['Open', 'High', 'Low', 'Close']])

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    # Adjust chart layout
    fig.update_layout(
        title="AAPL Candlestick Chart (Last 3 Months)",
        yaxis_title="Stock Price (USD)",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        height=600,
        width=800,
        template='plotly_white'
    )

    # Plot the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Run the function
plot_candlestick_chart()


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
