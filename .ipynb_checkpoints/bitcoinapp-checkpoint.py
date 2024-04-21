import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load Bitcoin data from a CSV file
#@st.cache  # This decorator helps cache the data load for speedup and less reloads
def load_data():
    data = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/final_raw_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure the 'Date' column is datetime type
    return data

# Title of the dashboard
st.title('Bitcoin Interactive Dashboard')

# A simple interaction: button
if st.button('Welcome Exploring our Bitcoin Dasboard!'):
    st.write("Greetings user! I am your AI Agent Kiki, We're excited to have you on board to explore its features and capabilities.")
    
# Load data
data = load_data()

# Display the first few rows of the dataframe
if st.button('Dataset'):
    st.write("Here are the first few rows of the Bitcoin price data:")
    st.write(data.head())



# List of features to include, sorted alphabetically by their first letter
features_to_include = sorted([
    "Altcoin", "BIT", "Bitcoin", "Bitcoin Address", "Bitcoin Investment", 
    "Bitcoin Mining", "Bitcoin News", "Bitcoin Price", "Bitcoin Wallet", 
    "Blockchain", "Block", "BTC", "Crypto", "Cryptography", 
    "Cryptocurrency", "DAPP", "DeFi", "Digital Currency", "Ethereum", 
    "Fiat money", "Fork", "Hash Rate", "ICO", "P2P", "Private Key", 
    "Stablecoin"
])

# Setting up the sidebar
st.sidebar.title("Settings")
selected_features = st.sidebar.multiselect("Select Features", features_to_include, default=["Bitcoin Price"])
selected_scale = st.sidebar.selectbox("Select Time Scale", ['D', 'W', 'M'], index=0, key = "Keyword Visualization")


# Main panel
st.title("Cryptocurrency Trends Visualization")
if st.button('Instruction'):
    st.write("Hey there! Take your pick from the keywords that tickle your fancy. We've got some adorable line charts waiting to show you the trends over the past couple of years. And guess what? You can choose whether to view it by day, week, or month—whatever floats your boat! One important thing is that you can zoom in or out for particular time range! Let's explore it!!!")
    

    
# Function to update the graph
def update_graph(selected_features, selected_scale):
    dff = data.copy()  # Make a copy of the dataframe for manipulation
    dff.set_index('Date', inplace=True)
    dff = dff.resample(selected_scale).mean()  # Resample based on selection
    
    # Create traces for the selected features
    traces = []
    for feature in selected_features:
        traces.append(go.Scatter(x=dff.index, y=dff[feature], mode='lines', name=feature))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Trend of Selected Features Over Time",
        title_x=0.5,  # Centers the title
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(
            x=1,  
            y=1, 
            title="Features",
            xanchor='right',  
            yanchor='top' 
        )
    )
    return fig

# Display the graph
fig = update_graph(selected_features, selected_scale)
#if st.button('Show Plot'):
st.plotly_chart(fig)



## Bitcoin OHLC and Volume Analysis
idx = pd.date_range(start='2023-01-01', periods=100, freq='D')
price_data = {
    'Open': pd.np.random.randint(10000, 50000, size=(100)),
    'High': pd.np.random.randint(50000, 70000, size=(100)),
    'Low': pd.np.random.randint(10000, 40000, size=(100)),
    'Close': pd.np.random.randint(20000, 60000, size=(100)),
    'Volume': pd.np.random.randint(100, 1000, size=(100))
}
df = pd.DataFrame(price_data, index=idx)

# Title
st.title("Bitcoin OHLC and Volume Analysis")
if st.button('Instruction', key = "OHLC"):
    st.write("Hey there! We have displayed Bitcoin OHLC and volumn charts, feel free to choose any time range you are interested in! Don't forget to zoom in to get a better look.")

# Dropdown for time scale
selected_scale = st.selectbox(
    'Select Time Scale',
    options=['D', 'W', 'M'],
    index=0,  # Default to 'Day'
    key = 'time_scale_selectbox'
)

# Function to resample data
def resample_data(scale):
    if scale == 'W':
        return df.resample('W-Mon').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    elif scale == 'M':
        return df.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    else:  # Daily or default
        return df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

# Resample based on selected scale
df_resampled = resample_data(selected_scale)

# OHLC Graph
ohlc_fig = go.Figure(data=[go.Candlestick(
    x=df_resampled.index,
    open=df_resampled['Open'],
    high=df_resampled['High'],
    low=df_resampled['Low'],
    close=df_resampled['Close']
)])
ohlc_fig.update_layout(title="Bitcoin OHLC Prices", xaxis_title="Date", yaxis_title="Price")


# Volume Graph
volume_fig = go.Figure(data=[go.Bar(
    x=df_resampled.index,
    y=df_resampled['Volume']
)])
volume_fig.update_layout(title="Bitcoin Trading Volume", xaxis_title="Date", yaxis_title="Volume")

# Display the graphs in Streamlit
st.plotly_chart(ohlc_fig)
st.plotly_chart(volume_fig)



# Moving Average
# Function to plot data with optional moving average
#pred_data['Date'] = pd.to_datetime(pred_data['Date'])
# Streamlit user interface
st.title('Bitcoin Moving Average Calculator')
st.write("This app calculates the moving average of the Bitcoin closing prices.")

# User input for the moving average period
ma_period = st.slider('Select the moving average period (days):', min_value=1, max_value=100, value=10)

# Calculate the moving average
data['Moving Average'] = data['Close'].rolling(window=ma_period).mean()
#data['Moving Average'] = data['Open'].rolling(window=ma_period).mean()
#data['Moving Average'] = data['High'].rolling(window=ma_period).mean()
#data['Moving Average'] = data['Low'].rolling(window=ma_period).mean()

# Plotting the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Moving Average'], mode='lines', name='Moving Average'))

# Update plot layout
fig.update_layout(
    title='Bitcoin Closing Prices and Moving Average',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Legend'
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)


# Prediction
pred_data = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/model_pred_actual.csv')
def trade_sim(expected_return, initial_money):
    money = initial_money
    bitcoins_quantity = 0
    balance_history = [initial_money]
    buy_days = []
    transactions = []  # For storing transactions

    for i in range(len(pred_data) - 1):
        actual_price_today = pred_data['Actual_Price'][i]
        predicted_price_tmr = pred_data['Predicted_Price'][i + 1]
        if bitcoins_quantity > 0:
            money = bitcoins_quantity * actual_price_today
            transactions.append(f"Day {i+1}: Sold Bitcoins for ${money:.2f} at ${round(actual_price_today, 2)} per BTC | Balance: ${round(money, 2)}")
            bitcoins_quantity = 0
        if (predicted_price_tmr / actual_price_today - 1) > expected_return:
            bitcoins_quantity = money / actual_price_today
            buy_days.append(i)
            transactions.append(f"Day {i+1}: Bought {bitcoins_quantity:.4f} Bitcoins at ${round(actual_price_today, 2)} per BTC")
            money = 0
        balance_history.append(money + bitcoins_quantity * actual_price_today)

    if bitcoins_quantity > 0:
        money = bitcoins_quantity * pred_data['Actual_Price'].iloc[-1]
        bitcoins_quantity = 0
        transactions.append(f"Last Day: Sold all Bitcoins for ${money:.2f} at ${round(actual_price_today, 2)} per BTC | Balance: ${round(money, 2)}")
        balance_history.append(money)

    chart = make_subplots(rows=1, cols=1)
    chart.add_trace(go.Scatter(x=list(range(len(balance_history))), y=balance_history, mode='lines+markers', name='Balance($)', line=dict(color='blue', width=1)))
    chart.update_layout(title='Account Balance Over Time', xaxis_title='Day', yaxis_title='Balance ($)', showlegend=True)
    
    final_balance = balance_history[-1]
    st.plotly_chart(chart)
    st.write(f"**Final Balance:** ${final_balance:.2f} | **Profit(Loss):** {((final_balance / initial_money) - 1) * 100:.2f}%")
    st.write(f"**Baseline Profit(Loss):** {((pred_data['Actual_Price'].iloc[-1] / pred_data['Actual_Price'].iloc[0]) - 1) * 100:.2f}%")

    st.subheader("Trading activity log:")
    st.write("Initial Balance:", initial_money)
    for transaction in transactions:
        st.write(transaction)

# Streamlit App
st.title("Bitcoin Trading Simulation")
expected_return = st.number_input("Enter expected return per transaction threshold (as decimal):", value=0.01)
initial_money = st.number_input("Enter the investment amount:", value=1000.0)
if st.button("Run Simulation"):
    trade_sim(expected_return, initial_money)