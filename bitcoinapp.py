import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta


# Load Bitcoin data from a CSV file
#@st.cache  # This decorator helps cache the data load for speedup and less reloads
def load_data():
    data = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/final_raw_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure the 'Date' column is datetime type
    return data

# Title of the dashboard
st.title('Bitcoin Trading Platform')

# A simple interaction: button
if st.button('Welcome Exploring our Bitcoin Dasboard!'):
    st.write("Hello there! I'm Kiki, your cheerful AI Agent! 🌈 I'm here to guide you through a thrilling journey into the world of Bitcoin price simulation. We'll dive into colorful charts and crystal-clear instructions to make your experience both fun and informative. Get ready to unlock the mysteries of Bitcoin forecasting with a sprinkle of joy! 🚀💖")
    
# Load data
data = load_data()

# Display the first few rows of the dataframe
with st.expander('Preview of Dataset'):
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
sections = ["1. Cryptocurrency Trends Visualization", "2. Close Price Moving Average", 
            "3. OHLC and Volume Analysis", "4. Prediction Forecast", 
            "5. Transaction Management", "6. Trading Simulator"]

# Write each item on a new line in the sidebar
st.sidebar.title("Sections")
for section in sections:
    st.sidebar.write(section)


# Main panel
st.title("Cryptocurrency Trends Visualization")
with st.expander('Instruction'):
    st.write("Hey there! Take your pick from the keywords that you are interested in. We've got some adorable line charts waiting to show you the trends over the past couple of years. And guess what? You can choose whether to view it by Day(D), Week(W), or Month(M)! One important thing is that you can zoom in or out for particular time range! Let's explore it!!!")

col1, col2 = st.columns(2)

with col1:
    selected_features = st.multiselect(
        "Select Keyword Features", 
        options=features_to_include, 
        default=["Bitcoin Address", "Bitcoin Price", "Blockchain", "Crypto"]
    )

with col2:
    # Direct use of full terms for the time scale
    time_scales = {'Day': 'D', 'Week': 'W', 'Month': 'M'} # List of options
    selected_time_scale = st.selectbox(
        "Select Time Scale", 
        options=time_scales,  # Use full names directly
        index=0
    )
    selected_scale = time_scales[selected_time_scale]


    
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

if st.button('Refresh Data'):
    data = load_data()  # Re-load data if needed
    st.experimental_rerun()  # Rerun the app to update graphs

# Moving Average
st.title("Close Price Moving Average ")
# Streamlit user interface
with st.expander('Instruction'):
    st.write("Hey! Feel free to any input window size you would like to check.")
st.write("This app calculates the moving average of the Bitcoin closing prices.")

# User input for the moving average period
ma_period = st.slider('Select the moving average period (days):', min_value=1, max_value=200, value=10)

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
with st.expander('Instruction'):
    st.write("Hey! We have displayed Bitcoin OHLC and volumn charts, feel free to choose any time range you are interested in! Don't forget to zoom in to get a better look.")

# Dropdown for time scale
selected_scale = st.selectbox(
    'Select Time Scale',
    options=['Day', 'Week', 'Month'],
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






# Prediction

pred_data = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/model_pred_actual.csv')

pred_data2 = pred_data.copy()
pred_data2['Day'] = pd.to_datetime(pred_data2['Day'], origin='2022-12-01', unit='D')

# Set Day as the index
pred_data2.set_index('Day', inplace=True)

# Streamlit app code
st.title('Interactive Prediction Comparison')
with st.expander('Model Prediction'):
    st.write("Hi! We've developed a robust model for predicting Bitcoin prices, testing 10 different models and their evaluation metrics. The model presented below has shown the best performance. Feel free to use the slider to explore the prediction results. Happy exploring! 🚀✨")

# Sidebar - Number of days to predict
days_to_predict = st.slider('Select the number of days to predict:', min_value=1, max_value=16, value=6, key = "days")

# Filter data based on the number of days to predict
filtered_data = pred_data2.iloc[:days_to_predict]

# Creating a line plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Actual_Price'], mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Predicted_Price'], mode='lines+markers', name='Predicted Price'))

# Update plot layout
fig.update_layout(
    title='Actual vs Predicted Prices',
    xaxis_title='Date',
    yaxis_title='Price',
    legend_title='Legend'
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Prediction and holding
final_forecast = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/final_forecast.csv')
final_forecast['Date'] = pd.to_datetime(final_forecast['Date'])
final_forecast.set_index('Date', inplace = True)


import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# Assuming df, df_results, final_forecast, and the VAR model functions are already defined

# Session state to store transactions
if 'transactions' not in st.session_state:
    st.session_state.transactions = []

def add_transaction():
    # Adding a new transaction with initial default values
    st.session_state.transactions.append({'price': 0.0, 'amount': 0.0})

def show_profit_loss(target_date):
    try:
        target_date = pd.to_datetime(target_date)
        date_range = pd.date_range(start=target_date - pd.Timedelta(days=2), end=target_date + pd.Timedelta(days=2))

        profit_loss_values = []
        hover_texts = []

        for date in date_range:
            if date in final_forecast.index:
                price = final_forecast.loc[date, 'Price_forecast']
                date_label = f"Forecast for {date.strftime('%Y-%m-%d')}"
            elif date in df.index:
                price = df.loc[date, 'Price']
                date_label = f"Actual for {date.strftime('%Y-%m-%d')}"
            else:
                continue

            daily_profit_loss = sum((price - trans['price']) * trans['amount'] for trans in st.session_state.transactions)
            profit_loss_values.append(daily_profit_loss)
            hover_texts.append(f"{date_label}: ${daily_profit_loss:.2f}")

        # Plotting with Plotly
        fig = go.Figure(data=[go.Scatter(x=date_range, y=profit_loss_values, text=hover_texts, hoverinfo='text', mode='lines+markers')])
        fig.update_layout(title='Profit/Loss for Selected Date and Surrounding Days', xaxis_title='Date', yaxis_title='Profit/Loss')
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")




# Main page setup
st.title("Transaction Management")
with st.expander('Instruction'):
    st.write('Just type in your buying price and how long you plan to keep it, and I’ll calculate your profit or loss using our super smart model! 🌟')
# Sidebar for transactions management
#st.header("Transactions Management")
if st.button("Add Transaction"):
    add_transaction()
# Display current transactions in the sidebar for editing
for i, trans in enumerate(st.session_state.transactions):
    with st.expander(f"Transaction {i + 1}"):
        trans['price'] = st.number_input(f"Buying Price {i + 1}", value=trans['price'])
        trans['amount'] = st.number_input(f"Amount Held {i + 1}", value=trans['amount'])

target_date = st.date_input("Target Date for Profit/Loss Calculation")

if st.button("Show Profit/Loss"):
    show_profit_loss(target_date)

## Trade Simulation
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
            transactions.append(f"Day {i+1}: Sold {money:.2f} at {round(actual_price_today, 2)} per BTC | Balance: **${round(money, 2)}$**")
            bitcoins_quantity = 0
        if (predicted_price_tmr / actual_price_today - 1) > expected_return:
            bitcoins_quantity = money / actual_price_today
            buy_days.append(i)
            transactions.append(f"Day {i+1}: Bought {bitcoins_quantity:.2f} Bitcoins at ${round(actual_price_today, 2)} per BTC")
            money = 0
        balance_history.append(money + bitcoins_quantity * actual_price_today)

    if bitcoins_quantity > 0:
        money = bitcoins_quantity * pred_data['Actual_Price'].iloc[-1]
        bitcoins_quantity = 0
        transactions.append(f"Last Day: Sold all Bitcoins for ${money:.4f} at ${round(actual_price_today, 2)} per BTC | Balance: ${round(money, 2)}")
        balance_history.append(money)

    chart = make_subplots(rows=1, cols=1)
    chart.add_trace(go.Scatter(x=list(range(len(balance_history))), y=balance_history, mode='lines+markers', name='Balance($)', line=dict(color='blue', width=1)))
    chart.update_layout(title='Account Balance Over Time', xaxis_title='Day', yaxis_title='Balance ($)', showlegend=True)
    
    final_balance = balance_history[-1]
    st.plotly_chart(chart)
    

    
    st.subheader("Trading activity log:")
    profit_loss_percentage = ((final_balance / initial_money) - 1) * 100
    baseline_profit_loss_percentage = ((pred_data['Actual_Price'].iloc[-1] / pred_data['Actual_Price'].iloc[0]) - 1) * 100
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Initial Balance: ${initial_money:,.2f}**")
        st.markdown(f"**Final Balance: ${final_balance:.2f}**")

    with col2:
        st.markdown(f"**Profit (Loss): {profit_loss_percentage:.2f}%**")
        st.markdown(f"**Baseline Profit (Loss):{baseline_profit_loss_percentage:.2f}%**")





    # Use one expander for the entire transaction log
    with st.expander("View All Transactions"):
        st.write(f"Initial Balance: {initial_money}")
        # Display each transaction within the expander
        for i, transaction in enumerate(transactions, start=1):
            st.write(f"Transaction {i}: {transaction}")

# Streamlit App
st.title("Automated Trading Simulation")
with st.expander('Instruction'):
    st.write("Welcome to our simulator!🌟 TReady to see some magic? Just pop in your expected return threshold and how much you're thinking of investing. We'll crunch the numbers with our super smart model and show you the potential profits. Let’s start exploring! 🚀✨")
expected_return = st.number_input("Enter expected return per transaction threshold (as decimal):", value=0.05)
initial_money = st.number_input("Enter the investment amount:", value=1000.0)
if st.button("Run Simulation"):
    trade_sim(expected_return, initial_money)

# Create days to predict


