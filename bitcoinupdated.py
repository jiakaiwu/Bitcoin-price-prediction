import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from matplotlib.animation import FuncAnimation
import datetime
import plotly.express as px

# Define colors
primary_color = "#555555"  # Dark charcoal
secondary_color = "#FFA500"  # Bright yellow

st.markdown(
    f"""
    <style>
    body {{
        background-color: #f0f2f6;  /* Light gray background */
        font-family: Arial, sans-serif;
        font-size: 16px;  /* Adjust font size for better readability */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Adjust the size and font of the title
st.markdown(
    f"""
    <div style="background-color: {primary_color}; padding: 10px 20px; border-radius: 8px; text-align: center;">
        <h1 style="margin-bottom: 0; color: {secondary_color}; font-size: 46px;">
            🪙 Bitcoin Trading Platform
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)



# Load Bitcoin data from a CSV file
#@st.cache  # This decorator helps cache the data load for speedup and less reloads
def load_data():
    data = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/final_raw_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure the 'Date' column is datetime type
    return data


# Load data
data = load_data()



st.sidebar.title("Navigation")

main_section = st.sidebar.radio("Go to", ['Home', 'Crypto Exploration','AI Trading'])
#if main_section == 'Explore Crypto':
    #options = [
    #"1. Cryptocurrency Trends Visualization",
    #"2. Close Price Moving Average",
    #"3. OHLC and Volume Analysis"
    #]

    #st.sidebar.write("Explore Crypto:")
    #for option in options:
        #st.sidebar.write(option)

#elif main_section == "Trading Practice":
    #options1 = [
        #"1. Prediction Forecast", 
        #"2. Transaction Management", 
        #"3. Trading Simulator"
    #]
    #st.sidebar.write(options1)
    #for option1 in options1:
        #st.sidebar.write(option1)

# Render main sections
if main_section == 'Home':
    st.title("Welcome to our ₿ App")
    st.write("""
        Hello there! I'm Kiki, your cheerful AI Agent! 🌈 I'm here to guide you through a thrilling journey into the world of Bitcoin price simulation. We'll dive into colorful charts and crystal-clear instructions to make your experience both fun and informative. Get ready to unlock the mysteries of Bitcoin forecasting with a sprinkle of joy! 🚀💖
        """)
    
    # Define the colors
    primary_color = "#555555"  # Dark charcoal
    secondary_color = "#FFA500"  # Bright yellow

    # Define the content for each box with a smaller font size and height
    box_content_format = """
    <div style="background-color: {primary_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px; height: 60px; display: flex; justify-content: center; align-items: center; text-align: center;">
        <h3 style="color: {secondary_color}; margin: 0; font-size: 22px;">{option}</h3>
    </div>
    """

    # Divide the options into two columns
    crypto_options = [
        "Search Trends",
        "Close Price",
        "OHLC and Volume"
    ]

    trading_options = [
        "Forecast",
        "Transaction",
        "Simulator"
    ]

    # Display the boxes
    col1, col2 = st.columns([1, 1])  # Equal width for both columns

    # Display crypto options
    with col1:
        #st.markdown("## Explore Crypto")
        for option in crypto_options:
            st.markdown(box_content_format.format(
                primary_color=primary_color,
                secondary_color=secondary_color,
                option=option
            ), unsafe_allow_html=True)

    # Display trading options
    with col2:
        #st.markdown("## Trading Practice")
        for option in trading_options:
            st.markdown(box_content_format.format(
                primary_color=primary_color,
                secondary_color=secondary_color,
                option=option
            ), unsafe_allow_html=True)



    # Dataset Preview Expander
    st.title("Dataset")
    st.write("We use web scraping to scrape the bitcoin's relevant data directly from google trends website and YFinance API. Then we constructed a comprehensive Bitcoin Price dataset including 26 relevant keywords with Bitcoin and Bitcoin price data (Volume, Open, High, Low, and Close price)📈📊.")
    if st.checkbox('Show Dataset Preview'):
        st.write("Here are the first few rows of the Bitcoin price data:")
        st.write(data.head())


elif main_section == 'Crypto Exploration':
    #if visualization_section == '1. Cryptocurrency Trends Visualization':
    # Your detailed section content
    st.title("Cryptocurrency Trends")
    with st.expander('Instruction'):
        st.write("Hey there! Take your pick from the keywords that you are interested in. We've got some adorable line charts waiting to show you the trends over the past couple of years. And guess what? You can choose whether to view it by Day(D), Week(W), or Month(M)! One important thing is that you can zoom in or out for particular time range! Let's explore it!!!")

    features_to_include = sorted([
    "Altcoin", "BIT", "Bitcoin", "Bitcoin Address", "Bitcoin Investment", 
    "Bitcoin Mining", "Bitcoin News", "Bitcoin Price", "Bitcoin Wallet", 
    "Blockchain", "Block", "BTC", "Crypto", "Cryptography", 
    "Cryptocurrency", "DAPP", "DeFi", "Digital Currency", "Ethereum", 
    "Fiat money", "Fork", "Hash Rate", "ICO", "P2P", "Private Key", 
    "Stablecoin"
])
    col1, col2 = st.columns(2)
    with col1:
        selected_features = st.multiselect(
            "Select Keyword Features", 
            options= features_to_include,
            default=['Bitcoin Address', 'Bitcoin Price', 'Blockchain','Crypto']
        )

    with col2:
        time_scales = {'Day': 'D', 'Week': 'W', 'Month': 'M'}
        selected_time_scale = st.selectbox(
            "Select Time Scale", 
            options=list(time_scales.keys()),
            index=0
        )
        selected_scale = time_scales[selected_time_scale]

    # Function to update the graph
    def update_graph(selected_features, selected_scale):
        dff = data.copy()  # Make sure 'data' is defined and loaded appropriately
        dff.set_index('Date', inplace=True)
        dff = dff.resample(selected_scale).mean()  # Resample based on selection
        
        traces = []
        for feature in selected_features:
            traces.append(go.Scatter(x=dff.index, y=dff[feature], mode='lines', name=feature))
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Trend of Selected Features Over Time",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Relative Frequency",
            legend_title="Features",
            legend=dict(
                title="Search Words",
                title_font=dict(size=21),  # Adjust the title font size
                font=dict(size=13)  # Adjust the font size of legend labels
            )
        )
        
        return fig

    fig = update_graph(selected_features, selected_scale)
    st.plotly_chart(fig)
    
    if st.button('Refresh Data'):
        data = load_data()  # Re-load data if needed
        st.experimental_rerun()  # Rerun the app to update graphs

    #elif visualization_section == "2. Close Price Moving Average":
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
        legend=dict(
                #title="Moving Average",
                title_font=dict(size=26),  # Adjust the title font size
                font=dict(size=16)  # Adjust the font size of legend labels
            )
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    

    #elif visualization_section =="3. OHLC and Volume Analysis":
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

else:
    #if trading_section == "4. Prediction Forecast":
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
        legend=dict(
                #title="Actual vs Predicted Prices",
                title_font=dict(size=26),  # Adjust the title font size
                font=dict(size=18)  # Adjust the font size of legend labels
            )
        
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

        
    
    #elif trading_section == "5. Transaction Management":
        # Prediction and holding
    final_forecast = pd.read_csv('/Users/jiakaiwu/Documents/ieor courses/243/project/final_forecast.csv')
    final_forecast['Date'] = pd.to_datetime(final_forecast['Date'])
    final_forecast.set_index('Date', inplace = True)
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

    default_date = datetime.date(2024, 2, 6)
    target_date = st.date_input("Target Date for Profit/Loss Calculation", default_date)

    if st.button("Show Profit/Loss"):
        show_profit_loss(target_date)
    #else:
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
        

        
       
        profit_loss_percentage = ((final_balance / initial_money) - 1) * 100
        baseline_profit_loss_percentage = ((pred_data['Actual_Price'].iloc[-1] / pred_data['Actual_Price'].iloc[0]) - 1) * 100
        col1, col2 = st.columns(2)
        # Data for initial balance and final balance
        balance_data = {
            'Category': ['Initial      ', 'Final      '],
            'Amount': [initial_money, final_balance]
        }

        # Data for profit/loss and baseline profit/loss
        profit_data = {
            'Category': ['Baseline','Model'],
            'Percentage': [baseline_profit_loss_percentage, profit_loss_percentage]
        }

        # Plotting the histograms
        fig1 = px.bar(balance_data, x='Category', y='Amount', color='Category',
                    labels={'Amount': 'Amount ($)'},
                    title='Initial and Final Balance',
                    color_discrete_map={'Initial      ': 'blue', 'Final      ': 'green'})

        fig2 = px.bar(profit_data, x='Category', y='Percentage', color='Category',
                    labels={'Percentage': 'Percentage (%)'},
                    title='Profit (Loss) for Baseline and Our Model',
                    color_discrete_map={'Baseline': 'orange', 'Model': 'red'})
        
        fig1.update_layout(width=345)  # Adjust width as needed
        fig2.update_layout(width=345)
        # Display the plots side by side
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1)
        col2.plotly_chart(fig2)

        st.subheader("Transactions")
        with col1:
            st.markdown(f"**Initial Balance: ${initial_money:,.2f}**")
            st.markdown(f"**Final Balance: ${final_balance:.2f}**")

        with col2:
            st.markdown(f"**Baseline Profit (Loss):{baseline_profit_loss_percentage:.2f}%**")
            st.markdown(f"**Our Model Profit (Loss): {profit_loss_percentage:.2f}%**")

        # Use one expander for the entire transaction log
        if st.button("View All Transactions"):
            st.write(f"Initial Balance: {initial_money}")
            # Display each transaction within the expander
            for i, transaction in enumerate(transactions, start=1):
                st.write(f"Transaction {i}: {transaction}")

    # Streamlit App
    st.title("Automated Trading Simulation")
    with st.expander('Instruction'):
        st.write("Welcome to our simulator!🌟 Ready to see some magic? Just pop in your expected return threshold and how much you're thinking of investing. We'll crunch the numbers with our super smart model and show you the potential profits. Let’s start exploring! 🚀✨")
    st.write("We do not offer investment advice on our platform. We recommend consulting with a financial advisor for personalized guidance.")
    expected_return = st.number_input("Enter expected return per transaction threshold (as decimal):", value=0.05)
    initial_money = st.number_input("Enter the investment amount:", value=1000.0)
    with st.expander("Run Simulation"):
        trade_sim(expected_return, initial_money)

    st.title('Conclusion and Further Implication')
    data = {
    "Key Advantage": "Utilization of data-driven solutions for optimal returns, outperform than Coinbase",
    "Scalability": "Support for other cryptocurrency coins and alternative derivatives such as stocks",
    "Partnerships": "Collaboration with brokerages to expand accessibility of our investment tool",
    "Ultimate Goal": "Seamless API integration to financial accounts for automated trading processes"
    }

    css = """
    <style>
    table {
    font-family: Arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    }

    td, th {
    border: 1px solid #FFA500;
    text-align: left;
    padding: 10px;
    }

    tr:nth-child(even) {
    background-color: #f2f2f2;
    }

    th {
    background-color: #555555;
    color: white;
    }

    td:first-child, th:first-child {
    min-width: 150px; /* Set minimum width for first column */
    color: #FFA500;
    text-align: center;
    }

    td:nth-child(2), th:nth-child(2) {
    min-width: 400px; /* Set minimum width for second column */
    }
    </style>
    """

    # Display the table using HTML and CSS
    st.write(css, unsafe_allow_html=True)
    st.write("<table>", unsafe_allow_html=True)
    for key, value in data.items():
        st.write(f"<tr><th>{key}</th><td>{value}</td></tr>", unsafe_allow_html=True)
    st.write("</table>", unsafe_allow_html=True)



    firework_animation_html = """
    <style>
    @keyframes firework {
    0% { transform: translate(var(--x), var(--initialY)); width: var(--initialSize); opacity: 1; }
    50% { width: 0.5vmin; opacity: 1; }
    100% { width: var(--finalSize); opacity: 0; }
    }

    /* @keyframes fireworkPseudo {
    0% { transform: translate(-50%, -50%); width: var(--initialSize); opacity: 1; }
    50% { width: 0.5vmin; opacity: 1; }
    100% { width: var(--finalSize); opacity: 0; }
    }
    */
    .firework,
    .firework::before,
    .firework::after
    {
    --initialSize: 0.5vmin;
    --finalSize: 45vmin;
    --particleSize: 0.4vmin;
    --color1: yellow;
    --color2: khaki;
    --color3: white;
    --color4: lime;
    --color5: gold;
    --color6: mediumseagreen;
    --y: -30vmin;
    --x: -50%;
    --initialY: 60vmin;
    content: "";
    animation: firework 2s infinite;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, var(--y));
    width: var(--initialSize);
    aspect-ratio: 1;
    background: 
        /*
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 0% 0%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 100% 0%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 100% 100%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 0% 100%,
        */
        
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 50% 0%,
        radial-gradient(circle, var(--color2) var(--particleSize), #0000 0) 100% 50%,
        radial-gradient(circle, var(--color3) var(--particleSize), #0000 0) 50% 100%,
        radial-gradient(circle, var(--color4) var(--particleSize), #0000 0) 0% 50%,
        
        /* bottom right */
        radial-gradient(circle, var(--color5) var(--particleSize), #0000 0) 80% 90%,
        radial-gradient(circle, var(--color6) var(--particleSize), #0000 0) 95% 90%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 90% 70%,
        radial-gradient(circle, var(--color2) var(--particleSize), #0000 0) 100% 60%,
        radial-gradient(circle, var(--color3) var(--particleSize), #0000 0) 55% 80%,
        radial-gradient(circle, var(--color4) var(--particleSize), #0000 0) 70% 77%,
        
        /* bottom left */
        radial-gradient(circle, var(--color5) var(--particleSize), #0000 0) 22% 90%,
        radial-gradient(circle, var(--color6) var(--particleSize), #0000 0) 45% 90%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 33% 70%,
        radial-gradient(circle, var(--color2) var(--particleSize), #0000 0) 10% 60%,
        radial-gradient(circle, var(--color3) var(--particleSize), #0000 0) 31% 80%,
        radial-gradient(circle, var(--color4) var(--particleSize), #0000 0) 28% 77%,
        radial-gradient(circle, var(--color5) var(--particleSize), #0000 0) 13% 72%,
        
        /* top left */
        radial-gradient(circle, var(--color6) var(--particleSize), #0000 0) 80% 10%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 95% 14%,
        radial-gradient(circle, var(--color2) var(--particleSize), #0000 0) 90% 23%,
        radial-gradient(circle, var(--color3) var(--particleSize), #0000 0) 100% 43%,
        radial-gradient(circle, var(--color4) var(--particleSize), #0000 0) 85% 27%,
        radial-gradient(circle, var(--color5) var(--particleSize), #0000 0) 77% 37%,
        radial-gradient(circle, var(--color6) var(--particleSize), #0000 0) 60% 7%,
        
        /* top right */
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 22% 14%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 45% 20%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 33% 34%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 10% 29%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 31% 37%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 28% 7%,
        radial-gradient(circle, var(--color1) var(--particleSize), #0000 0) 13% 42%
        ;
    background-size: var(--initialSize) var(--initialSize);
    background-repeat: no-repeat;
    }

    .firework::before {
    --x: -50%;
    --y: -50%;
    --initialY: -50%;
    /*   transform: translate(-20vmin, -2vmin) rotate(40deg) scale(1.3) rotateY(40deg); */
    transform: translate(-50%, -50%) rotate(40deg) scale(1.3) rotateY(40deg);
    /*   animation: fireworkPseudo 2s infinite; */
    }

    .firework::after {
    --x: -50%;
    --y: -50%;
    --initialY: -50%;
    /*   transform: translate(44vmin, -50%) rotate(170deg) scale(1.15) rotateY(-30deg); */
    transform: translate(-50%, -50%) rotate(170deg) scale(1.15) rotateY(-30deg);
    /*   animation: fireworkPseudo 2s infinite; */
    }

    .firework:nth-child(2) {
    --x: 30vmin;
    }

    .firework:nth-child(2),
    .firework:nth-child(2)::before,
    .firework:nth-child(2)::after {
    --color1: pink;
    --color2: violet;
    --color3: fuchsia;
    --color4: orchid;
    --color5: plum;
    --color6: lavender;  
    --finalSize: 40vmin;
    left: 30%;
    top: 60%;
    animation-delay: -0.25s;
    }

    .firework:nth-child(3) {
    --x: -30vmin;
    --y: -50vmin;
    }

    .firework:nth-child(3),
    .firework:nth-child(3)::before,
    .firework:nth-child(3)::after {
    --color1: cyan;
    --color2: lightcyan;
    --color3: lightblue;
    --color4: PaleTurquoise;
    --color5: SkyBlue;
    --color6: lavender;
    --finalSize: 35vmin;
    left: 70%;
    top: 60%;
    animation-delay: -0.4s;
    }

    </style>


    <div class="firework"></div>
    <div class="firework"></div>
    <div class="firework"></div>
    """

    # Display the rocket animation in Streamlit
    if st.button('Thank you'):
        st.markdown(firework_animation_html, unsafe_allow_html=True)
        st.write("Thank you very much for your support and for using our product! We greatly appreciate your feedback and suggestions. Your input helps us improve and provide a better experience for all our users. Please feel free to share any ideas or suggestions you may have at any time. We're here to listen and continuously improve our product based on your needs. 🚀")