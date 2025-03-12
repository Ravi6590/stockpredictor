import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained LSTM model
model = load_model(r'C:\Users\Ravi Gupta\Desktop\jn\Stock Predictions Model.keras')

# Set up the Streamlit app
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

# Header
st.title("📈 Stock Market Predictor")
st.markdown("Predict stock prices using LSTM model and visualize trends.")

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # Theme Selector
    theme = st.radio("Select Theme", ["Light", "Dark"])
    
    stock = st.text_input("Enter Stock Symbol", "GOOG")
    start_date = st.date_input("Start Date", pd.to_datetime("2012-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2022-12-31"))
    
    # Ensure end_date is not in the future
    today = datetime.date.today()
    if end_date > today:
        st.warning(f"End date adjusted to today's date: {today}")
        end_date = today
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

# Download stock data
try:
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.error("Stock data could not be downloaded. Please check the stock symbol and try again.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display raw data
st.header("📊 Stock Data")
with st.expander("View Raw Data"):
    st.write(data)

# Split data into training and testing sets
data_train = data.Close[:int(len(data) * 0.80)]
data_test = data.Close[int(len(data) * 0.80):]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

if data_test.empty:
    st.error("Not enough stock data for prediction. Try selecting a different date range.")
    st.stop()

data_test_scaled = scaler.fit_transform(np.array(data_test).reshape(-1, 1))

# Moving Average Visualizations
st.header("📈 Stock Price Trends")
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.Close, label="Closing Price", color='g')
ax.plot(ma_50, label="50-Day MA", color='r')
ax.plot(ma_100, label="100-Day MA", color='b')
ax.plot(ma_200, label="200-Day MA", color='orange')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Prepare data for prediction
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions
predictions = model.predict(x)
predictions = predictions * (1 / scaler.scale_)
y = y * (1 / scaler.scale_)

# Plot Predictions
st.header("🔮 Predictions")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y, label="Original Price", color='g')
ax.plot(predictions, label="Predicted Price", color='r')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Footer
st.markdown("---")
