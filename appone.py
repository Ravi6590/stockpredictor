import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load Model
model = load_model(r'C:\Users\Ravi Gupta\Desktop\jn\Stock Predictions Model.keras')

# Streamlit UI
st.set_page_config(page_title="Stock Market Predictor", layout="wide")
st.title("📈 Stock Market Predictor using LSTM")

# Sidebar Input
st.sidebar.header("Stock Selection")
stock = st.sidebar.text_input("Enter Stock Symbol", 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Fetch Data
data = yf.download(stock, start, end)

# Display Data
st.subheader(f"Stock Data for {stock}")
st.dataframe(data.style.format("{:.2f}"))

# Data Preprocessing
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Plot Moving Averages
st.subheader("📊 Moving Averages")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data.Close, label="Stock Price", color='g')
ax.plot(ma_50_days, label="MA 50 Days", color='r', linestyle='dashed')
ax.plot(ma_100_days, label="MA 100 Days", color='b', linestyle='dashed')
ax.plot(ma_200_days, label="MA 200 Days", color='orange', linestyle='dashed')
ax.legend()
st.pyplot(fig)

# Prepare Data for Prediction
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predictions
predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Prediction vs Original
st.subheader("📈 Original Price vs Predicted Price")
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(y, label="Original Price", color='g')
ax2.plot(predict, label="Predicted Price", color='r', linestyle='dashed')
ax2.set_xlabel("Time")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.sidebar.success("Enter a stock symbol to predict prices!")
