from _typeshed import FileDescriptor
import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import style
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential,load_model

import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
style.use('ggplot')

import streamlit as st

start = '2000-01-01'
end = '2020-12-31'

st.title("Stock Price Prediction")

user_input = st.text_input("Enter Stock Ticker","^NSEI")

# if user_input == '^NSEI':
#     df=pd.read_csv('~/projects/Class Project/training model/nse_historical_data.csv',index_col='Date', parse_dates=True)

# else:
#     df=pd.read_csv('~/projects/Class Project/training model/nse_dfs/{}.csv'.format(user_input), index_col='Date', parse_dates=True)

df = data.DataReader(user_input, 'yahoo', start, end)


data_training = pd.DataFrame(df['Adj Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Adj Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model('stock_price_prediction_model_4.h5')

past_30_days = data_training.tail(30)

final_df = past_30_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-30: i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_predicted = model.predict(X_test)

print(len(y_predicted))
print(len(y_test))

scaler = scaler.scale_
scale_factor = 1/scaler[0]

y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

n = 0
y_predicted = y_predicted[:, n, n]

values = y_predicted[-20:]
predicted_average = np.mean(values)

current = y_predicted[-21]

if current > predicted_average:
    signal='SELL'

elif current < predicted_average:
    signal='BUY'

else:
    signal='NEUTRAL'

st.subheader("Action: {}".format(signal))

fig=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original price')
plt.plot(y_predicted, 'r', label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader("Description of {}'s data from 2010".format(user_input))
st.write(df.describe())

st.subheader("Closing price vs Time Chart")
fig=plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing price vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader("Closing price vs Time Chart with 200MA")
ma200 = df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma200, 'g')
st.pyplot(fig)

st.subheader("Closing price vs Time Chart with 100MA and 200MA")
ma200 = df.Close.rolling(200).mean()
ma100 = df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)


add hoook for exporting product plans

payer integration sevices

file sil adapters jik member views members.py

export FileDescriptor
sort of record used to generate an excel

retrieve the specified field

view of members and add the export field