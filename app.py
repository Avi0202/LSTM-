import streamlit as st
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

st.title("LSTM Stock Price Predictor")

symbol = st.text_input("Enter stock symbol (e.g., AAPL):", "AAPL")
api_key = 'LLSGCNAU6ZQMK8KL'

# Add a button
if st.button('Run Analysis'):
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()
        df = data.copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        close_data = df['Close']
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data.values.reshape(-1, 1))
        
        # Create sequences for LSTM
        sequence_length = 60
        x, y = [], []

        for i in range(sequence_length, len(scaled_data)):
            x.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        x = np.array(x)
        y = np.array(y)
        x = x.reshape((x.shape[0], x.shape[1], 1))

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.2,
            shuffle=False
        )
        
        with st.spinner("Training LSTM model..."):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(
                x_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(x_test, y_test),
                verbose=0   # suppress output
            )
        
        # Predict on test
        y_pred_scaled = model.predict(x_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot results
        st.subheader("Actual vs Predicted Closing Price")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test_actual, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.legend()
        st.pyplot(fig)
        
        # Predict the next day's price
        last_60 = close_data.values[-sequence_length:]
        last_60_scaled = scaler.transform(last_60.reshape(-1, 1))
        x_input = last_60_scaled.reshape(1, sequence_length, 1)
        next_day_pred_scaled = model.predict(x_input)
        predicted_price = scaler.inverse_transform(next_day_pred_scaled)[0][0]
        st.success(f'Predicted next close price for {symbol}: ${predicted_price:.2f}')

    except Exception as e:
        st.error(f"Error: {e}")


    
    
    