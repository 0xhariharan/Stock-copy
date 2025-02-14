import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Cache the data loading function for performance improvement
@st.cache
def load_data(ticker_symbol):
    stock_data = yf.Ticker(ticker_symbol)
    stock_history = stock_data.history(start="2001-01-01", actions=False)[["Open", "High", "Low", "Close"]]
    
    # Convert datetime index to date datatype
    stock_history.index = pd.to_datetime(stock_history.index).date
    stock_history['Date'] = stock_history.index  # No need to convert datetime to date 
        
    final_df = stock_history[["Close"]]
    
    # Set index and column names to None
    final_df.index.name = None
    final_df.columns.name = None
    
    return final_df

def main():
    st.title('Stock Price Forecasting with ARIMA Model')

    # List of Indian stock ticker symbols (use NSE for Indian stocks)
    tickers = ['NSE:TCS', 'NSE:INFY', 'NSE:RELIANCE', 'NSE:HDFCBANK', 'NSE:SBIN', 'NSE:BAJAJ-AUTO']  # Update this with more stocks
    ticker_symbol = st.sidebar.selectbox('Select Ticker Symbol', tickers)

    final_df = load_data(ticker_symbol)

    order_p = st.sidebar.slider('Order p', 0, 10, 2)
    order_d = st.sidebar.slider('Order d', 0, 10, 1)
    order_q = st.sidebar.slider('Order q', 0, 10, 2)

    # Fit ARIMA model
    model = ARIMA(final_df["Close"], order=(order_p, order_d, order_q))
    model_fit = model.fit()

    n_train = len(final_df) - 90
    train = final_df["Close"][:n_train]
    test = final_df["Close"][n_train:]

    predictions = model_fit.predict(start=n_train, end=len(final_df) - 1, typ='levels')

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((test - predictions) / test)) * 100

    st.write(f"Mean Absolute Percentage Error (MAPE) on the test set: {mape:.2f}%")

    # Forecasting the next 90 days
    forecast = model_fit.forecast(steps=90)
    forecast_dates = pd.date_range(final_df.index[-1], periods=91)[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    
    # Set index and column names to None for forecast_df
    forecast_df.index.name = None
    forecast_df.columns.name = None

    # Plot the actual, predicted, and forecasted stock prices
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(final_df.index, final_df["Close"], label='Actual')
    ax.plot(predictions.index, predictions, label='Predictions', color='red')
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='green')
    ax.legend()

    st.pyplot(fig)

    # Display the first 10 forecasted prices
    st.write("Predictions:")    
    st.write(forecast_df[['Date', 'Forecast']].head(10).set_index('Date').round(2))  # Display only 'Date' and 'Forecast' columns

if __name__ == '__main__':
    main()
