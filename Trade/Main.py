import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import datetime
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.models import load_model

# Define the directory to save the model
model_directory = "saved_models"
os.makedirs(model_directory, exist_ok=True)
model_file_path = os.path.join(model_directory, "lstm_crypto_predictor.h5")

# Initialize the exchange
exchange = ccxt.binance()

# Define the cryptocurrency symbol and market
symbol = "BTC"
market = "USDT"

# Function to fetch historical data
def fetch_historical_data(exchange, symbol, market, start_date, timeframe="30m", max_points_per_request=1000):
    data = []
    since = int(start_date.timestamp() * 1000)

    while True:
        fetched_data = exchange.fetch_ohlcv(f"{symbol}/{market}", timeframe=timeframe, since=since, limit=max_points_per_request)
        if len(fetched_data) == 0:
            break

        data += fetched_data
        since = data[-1][0] + 1

    return data

start_date = datetime.datetime.now() - datetime.timedelta(days=3*365)
ohlcv = fetch_historical_data(exchange, symbol, market, start_date)

# Preprocess the data
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)

def fibonacci_retracement_levels(highest_high, lowest_low):
    difference = highest_high - lowest_low
    levels = [lowest_low,
              lowest_low + difference * 0.236,
              lowest_low + difference * 0.382,
              lowest_low + difference * 0.5,
              lowest_low + difference * 0.618,
              lowest_low + difference * 0.786,
              highest_high]
    return levels


# Calculate Fibonacci retracement levels
lookback_period = 100
df['highest_high'] = df['high'].rolling(window=lookback_period).max()
df['lowest_low'] = df['low'].rolling(window=lookback_period).min()
df['fib_levels'] = df.apply(lambda row: fibonacci_retracement_levels(row['highest_high'], row['lowest_low']), axis=1)

# Extract individual Fibonacci retracement levels
for i, level in enumerate([0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]):
    df[f'fib_level_{i}'] = df['fib_levels'].apply(lambda levels: levels[i])

# Drop the 'fib_levels' column
df = df.drop(columns=['fib_levels'])

def generate_y(df, n=24):
    y = np.zeros((len(df) - n, n))
    for i in range(len(df) - n):
        y[i, :] = df["close"].iloc[i + 1: i + n + 1].values
    return y



# Calculate Moving Averages
df['MA_10'] = df['close'].rolling(window=10).mean()

# Calculate the Ichimoku Cloud components
ichimoku_indicator = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
df['ichimoku_tenkan_sen'] = ichimoku_indicator.ichimoku_conversion_line()
df['ichimoku_kijun_sen'] = ichimoku_indicator.ichimoku_base_line()
df['ichimoku_senkou_span_a'] = ichimoku_indicator.ichimoku_a()
df['ichimoku_senkou_span_b'] = ichimoku_indicator.ichimoku_b()
df['ichimoku_chikou_span'] = df['close'].shift(-26)

# Calculate RSI
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

# Calculate MACD
macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd_indicator.macd()
df['MACD_signal'] = macd_indicator.macd_signal()


# Add features (e.g., moving averages)
df["SMA_5"] = df["close"].rolling(window=5).mean()
df["SMA_10"] = df["close"].rolling(window=10).mean()

# Drop rows with NaN values
df = df.dropna()

# Prepare data for training and testing
X = df[["SMA_5", "SMA_10", "MA_10", "RSI", "MACD", "MACD_signal", "ichimoku_tenkan_sen", "ichimoku_kijun_sen", "ichimoku_senkou_span_a", "ichimoku_senkou_span_b", "ichimoku_chikou_span"]].values
y = generate_y(df, n=24)

def generate_y(df, n=24):
    y = np.zeros((len(df) - n, n))
    for i in range(len(df) - n):
        y[i, :] = df["close"].iloc[i + 1: i + n + 1].values
    return y

X = X[:len(y), :]

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Define the LSTM model with dropout
model = Sequential()
model.add(LSTM(50, input_shape=(1, 11), activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(24, kernel_regularizer=l2(0.001)))
model.compile(optimizer="adam", loss="mean_squared_error")



# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)
mse_list = []
r2_list = []
mae_list = []
rmse_list = []

n_steps = 1

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index, :], y[test_index, :]

    # Reshape the input data for LSTM layers
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Train the model
    model.fit(X_train, y_train, epochs=100000, batch_size=8, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Predict the closing prices for the testing set
    y_pred = model.predict(X_test)
    
    # Prepare the last available data point in the correct format for prediction
    X_last = X[-1].reshape(1, 1, -1)

    # Generate predictions for the next 24 days
    next_24_days_pred = model.predict(X_last)

    # Invert the scaling
    next_24_days_pred = scaler.inverse_transform(next_24_days_pred)

    # Print the predictions with the corresponding date and time
    for i in range(24):
        prediction_date = df.index[-1] + datetime.timedelta(hours=i + 1)
        print(f"Day {i + 1} ({prediction_date}): {next_24_days_pred[0][i]:.2f} {market}")

    # Invert the scaling
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    # Evaluate the model for each step
    mse = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(n_steps)]
    mae = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(n_steps)]
    rmse = [np.sqrt(m) for m in mse]  # Calculate RMSE from MSE
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(n_steps)]

    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)


# Train and evaluate the model only once
model.fit(X_train, y_train, epochs=100000, batch_size=8, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
y_pred = model.predict(X_test)

# Save the trained model to a file
model.save(model_file_path)

X_last = X[-1].reshape(1, 1, -1)

mse_avg = np.mean(mse_list, axis=0)
mae_avg = np.mean(mae_list, axis=0)
rmse_avg = np.mean(rmse_list, axis=0)
r2_avg = np.mean(r2_list, axis=0)

for i in range(n_steps):
    print(f"Step {i + 1} - Average Mean Squared Error: {mse_avg[i]}")
    print(f"Step {i + 1} - Average Mean Absolute Error: {mae_avg[i]}")
    print(f"Step {i + 1} - Average Root Mean Squared Error: {rmse_avg[i]}")
    print(f"Step {i + 1} - Average R-squared: {r2_avg[i]}")

# Visualize the results
train_index, test_index = list(tscv.split(X))[-1]  # Get the last split
X_test = X[test_index].reshape((-1, 1, 11))  # Reshape X_test here
y_test = y[test_index]
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Prepare data for mplfinance
plot_df = df.iloc[test_index[0]:-24].copy()
plot_df["Predicted"] = y_pred[:, 0]  # Select the first column of y_pred

# Plot candlestick chart with predicted line
s = mpf.make_mpf_style(base_mpl_style="seaborn", gridstyle="")
ap0 = [mpf.make_addplot(plot_df["Predicted"], panel=0, color='orange', secondary_y=False, linestyle='--', ylabel='Predicted Price (USDT)')]
mpf.plot(plot_df, type='candle', style=s, ylabel="Price (USDT)", title=f"{symbol}/{market} Closing Prices", addplot=ap0, figratio=(14, 6), volume=True)

# Display the last predicted price
last_predicted_price = y_pred[-1][0]
print(f"Last predicted price: {last_predicted_price:.2f} {market}")

