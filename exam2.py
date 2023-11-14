import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Fetch the stock data for Tesla for the past 30 days
ticker = "TSLA"
period = "1mo"  # approximately 30 days
interval = "1d"  # daily data

# Download the data
data = yf.download(ticker, period=period, interval=interval)

# Prepare data for linear regression
# We'll use days as our X (independent variable) and closing prices as Y (dependent variable)
X = np.arange(len(data)).reshape(-1, 1)
y = data['Close'].values

# Standardize the features (important for SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the model using SGD
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_scaled, y)

# Make predictions
y_pred = model.predict(X_scaled)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='Actual Prices')
plt.plot(data.index, y_pred, label='Predicted Prices (Linear Regression)', color='red')
plt.title('Tesla Stock Price - Last 30 Days with Linear Regression Fit')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
