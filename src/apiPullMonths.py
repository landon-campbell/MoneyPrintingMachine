#uses 'yfinance,' run 'pip install yfinance' in terminal

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Ticker for S&P 500 index
symbol = "AMZN"
sp500 = yf.Ticker(symbol)
index = "amazon"

# Years you want
for year in range(2010,2025):
    for month in range(1, 13):
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        # Get historical data
        hist = sp500.history(start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'))

        # Extract volume column
        volume_data = hist[['Volume']]

        os.makedirs(f"./data/{index} volume per day/{year}", exist_ok=True)

        # Create file name
        filename = f"./data/{index} volume per day/{year}/{symbol}_volume_{year}_{month:02}.csv"

        # Save to CSV
        volume_data.to_csv(filename)
        print(f"Saved: {filename}")
