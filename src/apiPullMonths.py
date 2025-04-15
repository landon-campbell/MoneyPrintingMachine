#uses 'yfinance,' run 'pip install yfinance' in terminal

import yfinance as yf
import pandas as pd
from datetime import datetime

# Ticker for S&P 500 index
sp500 = yf.Ticker("^GSPC")

# Years you want
for year in [2022, 2023]:
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

        # Create file name
        filename = f"./data/months/{year}/sp500_volume_{year}_{month:02}.csv"

        # Save to CSV
        volume_data.to_csv(filename)
        print(f"Saved: {filename}")
