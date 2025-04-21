import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Ticker for S&P 500 index
sp500 = yf.Ticker("^GSPC")

# Years you want
for year in range(2010, 2020):  # Looping through years 2010 to 2019
    # Start and end dates for the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)

    # Get historical data
    hist = sp500.history(start=start_date.strftime('%Y-%m-%d'),
                         end=end_date.strftime('%Y-%m-%d'))

    # Resample data by week and sum the volume
    weekly_data = hist[['Volume']].resample('W').sum()

    # Ensure we are only including complete weeks (from Sunday to Saturday)
    weekly_data = weekly_data[weekly_data.index.weekday == 6]  # Keep only weeks starting on Sunday

    if not weekly_data.empty:
        # Prepare the directory to save the CSV
        filename = f"./data/years/sp500_volume_{year}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the entire year's weekly volume data to a single CSV
        weekly_data.to_csv(filename, header=True)
        print(f"Saved: {filename}")
    else:
        print(f"No complete weekly data to save for {year}.")
