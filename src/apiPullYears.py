import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Create output directory
output_dir = "./data/years/"
os.makedirs(output_dir, exist_ok=True)

# Ticker for S&P 500 Index
sp500 = yf.Ticker("^GSPC")

# Loop through years
for year in range(2005, 2016):
    # Define start and end of year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)

    # Download daily data for the year
    hist = sp500.history(start=start_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"))

    # Reset index to access date column
    hist = hist.reset_index()

    # Create 'Date' column in format YYYY-MM
    hist['Date'] = hist['Date'].dt.to_period('M').astype(str)

    # Group by 'Date' and sum the volume
    monthly_volume = hist.groupby('Date')['Volume'].sum().reset_index()

    # Save to CSV file
    filename = f"{output_dir}/sp500_volume_{year}.csv"
    monthly_volume.to_csv(filename, index=False)
    print(f"Saved: {filename}")
