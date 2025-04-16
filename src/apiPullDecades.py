import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Output folder
output_dir = "./data/decade/"
os.makedirs(output_dir, exist_ok=True)

# Ticker for S&P 500
sp500 = yf.Ticker("^GSPC")

# Define decades and their year ranges
decades = {
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
}

for label, (start_year, end_year) in decades.items():
    # Download full decade of data
    hist = sp500.history(start=f"{start_year}-01-01", end=f"{end_year+1}-01-01")
    hist = hist.reset_index()

    # Extract year from date
    hist['Date'] = hist['Date'].dt.year

    # Group by year and sum volume
    yearly_volume = hist.groupby('Date')['Volume'].sum().reset_index()

    # Save to CSV
    filename = f"{output_dir}/sp500_volume_{label}.csv"
    yearly_volume.to_csv(filename, index=False)
    print(f"Saved: {filename}")
