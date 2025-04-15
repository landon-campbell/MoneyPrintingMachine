import yfinance as yf
import pandas as pd
import os

# Download hourly SPY data for the last 60 days
data = yf.download("SPY", interval="1h", period="60d")

# Remove timezone info from index
data.index = data.index.tz_localize(None)

# Reset index so Datetime becomes a column
data = data.reset_index()

# Keep only 'Datetime' and 'Volume', rename 'Datetime' to 'Date'
data = data[['Datetime', 'Volume']].rename(columns={'Datetime': 'Date'})

# Output folder
output_dir = "./data/days/"
os.makedirs(output_dir, exist_ok=True)

# Group by date part of the timestamp
data['OnlyDate'] = data['Date'].dt.date
for only_date, group in data.groupby('OnlyDate'):
    # Format filename
    os.makedirs(f"{output_dir}/{only_date.month:02}/", exist_ok=True)
    filename = f"{output_dir}/{only_date.month:02}/spy_volume_{only_date.year}_{only_date.month:02}_{only_date.day:02}.csv"
    
    # Drop the helper column and save
    group[['Date', 'Volume']].to_csv(filename, index=False)
    print(f"Saved: {filename}")
