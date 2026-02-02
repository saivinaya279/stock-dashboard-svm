import yfinance as yf
import os

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

stocks = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "NFLX",
    "IBM",
    "ORCL"
]

for stock in stocks:
    print(f"Downloading {stock} data...")
    df = yf.download(stock, start="2018-01-01", end="2024-01-01")
    df.to_csv(f"data/{stock}.csv")

print("All stock data downloaded successfully!")
