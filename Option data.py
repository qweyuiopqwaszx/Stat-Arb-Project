import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

apple = yf.Ticker("AAPL")

# Get option expiration dates
expiration_dates = apple.options
print("Available expiration dates:", expiration_dates)

# Get option chain for a specific expiration date
if expiration_dates:
    # Use the first available expiration date
    opt = apple.option_chain(expiration_dates[0])
    
    # Calls data
    calls = opt.calls
    print("\nCalls data:")
    print(calls.head())
    
    # Puts data
    puts = opt.puts
    print("\nPuts data:")
    print(puts.head())