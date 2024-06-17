# Implement Monte Carlo Method to simulate returns on stock portfolio

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
import yfinance as yf


# import the data
def get_data(stocks, start_date, end_date):
    # Get historical stock data from Yahoo Finance and select the close price
    data = yf.download(stocks, start=start_date, end=end_date)
    data = data['Close']

    # Calculate the daily returns
    returns = data.pct_change()

    # get the mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    return mean_returns, cov_matrix


stock_list = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CSCO']
# To add the stocks you want to simulate per region, add an extension for example '.AX' for Australian stocks

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365)
# time range for the simulation is 1 year

mean_returns, cov_matrix = get_data(stock_list, start_date, end_date)
print(mean_returns)


weights = np.random.random(len(mean_returns))
weights = weights / np.sum(weights) # normalise the weights
