# Implement Monte Carlo Method to simulate returns on stock portfolio

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from pandas_datareader import data as pdr


# import the data
def get_data(stocks, start_date, end_date):
    # Get historical stock data from Yahoo Finance and select the close price
    data = pdr.get_data_yahoo(stocks, start=start_date, end=end_date)
    data = data['Close']

    # Calculate the daily returns
    returns = data.pct_change()

    # get the mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    return mean_returns, cov_matrix


stock_list = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CSCO']
# add the extension '.AX' to the stock_list to get the Australian stock data
stock_list = [stock + '.AX' for stock in stock_list]