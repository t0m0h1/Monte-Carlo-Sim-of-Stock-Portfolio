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
# print(weights)



# Monte Carlo Simulation
num_sims = 100
timeframe = 100 # number of days to simulate
mean_m = np.full(shape=(timeframe, len(weights)), fill_value=mean_returns) # mean returns matrix
mean_m = mean_m.T # transpose the matrix

initial_investment = 10000

portfolio = np.full(shape=(timeframe, num_sims), fill_value=0.0)

for i in range(0, num_sims):
    samples = np.random.normal(size=(timeframe, len(weights)))
    lower = np.linalg.cholesky(cov_matrix)
    daily_returns = mean_m + np.inner(lower, samples)

    portfolio[:, i] = np.cumprod(np.inner(weights, daily_returns.T) + 1)


plt.plot(portfolio)
plt.ylabel('Portfolio Value')
plt.xlabel('Time')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.show()