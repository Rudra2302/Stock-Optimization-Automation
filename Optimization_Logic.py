import sys
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime
from datetime import datetime, timedelta
import json  # Import json module for output

def fetch_stock_data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        data[symbol] = stock_data['Close'].tolist()
    return data

def calculate_daily_returns(data):
    returns = {}
    for symbol, prices in data.items():
        returns[symbol] = [((prices[i] / prices[i - 1]) - 1) * 100 for i in range(1, len(prices))]
    return returns

def calculate_statistics(daily_returns):
    returns_df = pd.DataFrame(daily_returns)
    mean_returns = returns_df.mean()
    std_devs = returns_df.std()
    
    excess_return_matrix = returns_df - returns_df.mean()
    product_of_sds = np.outer(std_devs, std_devs)
    
    n = len(returns_df)
    covariance_matrix = np.dot(excess_return_matrix.T, excess_return_matrix) / n
    correlation_matrix = covariance_matrix / product_of_sds
    np.fill_diagonal(correlation_matrix, 1)
    
    return mean_returns, std_devs, covariance_matrix, correlation_matrix

def portfolio_variance(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

def optimize_portfolio(covariance_matrix, num_stocks):
    init_weights = np.array([1.0 / num_stocks] * num_stocks)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_stocks)]
    result = minimize(portfolio_variance, init_weights, args=(covariance_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

if __name__ == "__main__":
    num_stocks = int(sys.argv[1])
    stock_symbols = sys.argv[2].split(',')

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(stock_symbols, start_date, end_date)
    daily_returns = calculate_daily_returns(stock_data)
    mean_returns, std_devs, covariance_matrix, correlation_matrix = calculate_statistics(daily_returns)

    weights = np.array([1.0 / num_stocks] * num_stocks)
    weights_sds = weights * std_devs
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    portfolio_variance_value = np.sqrt(m2)
    expected_yearly_returns = mean_returns * 252
    eyr_by_weight = weights * expected_yearly_returns
    expected_portfolio_return = np.sum(eyr_by_weight)
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)

    optimized_weights = optimize_portfolio(covariance_matrix, num_stocks)

    results = {
        'optimized_weights': optimized_weights.tolist(),
        'expected_portfolio_return': expected_portfolio_return/100,
        'annual_portfolio_variance': annual_portfolio_variance/100
    }

    with open("results.txt", "w") as f:
        f.write(json.dumps(results, indent=4))
