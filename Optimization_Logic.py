import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
import argparse
import json

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
    
    # Excess returns matrix
    excess_return_matrix = returns_df - returns_df.mean()
    
    # Product of standard deviations
    product_of_sds = np.outer(std_devs, std_devs)
    
    # Variance-Covariance Matrix
    n = len(returns_df)
    covariance_matrix = np.dot(excess_return_matrix.T, excess_return_matrix) / n
    
    # Correlation Matrix
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

def main(args):
    num_stocks = args.num_stocks
    stock_symbols = args.stock_symbols
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch stock data
    stock_data = fetch_stock_data(stock_symbols, start_date, end_date)
    
    # Calculate daily returns
    daily_returns = calculate_daily_returns(stock_data)
    
    # Calculate statistics
    mean_returns, std_devs, covariance_matrix, correlation_matrix = calculate_statistics(daily_returns)
    
    # Create weight array
    weights = np.array([1.0 / num_stocks] * num_stocks)
    
    # Create weights by SDs
    weights_sds = weights * std_devs
    
    # Calculate M1 and M2
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    
    # Portfolio Variance
    portfolio_variance_value = np.sqrt(m2)
    
    # Expected yearly returns
    expected_yearly_returns = mean_returns * 252
    
    # EYR by weight
    eyr_by_weight = weights * expected_yearly_returns
    
    # Expected Portfolio Return
    expected_portfolio_return = np.sum(eyr_by_weight)
    
    # Annual Portfolio Variance
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)
    
    # Optimize weights
    optimized_weights = optimize_portfolio(covariance_matrix, num_stocks)
    
    # Output results
    results = {
        'optimized_weights': optimized_weights.tolist(),
        'expected_portfolio_return': expected_portfolio_return,
        'annual_portfolio_variance': annual_portfolio_variance
    }

    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Optimization Script')
    parser.add_argument('--num_stocks', type=int, required=True, help='Number of stocks')
    parser.add_argument('--stock_symbols', type=str, nargs='+', required=True, help='List of stock symbols')
    
    args = parser.parse_args()
    main(args)
