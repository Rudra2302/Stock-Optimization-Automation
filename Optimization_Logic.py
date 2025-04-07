import sys
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime
from datetime import datetime, timedelta
import json

import yfinance as yf

def fetch_stock_data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        if "." in symbol and not symbol.endswith((".NS", ".BO")):
            yf_symbol = symbol.replace('.', '-')
        else:
            yf_symbol = symbol

        stock_data = yf.download(yf_symbol, start=start_date, end=end_date)

        if 'Close' in stock_data.columns:
            data[symbol] = stock_data['Close'].dropna().values.tolist()
        else:
            data[symbol] = []  

    return data

def calculate_daily_returns(data):
    returns = {}
    for symbol, prices in data.items():
        returns[symbol] = [((prices[i][0] / prices[i - 1][0]) - 1) * 100 for i in range(1, len(prices))]
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

def portfolio_variance_minimization(weights, correlation_matrix, threshold):
    weights_sds = weights * std_devs
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    portfolio_variance_value = np.sqrt(m2)
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)
    if annual_portfolio_variance >= threshold:
        return annual_portfolio_variance
    else:
        return 100
    
def portfolio_variance_maximization(weights, correlation_matrix):
    weights_sds = weights * std_devs
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    portfolio_variance_value = np.sqrt(m2)
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)
    return -annual_portfolio_variance

def portfolio_return_minimization(weights, mean_returns, threshold):
    expected_yearly_returns = mean_returns * 252
    eyr_by_weight = weights * expected_yearly_returns
    expected_portfolio_return = np.sum(eyr_by_weight)
    if expected_portfolio_return >= threshold:
        return expected_portfolio_return
    else:
        return 100

def portfolio_return_maximization(weights, mean_returns):
    expected_yearly_returns = mean_returns * 252
    eyr_by_weight = weights * expected_yearly_returns
    expected_portfolio_return = np.sum(eyr_by_weight)
    return -expected_portfolio_return

def portfolio_return_risk_minimization(weights, correlation_matrix, mean_returns):
    weights_sds = weights * std_devs
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    portfolio_variance_value = np.sqrt(m2)
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)

    expected_yearly_returns = mean_returns * 252
    eyr_by_weight = weights * expected_yearly_returns
    expected_portfolio_return = np.sum(eyr_by_weight)

    if annual_portfolio_variance == 0:
        annual_portfolio_variance = 0.0001

    return (-1)*expected_portfolio_return/annual_portfolio_variance

def optimize_portfolio_risk(correlation_matrix, num_stocks, threshold):
    init_weights = np.array([1.0 / num_stocks] * num_stocks)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_stocks)]

    init_weights_sds = init_weights * std_devs
    m1 = np.dot(init_weights_sds, correlation_matrix)
    m2 = np.dot(m1, init_weights_sds.T)
    init_portfolio_variance_value = np.sqrt(m2)
    init_annual_portfolio_variance = init_portfolio_variance_value * np.sqrt(252)

    if init_annual_portfolio_variance >= threshold:
        result = minimize(portfolio_variance_minimization, init_weights, args=(correlation_matrix, threshold),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(portfolio_variance_maximization, init_weights, args=(correlation_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
        result = minimize(portfolio_variance_minimization, result.x, args=(correlation_matrix, threshold),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimize_portfolio_return(mean_returns, num_stocks, threshold):
    init_weights = np.array([1.0 / num_stocks] * num_stocks)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_stocks)]

    init_expected_yearly_returns = mean_returns * 252
    init_eyr_by_weight = init_weights * init_expected_yearly_returns
    init_expected_portfolio_return = np.sum(init_eyr_by_weight)

    if init_expected_portfolio_return >= threshold:
        result = minimize(portfolio_return_minimization, init_weights, args=(mean_returns, threshold),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(portfolio_return_maximization, init_weights, args=(mean_returns),
                      method='SLSQP', bounds=bounds, constraints=constraints)
        result = minimize(portfolio_return_minimization, result.x, args=(mean_returns, threshold),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def optimize_portfolio_return_risk(correlation_matrix, mean_returns, num_stocks):
    init_weights = np.array([1.0 / num_stocks] * num_stocks)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_stocks)]

    result = minimize(portfolio_return_risk_minimization, init_weights, args=(correlation_matrix, mean_returns),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

if __name__ == "__main__":
    num_stocks = int(sys.argv[1])
    stock_symbols = sys.argv[2].split(',')
    threshold = int(sys.argv[3])
    type = sys.argv[4]

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(stock_symbols, start_date, end_date)
    daily_returns = calculate_daily_returns(stock_data)
    mean_returns, std_devs, covariance_matrix, correlation_matrix = calculate_statistics(daily_returns)

    if type == "risk":
        optimized_weights = optimize_portfolio_risk(correlation_matrix, num_stocks, threshold)
    elif type == "return":
        optimized_weights = optimize_portfolio_return(mean_returns, num_stocks, threshold)
    else:
        optimized_weights = optimize_portfolio_return_risk(correlation_matrix, mean_returns, num_stocks)

    weights_sds = optimized_weights * std_devs
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    portfolio_variance_value = np.sqrt(m2)
    expected_yearly_returns = mean_returns * 252
    eyr_by_weight = optimized_weights * expected_yearly_returns
    expected_portfolio_return = np.sum(eyr_by_weight)
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)
    if annual_portfolio_variance == 0:
        annual_portfolio_variance = 0.0001
    expected_ratio = expected_portfolio_return / annual_portfolio_variance

    results = {
        'optimized_weights': optimized_weights.tolist(),
        'expected_portfolio_return': expected_portfolio_return/100,
        'annual_portfolio_variance': annual_portfolio_variance/100,
        'expected_ratio': expected_ratio,
    }

    with open("results.txt", "w") as f:
        f.write(json.dumps(results, indent=4))
