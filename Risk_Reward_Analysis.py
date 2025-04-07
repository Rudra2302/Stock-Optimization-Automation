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

def portfolio_variance(weights, correlation_matrix, type):
    weights_sds = weights * std_devs
    m1 = np.dot(weights_sds, correlation_matrix)
    m2 = np.dot(m1, weights_sds.T)
    portfolio_variance_value = np.sqrt(m2)
    annual_portfolio_variance = portfolio_variance_value * np.sqrt(252)
    if type == "min":
        return annual_portfolio_variance
    else:
        return (-1) * annual_portfolio_variance
    
def portfolio_return(weights, mean_returns, type):
    expected_yearly_returns = mean_returns * 252
    eyr_by_weight = weights * expected_yearly_returns
    expected_portfolio_return = np.sum(eyr_by_weight)
    if type == "min":
        return expected_portfolio_return
    else:
        return (-1) * expected_portfolio_return

def find_max_min_variance(correlation_matrix, num_stocks):
    init_weights = np.array([1.0 / num_stocks] * num_stocks)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_stocks)]
    result_min = minimize(portfolio_variance, init_weights, args=(correlation_matrix, "min"),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    result_max = minimize(portfolio_variance, init_weights, args=(correlation_matrix, "max"),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result_min.x, result_max.x

def find_max_min_return(mean_returns, num_stocks):
    init_weights = np.array([1.0 / num_stocks] * num_stocks)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_stocks)]
    result_min = minimize(portfolio_return, init_weights, args=(mean_returns, "min"),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    result_max = minimize(portfolio_return, init_weights, args=(mean_returns, "max"),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result_min.x, result_max.x

if __name__ == "__main__":
    num_stocks = int(sys.argv[1])
    stock_symbols = sys.argv[2].split(',')

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(stock_symbols, start_date, end_date)
    daily_returns = calculate_daily_returns(stock_data)
    mean_returns, std_devs, covariance_matrix, correlation_matrix = calculate_statistics(daily_returns)

    min_risk_weights, max_risk_weights = find_max_min_variance(correlation_matrix, num_stocks)

    min_risk_weights_sds = min_risk_weights * std_devs
    m1 = np.dot(min_risk_weights_sds, correlation_matrix)
    m2 = np.dot(m1, min_risk_weights_sds.T)
    min_risk_portfolio_variance_value = np.sqrt(m2)
    min_risk = min_risk_portfolio_variance_value * np.sqrt(252)

    max_risk_weights_sds = max_risk_weights * std_devs
    m1 = np.dot(max_risk_weights_sds, correlation_matrix)
    m2 = np.dot(m1, max_risk_weights_sds.T)
    max_risk_portfolio_variance_value = np.sqrt(m2)
    max_risk = max_risk_portfolio_variance_value * np.sqrt(252)

    min_return_weights, max_return_weights = find_max_min_return(mean_returns, num_stocks)

    min_expected_yearly_returns = mean_returns * 252
    min_eyr_by_weight = min_return_weights * min_expected_yearly_returns
    min_return = np.sum(min_eyr_by_weight)

    max_expected_yearly_returns = mean_returns * 252
    max_eyr_by_weight = max_return_weights * max_expected_yearly_returns
    max_return = np.sum(max_eyr_by_weight)

    results = {
        'minRisk': min_risk,
        'maxRisk': max_risk,
        'minReturn': min_return,
        'maxReturn': max_return,
    }

    with open("analysis.txt", "w") as f:
        f.write(json.dumps(results, indent=4))
