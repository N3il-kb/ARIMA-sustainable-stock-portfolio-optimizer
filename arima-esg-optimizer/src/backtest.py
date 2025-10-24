import pandas as pd
import numpy as np
from tqdm import tqdm
from .models import generate_forecasts
from .optimize import compute_covariance_matrix, optimize_portfolio

def run_backtest(returns, esg, cfg):
    """
    Runs a rolling window backtest of the ARIMA+ESG portfolio strategy.
    """
    train_window = cfg['backtest']['train_window_weeks']
    n_periods = len(returns) - train_window

    # Initialize results storage
    portfolio_weights = pd.DataFrame(index=returns.index[train_window:], columns=returns.columns)
    portfolio_returns = pd.Series(index=returns.index[train_window:], dtype=float)

    print("Running weekly rebalancing backtest...")
    for t in tqdm(range(n_periods)):
        # Define the training and testing periods for this step
        train_start = t
        train_end = t + train_window

        # Get the historical returns for the training window
        historical_returns = returns.iloc[train_start:train_end]

        # 1. Forecast next period's returns (r_hat) and variances
        r_hat, _ = generate_forecasts(historical_returns, cfg)

        # 2. Compute the covariance matrix (Sigma)
        Sigma = compute_covariance_matrix(
            historical_returns,
            cfg['risk']['cov_method'],
            cfg['risk']['ewma_lambda']
        )

        # 3. Optimize the portfolio
        weights = optimize_portfolio(r_hat, Sigma, esg.values, cfg)

        # 4. Store the optimal weights
        portfolio_weights.iloc[t] = weights

        # 5. Calculate the realized portfolio return for the next period
        realized_returns = returns.iloc[train_end]
        portfolio_returns.iloc[t] = np.dot(weights, realized_returns)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Create a results dictionary
    backtest_results = {
        'weights': portfolio_weights,
        'returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
    }

    return backtest_results
