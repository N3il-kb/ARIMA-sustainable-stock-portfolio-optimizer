# ARIMA + ESG Portfolio Optimizer

This project implements a quantitative investment strategy that combines ARIMA time series forecasting with Environmental, Social, and Governance (ESG) metrics to build and backtest a long-only, weekly rebalanced portfolio.

## Overview

The core of the strategy is to:

1.  **Forecast Returns:** Use ARIMA models to forecast the one-week-ahead returns for a universe of large-cap stocks.
2.  **Incorporate ESG:** Integrate a measure of a company's ESG performance into the portfolio construction process.
3.  **Optimize Portfolio:** Employ mean-variance optimization with an added ESG term to find the optimal asset allocation.
4.  **Backtest:** Simulate the strategy's performance over a historical period using a rolling window approach.

The goal is to create a portfolio that not only seeks to maximize risk-adjusted returns but also aligns with investors' sustainability preferences.

## Installation & Usage

To get started with this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/arima-esg-optimizer.git
    cd arima-esg-optimizer
    ```

2.  **Set up the environment and install dependencies:**
    ```bash
    make setup
    ```
    This will create a virtual environment, activate it, and install all the required Python packages from `requirements.txt`.

3.  **Launch Jupyter Notebook:**
    ```bash
    . venv/bin/activate
    jupyter notebook
    ```

4.  **Run the notebook:**
    Open `notebooks/01_arima_esg_portfolio.ipynb` and run the cells to execute the full backtest and generate the analysis.

## Configurable Parameters

The behavior of the backtest is controlled by the `config/settings.yaml` file. Here are some of the key parameters you can adjust:

*   `tickers`: The list of stock tickers to include in the portfolio.
*   `price.lookback_years`: The number of years of historical price data to download.
*   `opt.alpha_risk_aversion`: The risk aversion parameter (α) in the optimization objective function. A higher value means a stronger penalty for risk.
*   `opt.beta_esg_pref`: The ESG preference parameter (β) in the optimization objective function. A higher value means a stronger reward for ESG performance.
*   `opt.weight_max`: The maximum weight that can be allocated to any single asset in the portfolio.

## ESG Data Sources & Caveats

The ESG scores are fetched from Yahoo Finance. The primary source is the `.sustainability` attribute of the `yfinance` Ticker object. If this is not available, the code falls back to scraping the ESG score from the Yahoo Finance sustainability webpage.

**Important Caveats:**

*   **Data Availability:** ESG data can be sparse and inconsistent across different providers and for different companies.
*   **Scoring Methodology:** ESG scores are subjective and can vary significantly depending on the provider's methodology.
*   **Data Inversion:** In this model, lower raw ESG scores are considered better. The scores are inverted and normalized to a 0-1 scale, where a higher value indicates better ESG performance.
*   **Static Scores:** The current implementation uses static ESG scores. A more advanced approach would be to use historical ESG data to avoid lookahead bias.
