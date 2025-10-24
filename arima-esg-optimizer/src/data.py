import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def download_prices(tickers, lookback_years, interval, auto_adjust):
    """
    Downloads historical price data for a list of tickers.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False
    )['Close']

    return df

def load_cached_or_fetch(cfg):
    """
    Loads cached price data if it exists, otherwise fetches it from yfinance.
    """
    cache_path = f"data/cache/weekly_prices_{'_'.join(cfg['tickers'])}.csv"

    if os.path.exists(cache_path):
        print("Loading cached prices...")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("Fetching prices...")
        prices = download_prices(
            cfg['tickers'],
            cfg['price']['lookback_years'],
            cfg['price']['interval'],
            cfg['price']['auto_adjust']
        )
        prices.to_csv(cache_path)
        print(f"Saved prices to {cache_path}")

    return prices

def to_weekly_returns(prices):
    """
    Computes weekly log returns from a DataFrame of prices.
    """
    return pd.DataFrame(data=np.log(prices / prices.shift(1)).dropna(), index=prices.index[1:], columns=prices.columns)
