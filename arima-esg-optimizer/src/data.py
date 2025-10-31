from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yf = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_prices(
    tickers: Sequence[str],
    lookback_years: int,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    """
    Downloads historical price data for a list of tickers.
    """
    if yf is None:
        raise ModuleNotFoundError(
            "yfinance is not installed. Install it with `pip install yfinance` "
            "or add a cached prices file under data/cache/."
        )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for tickers {tickers}. "
            "Check ticker symbols or ensure network access."
        )

    # yf.download returns a column MultiIndex when multiple tickers provided.
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.dropna(how="all")

    return close


def load_cached_or_fetch(cfg: dict) -> pd.DataFrame:
    """
    Loads cached price data if it exists, otherwise fetches it from yfinance.
    """
    cache_path = CACHE_DIR / f"weekly_prices_{'_'.join(cfg['tickers'])}.csv"

    if cache_path.exists():
        print("Loading cached prices...")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("Fetching prices...")
        prices = download_prices(
            cfg["tickers"],
            cfg["price"]["lookback_years"],
            cfg["price"]["interval"],
            cfg["price"]["auto_adjust"],
        )
        prices.to_csv(cache_path)
        print(f"Saved prices to {cache_path.relative_to(PROJECT_ROOT)}")

    return prices


def to_weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes weekly log returns from a DataFrame of prices.
    """
    aligned_prices = prices.sort_index()
    log_returns = np.log(aligned_prices / aligned_prices.shift(1))
    log_returns = log_returns.dropna(how="all")
    return log_returns
