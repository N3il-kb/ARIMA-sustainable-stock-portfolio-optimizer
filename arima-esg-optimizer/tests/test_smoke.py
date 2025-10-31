from datetime import datetime

import pandas as pd
import pytest

from src import data as data_module


def test_cached_prices_loaded(tmp_path):
    """
    Ensures cached price loading works without needing yfinance.
    """
    tickers = ["AAA", "BBB"]
    cfg = {
        "tickers": tickers,
        "price": {
            "lookback_years": 1,
            "interval": "1wk",
            "auto_adjust": True,
        },
    }

    cache_file = data_module.CACHE_DIR / f"weekly_prices_{'_'.join(tickers)}.csv"
    if cache_file.exists():
        cache_file.unlink()

    dates = pd.date_range(end=datetime.today(), periods=10, freq="W")
    base_values = [[100 + idx, 110 + idx * 0.5] for idx in range(len(dates))]
    sample_prices = pd.DataFrame(base_values, index=dates, columns=tickers).round(2)
    sample_prices.to_csv(cache_file)

    loaded = data_module.load_cached_or_fetch(cfg)
    assert not loaded.empty
    assert list(loaded.columns) == tickers


@pytest.mark.integration
def test_prices_download():
    """
    Tests that the download_prices function returns a non-empty DataFrame.
    """
    pytest.importorskip("yfinance", reason="yfinance is required for this integration test.")
    try:
        df = data_module.download_prices(["AAPL", "MSFT"], 5, "1wk", True)
    except RuntimeError as exc:
        pytest.skip(f"Skipping download test due to data provider error: {exc}")
    assert not df.empty, "The downloaded prices DataFrame should not be empty."
    assert "AAPL" in df.columns and "MSFT" in df.columns, "The DataFrame should contain the requested tickers."
