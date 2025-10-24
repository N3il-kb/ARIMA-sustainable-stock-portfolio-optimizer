from src.data import download_prices

def test_prices_download():
    """
    Tests that the download_prices function returns a non-empty DataFrame.
    """
    df = download_prices(["AAPL", "MSFT"], 5, "1wk", True)
    assert not df.empty, "The downloaded prices DataFrame should not be empty."
    assert "AAPL" in df.columns and "MSFT" in df.columns, "The DataFrame should contain the requested tickers."
