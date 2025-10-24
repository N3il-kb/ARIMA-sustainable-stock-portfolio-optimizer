import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def get_esg_from_yfinance(ticker):
    """Fetches ESG score from yfinance .sustainability attribute."""
    try:
        t = yf.Ticker(ticker)
        sustainability = t.sustainability
        if sustainability is not None and not sustainability.empty:
            return sustainability.loc['totalEsg']['Value']
    except Exception:
        return None
    return None

def get_esg_from_html(ticker):
    """Fetches ESG score by scraping the Yahoo Finance sustainability page."""
    url = f"https://finance.yahoo.com/quote/{ticker}/sustainability"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find the div containing the Total ESG Score
        esg_div = soup.find('div', {'class': 'Fz(36px) Fw(600) D(ib) Mend(5px)'})
        if esg_div:
            return float(esg_div.text)
    except Exception:
        return None
    return None

def collect_esg_scores(tickers, source_priority):
    """
    Collects ESG scores for a list of tickers using the specified source priority.
    """
    esg_scores = {}
    print("Collecting ESG scores...")
    for ticker in tqdm(tickers):
        score = None
        for source in source_priority:
            if source == 'yfinance':
                score = get_esg_from_yfinance(ticker)
            elif source == 'yahoo_html':
                score = get_esg_from_html(ticker)

            if score is not None:
                break  # Found a score, move to the next ticker

        esg_scores[ticker] = score

    return pd.Series(esg_scores, name="esg_score")

def normalize_esg(esg_scores, method="zscore_to_01"):
    """
    Normalizes ESG scores.
    """
    # Lower ESG score is better, so we invert the scores before normalizing
    inverted_scores = esg_scores.max() - esg_scores

    # Handle missing values by filling with the median
    if inverted_scores.isnull().any():
        inverted_scores.fillna(inverted_scores.median(), inplace=True)

    if method == "zscore_to_01":
        # First, apply z-score normalization
        z_scores = (inverted_scores - inverted_scores.mean()) / inverted_scores.std()

        # Then, scale to a 0-1 range using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform(z_scores.values.reshape(-1, 1)).flatten()
        return pd.Series(normalized_scores, index=inverted_scores.index)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
