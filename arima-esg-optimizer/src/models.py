import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning
from tqdm import tqdm

try:
    import pmdarima as pm  # type: ignore
except ModuleNotFoundError:
    pm = None  # Allow graceful fallback when optional dependency missing

_PMDARIMA_WARNING_EMITTED = False

def fit_and_forecast_arima(series, cfg):
    """
    Fits an ARIMA model to a time series and forecasts one step ahead.
    Returns the forecasted mean and variance.
    """
    use_pmd = cfg['arima'].get('use_pmdarima', True)
    if use_pmd and pm is None:
        global _PMDARIMA_WARNING_EMITTED
        if not _PMDARIMA_WARNING_EMITTED:
            print("pmdarima not installed; falling back to statsmodels ARIMA.")
            _PMDARIMA_WARNING_EMITTED = True
        use_pmd = False

    if use_pmd:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model = pm.auto_arima(
                    series,
                    start_p=1, start_q=1,
                    max_p=cfg['arima']['max_p'],
                    max_q=cfg['arima']['max_q'],
                    d=cfg['arima']['max_d'],
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )

            forecast_mean, conf_int = model.predict(n_periods=1, return_conf_int=True, alpha=0.05)

            # CI = mean +/- 1.96 * std_err, so std_err = (upper - lower) / 3.92
            std_err = (conf_int[0][1] - conf_int[0][0]) / 3.92
            forecast_variance = std_err**2

            return float(forecast_mean[0]), float(forecast_variance)

        except Exception as e:
            # Fallback if auto_arima fails
            print(f"Auto ARIMA failed for {series.name}: {e}. Using mean/variance.")
            return np.mean(series), np.var(series)
    else:
        # Fallback to a standard ARIMA if pmdarima is disabled
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=ValueWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model = ARIMA(series, order=(1, 0, 1)).fit()
            forecast = model.get_forecast(steps=1)

            forecast_mean = forecast.predicted_mean.iloc[0]
            forecast_variance = forecast.se_mean.iloc[0]**2

            return float(forecast_mean), float(forecast_variance)
        except Exception as e:
            print(f"Standard ARIMA failed for {series.name}: {e}. Using mean/variance.")
            return np.mean(series), np.var(series)

def generate_forecasts(historical_returns, cfg):
    """
    Generates 1-week ahead return and variance forecasts for all tickers.
    """
    tickers = historical_returns.columns
    forecast_means = pd.Series(index=tickers, dtype=float)
    forecast_variances = pd.Series(index=tickers, dtype=float)

    print("Generating ARIMA forecasts for all assets...")
    for ticker in tqdm(tickers):
        series = historical_returns[ticker].dropna()
        mean, variance = fit_and_forecast_arima(series, cfg)
        forecast_means[ticker] = mean
        forecast_variances[ticker] = variance

    return forecast_means, forecast_variances
