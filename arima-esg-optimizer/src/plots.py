from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_cumulative_returns(backtest_results, cfg):
    """Plots the cumulative returns of the portfolio."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_results['cumulative_returns'].index,
        y=backtest_results['cumulative_returns'],
        mode='lines',
        name='ARIMA+ESG Portfolio'
    ))
    fig.update_layout(
        title='Cumulative Portfolio Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        template='plotly_dark'
    )
    return fig

def plot_weights_heatmap(backtest_results, cfg):
    """Plots a heatmap of the portfolio weights over time."""
    weights = backtest_results['weights']
    fig = go.Figure(data=go.Heatmap(
        z=weights.T,
        x=weights.index,
        y=weights.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title='Portfolio Weights Heatmap',
        xaxis_title='Date',
        yaxis_title='Ticker',
        template='plotly_dark'
    )
    return fig

def plot_risk_return_esg_frontier(returns, esg, cfg):
    """
    Plots a simulated risk-return frontier, with ESG scores represented by color.
    Note: This is a simplified representation and not a true efficient frontier.
    """
    # Simulate a set of random portfolios to illustrate the trade-offs
    n_portfolios = 5000
    n_assets = len(returns.columns)
    sim_returns, sim_vols, sim_esg = [], [], []

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights) # Normalize

        # Portfolio metrics
        ret = np.sum(returns.mean() * weights) * 52 # Annualized
        vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 52, weights)))
        esg_score = np.dot(weights, esg)

        sim_returns.append(ret)
        sim_vols.append(vol)
        sim_esg.append(esg_score)

    fig = go.Figure(data=go.Scatter(
        x=sim_vols,
        y=sim_returns,
        mode='markers',
        marker=dict(
            color=sim_esg,
            colorscale='YlGn',
            showscale=True,
            colorbar=dict(title='ESG Score')
        )
    ))
    fig.update_layout(
        title='Simulated Risk-Return-ESG Frontier',
        xaxis_title='Annualized Volatility (Risk)',
        yaxis_title='Annualized Expected Return',
        template='plotly_dark'
    )
    return fig

def plot_all(backtest_results, esg, cfg, returns_df):
    """
    Generates and saves all plots.
    """
    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if cfg['plots']['engine'] == 'plotly':
        # Cumulative returns plot
        fig_cum_returns = plot_cumulative_returns(backtest_results, cfg)
        if cfg['plots'].get('show', False):
            fig_cum_returns.show()
        fig_cum_returns.write_html(figures_dir / "cumulative_returns.html")

        # Weights heatmap
        fig_weights = plot_weights_heatmap(backtest_results, cfg)
        if cfg['plots'].get('show', False):
            fig_weights.show()
        fig_weights.write_html(figures_dir / "weights_heatmap.html")

        # Risk-Return-ESG Frontier
        fig_frontier = plot_risk_return_esg_frontier(returns_df, esg, cfg)
        if cfg['plots'].get('show', False):
            fig_frontier.show()
        fig_frontier.write_html(figures_dir / "risk_return_esg_frontier.html")

        print("All plots generated and saved to outputs/figures/.")

    else:
        print(f"Plotting engine '{cfg['plots']['engine']}' not supported.")
