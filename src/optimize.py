import cvxpy as cp
import numpy as np
import pandas as pd

def compute_covariance_matrix(returns, method, ewma_lambda):
    """
    Computes the covariance matrix of returns.
    """
    if method == "ewma":
        # The correct formula for span from the decay factor lambda is: span = 2 / (1 - lambda) - 1
        span = 2 / (1 - ewma_lambda) - 1
        return returns.ewm(span=span).cov()
    else: # Default to sample covariance
        return returns.cov()

def optimize_portfolio(r_hat, Sigma, esg_norm, cfg):
    """
    Solves the mean-variance optimization problem with an ESG term.

    Maximizes: w^T * r_hat - alpha * w^T * Sigma * w + beta * w^T * esg_norm
    """
    n = len(r_hat)
    w = cp.Variable(n)

    alpha = cfg["opt"]["alpha_risk_aversion"]
    beta = cfg["opt"]["beta_esg_pref"]
    cap = cfg["opt"]["weight_max"]

    objective = cp.Maximize(w @ r_hat - alpha * cp.quad_form(w, Sigma) + beta * (w @ esg_norm))

    constraints = [
        w >= 0,           # Long-only constraint
        w <= cap,         # Max weight per asset
        cp.sum(w) == 1    # Fully invested
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status != 'optimal':
        # Fallback to a simple equal-weight portfolio if optimization fails
        print("Optimization failed. Returning equal-weight portfolio.")
        return np.full(n, 1/n)

    return np.array(w.value)
