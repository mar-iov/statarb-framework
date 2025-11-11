"""
Beta estimation methods for pairs trading.

Supports static OLS, rolling window, and Kalman filter approaches.
"""

import logging
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def estimate_static_beta(S1: pd.Series, S2: pd.Series) -> float:
    """
    Estimate constant hedge ratio via OLS regression.

    Args:
        S1: Dependent series
        S2: Independent series

    Returns:
        Scalar beta coefficient
    """
    df = pd.DataFrame({"S1": S1, "S2": S2}).dropna()
    if df.empty:
        raise ValueError("Empty data after dropna")

    X = sm.add_constant(df["S2"])
    model = sm.OLS(df["S1"], X).fit()
    beta = float(model.params["S2"])

    logger.info(f"Static beta: {beta:.6f}")
    return beta


def estimate_rolling_beta(S1: pd.Series,
                          S2: pd.Series,
                          window: int,
                          min_periods: Optional[int] = None) -> pd.Series:
    """
    Time-varying hedge ratio using rolling OLS.

    Args:
        S1: Dependent series
        S2: Independent series
        window: Rolling window size
        min_periods: Minimum observations required (defaults to window)

    Returns:
        Series of beta estimates aligned to input index
    """
    if min_periods is None:
        min_periods = window

    df = pd.DataFrame({"S1": S1, "S2": S2})
    betas = []

    for i in range(len(df)):
        if i < min_periods - 1:
            betas.append(np.nan)
            continue

        start_idx = max(0, i - window + 1)
        window_data = df.iloc[start_idx:i+1]

        # Drop NaN values
        clean = window_data.dropna()

        if len(clean) < 2:
            betas.append(np.nan)
            continue

        try:
            X = sm.add_constant(clean["S2"])
            res = sm.OLS(clean["S1"], X).fit()
            betas.append(float(res.params["S2"]))
        except Exception:
            betas.append(np.nan)

    beta_series = pd.Series(betas, index=df.index, name="beta_rolling")
    return beta_series


def estimate_kalman_beta(S1: pd.Series,
                         S2: pd.Series,
                         delta: float = 1e-5,
                         R: float = 0.001,
                         init_beta: Optional[float] = None,
                         init_P: float = 1.0) -> pd.Series:
    """
    Adaptive hedge ratio via Kalman filter.

    State-space model:
        S1_t = beta_t * S2_t + epsilon_t  (observation)
        beta_{t+1} = beta_t + w_t         (random walk state)

    Args:
        S1: Dependent series
        S2: Independent series
        delta: Process noise parameter (smaller = smoother beta)
        R: Observation noise variance
        init_beta: Initial beta estimate (uses OLS if None)
        init_P: Initial state covariance

    Returns:
        Series of beta estimates
    """
    df = pd.DataFrame({"y": S1, "x": S2}).astype(float)
    n = len(df)
    betas = np.zeros(n)

    # Initialize beta
    if init_beta is None:
        clean = df.dropna()
        if len(clean) >= 2:
            X = sm.add_constant(clean["x"])
            res = sm.OLS(clean["y"], X).fit()
            beta = float(res.params["x"])
        else:
            beta = 0.0
    else:
        beta = float(init_beta)

    P = float(init_P)
    Q = (delta / (1.0 - delta)) if (0 < delta < 1) else delta

    for t, (yt, xt) in enumerate(zip(df["y"], df["x"])):
        if np.isnan(yt) or np.isnan(xt):
            betas[t] = np.nan
            continue

        # Predict step
        beta_pred = beta
        P_pred = P + Q

        # Update step
        S = (xt**2) * P_pred + R
        K = (P_pred * xt) / S if S > 0 else 0.0

        residual = yt - xt * beta_pred
        beta = beta_pred + K * residual
        P = (1 - K * xt) * P_pred

        betas[t] = beta

    return pd.Series(betas, index=df.index, name="beta_kalman")


def get_beta_series(S1: pd.Series,
                    S2: pd.Series,
                    beta: Optional[Union[float, pd.Series]] = None,
                    beta_method: str = "static",
                    beta_kwargs: Optional[Dict[str, Any]] = None) -> Union[float, pd.Series]:
    """
    Unified interface for beta estimation.

    Args:
        S1: Dependent series
        S2: Independent series
        beta: Pre-computed beta (scalar or Series). If provided, estimation is skipped.
        beta_method: Estimation method ("static", "rolling", "kalman")
        beta_kwargs: Additional parameters for estimation method, like "delta" for the Kalman method

    Returns:
        Beta as scalar or Series depending on method
    """
    if beta_kwargs is None:
        beta_kwargs = {}

    # Return pre-computed beta if provided
    if isinstance(beta, pd.Series):
        return beta.reindex(S1.index).astype(float)
    if beta is not None and np.isscalar(beta):
        return float(beta)

    # Estimate beta
    method = beta_method.lower()

    if method == "static":
        return estimate_static_beta(S1, S2)

    elif method == "rolling":
        window = beta_kwargs.get("window")
        if window is None:
            raise ValueError("Rolling beta requires 'window' in beta_kwargs")
        min_periods = beta_kwargs.get("min_periods", window)
        return estimate_rolling_beta(S1, S2, window=int(window), min_periods=int(min_periods))

    elif method == "kalman":
        delta = float(beta_kwargs.get("delta", 1e-5))
        R = float(beta_kwargs.get("R", 0.001))
        init_beta = beta_kwargs.get("init_beta")
        init_P = float(beta_kwargs.get("init_P", 1.0))
        return estimate_kalman_beta(S1, S2, delta=delta, R=R, init_beta=init_beta, init_P=init_P)

    else:
        raise ValueError(f"Unknown beta_method: {beta_method}")