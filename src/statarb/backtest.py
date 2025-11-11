"""
Core backtesting engine for pairs trading strategies.

Supports dynamic hedge ratios and flexible entry/exit rules.
"""

import logging
from typing import Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd

from .beta_estimation import get_beta_series
from .utils import calculate_zscore

logger = logging.getLogger(__name__)


def run_backtest(S1: pd.Series,
                 S2: pd.Series,
                 beta: Optional[Union[float, pd.Series]] = None,
                 beta_method: str = "static",
                 beta_kwargs: Optional[Dict] = None,
                 params: Optional[Dict] = None,
                 cost_params: Optional[Dict] = None) -> Tuple:
    """
    Execute pairs trading backtest with transaction costs.

    Args:
        S1: Price series for asset 1
        S2: Price series for asset 2
        beta: Pre-computed hedge ratio (optional)
        beta_method: Method for beta estimation
        beta_kwargs: Parameters for beta estimation
        params: Trading parameters (entry/exit thresholds, z-score windows, etc.)
        cost_params: Transaction cost parameters (commission, slippage)

    Returns:
        Tuple of (positions, returns_net, spread, spread_zscore, stop_loss_events,
                  pos_changes, total_costs, betas, metadata)
    """
    if params is None:
        params = {}
    if cost_params is None:
        cost_params = {}
    if beta_kwargs is None:
        beta_kwargs = {}

    # Estimate hedge ratio
    raw_betas = get_beta_series(S1, S2, beta=beta, beta_method=beta_method, beta_kwargs=beta_kwargs)

    if np.isscalar(raw_betas):
        betas = pd.Series(float(raw_betas), index=S1.index, name="beta_static")
    else:
        betas = raw_betas.reindex(S1.index).astype(float)

    # Construct spread using time-varying hedge ratio
    spread = S1 - betas * S2

    # Extract trading parameters
    entry_long = params.get('entry_threshold_long', -2.0)
    entry_short = params.get('entry_threshold_short', 2.0)
    exit_long = params.get('exit_threshold_long', 0.0)
    exit_short = params.get('exit_threshold_short', 0.0)
    stop_loss_long = params.get('stop_loss_threshold_long', -3.0)
    stop_loss_short = params.get('stop_loss_threshold_short', 3.0)
    z_window_long = int(params.get('z_window_long', 1000))
    z_window_short = int(params.get('z_window_short', z_window_long))
    cooldown_bars = int(params.get('cooldown_bars', 5))

    commission_rate = float(cost_params.get('commission_rate', 0.001))
    slippage_bps = float(cost_params.get('slippage_bps', 5))
    slippage_rate = slippage_bps / 10000.0

    # Calculate z-scores
    if z_window_long == z_window_short or z_window_short == None:
        spread_zscore = calculate_zscore(spread, window=z_window_long)
        z_long = z_short = spread_zscore
    else:
        z_long = calculate_zscore(spread, window=z_window_long)
        z_short = calculate_zscore(spread, window=z_window_short)
        spread_zscore = z_long

    # Initialize state
    idx = spread.index
    n = len(idx)
    positions = pd.Series(0, index=idx, dtype=float)
    stop_loss_events = pd.Series(False, index=idx, dtype=bool)
    in_cooldown = False
    bars_since_stop = 0

    # Trading logic
    for i in range(1, n):
        prev_pos = positions.iloc[i - 1]

        # Handle cooldown after stop loss
        if in_cooldown:
            bars_since_stop += 1
            if bars_since_stop >= cooldown_bars:
                in_cooldown = False
                bars_since_stop = 0

        z_long_val = z_long.iloc[i]
        z_short_val = z_short.iloc[i]

        if not np.isfinite(z_long_val) and not np.isfinite(z_short_val):
            positions.iloc[i] = prev_pos
            continue

        # Select appropriate z-score based on position
        if prev_pos != 0:
            current_z = z_long_val if prev_pos >= 0 else z_short_val
        else:
            current_z = z_long_val if np.isfinite(z_long_val) else z_short_val

        # Check stop loss conditions
        stop_long = (prev_pos > 0) and np.isfinite(current_z) and current_z < stop_loss_long
        stop_short = (prev_pos < 0) and np.isfinite(current_z) and current_z > stop_loss_short

        # Check exit conditions
        exit_long_cond = (prev_pos > 0) and np.isfinite(current_z) and current_z >= exit_long
        exit_short_cond = (prev_pos < 0) and np.isfinite(current_z) and current_z <= exit_short

        # Check entry conditions
        if z_window_long != z_window_short:
            enter_long = (not in_cooldown and prev_pos == 0 and
                          np.isfinite(z_long_val) and z_long_val < entry_long)
            enter_short = (not in_cooldown and prev_pos == 0 and
                           np.isfinite(z_short_val) and z_short_val > entry_short)
        else:
            enter_long = (not in_cooldown and prev_pos == 0 and
                          np.isfinite(current_z) and current_z < entry_long)
            enter_short = (not in_cooldown and prev_pos == 0 and
                           np.isfinite(current_z) and current_z > entry_short)

        # Execute trading decisions
        if stop_long or stop_short:
            positions.iloc[i] = 0
            stop_loss_events.iloc[i] = True
            in_cooldown = True
            bars_since_stop = 0
        elif exit_long_cond or exit_short_cond:
            positions.iloc[i] = 0
        elif enter_long:
            positions.iloc[i] = 1.0
        elif enter_short:
            positions.iloc[i] = -1.0
        else:
            positions.iloc[i] = prev_pos

    positions = positions.astype(float)

    # Calculate entry notional for normalization
    entry_notional = pd.Series(0.0, index=idx, dtype=float)
    pos_changes = positions.diff().fillna(0).abs()

    for i in range(1, n):
        if pos_changes.iloc[i] > 0 and positions.iloc[i] != 0:
            beta_now = abs(betas.iloc[i])
            entry_notional.iloc[i] = abs(S1.iloc[i]) + beta_now * abs(S2.iloc[i])

    entry_notional = entry_notional.where(positions != 0).ffill().fillna(0.0)

    # Calculate returns
    spread_diff = spread.diff().fillna(0.0)
    prev_positions = positions.shift(1).fillna(0.0)
    prev_entry_notional = entry_notional.shift(1).fillna(0.0)

    returns = pd.Series(0.0, index=idx, dtype=float)
    mask = (prev_positions != 0) & (prev_entry_notional > 0)
    returns.loc[mask] = (prev_positions.loc[mask] * spread_diff.loc[mask]) / prev_entry_notional.loc[mask]
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Calculate transaction costs
    cost_fraction = (commission_rate + slippage_rate) * pos_changes
    total_costs = pd.Series(0.0, index=idx, dtype=float)
    trade_mask = (pos_changes > 0)
    total_costs.loc[trade_mask] = cost_fraction.loc[trade_mask]

    # Net returns after costs
    returns_net = returns.copy()
    returns_net.loc[trade_mask] -= total_costs.loc[trade_mask]
    returns_net = returns_net.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    metadata = {
        "beta_method": beta_method,
        "beta_kwargs": beta_kwargs,
        "trade_params": params,
        "cost_params": cost_params,
    }

    return (positions, returns_net, spread, spread_zscore, stop_loss_events,
            pos_changes, total_costs, betas, metadata)