"""
Performance analytics and trade analysis for pairs trading.

Includes trade-by-trade breakdown and portfolio-level metrics.
"""

import logging
from typing import Union, Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def analyze_trades(positions: pd.Series,
                            spread: pd.Series,
                            spread_zscore: pd.Series,
                            S1: pd.Series,
                            S2: pd.Series,
                            beta: Union[float, pd.Series],
                            total_costs: pd.Series,
                            returns_net: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Breakdown of individual trade performance with dynamic hedge ratio support.

    Args:
        positions: Position series (+1 long, -1 short, 0 flat)
        spread: Spread time series
        spread_zscore: Normalized spread
        S1: Asset 1 prices
        S2: Asset 2 prices
        beta: Hedge ratio (scalar or time series)
        total_costs: Transaction costs per bar
        returns_net: Optional net returns series for compound return calculation

    Returns:
        DataFrame with one row per trade including multiple return metrics
    """
    trades = []
    in_trade = False
    entry_index = None
    entry_beta = None
    entry_notional = None
    position_type = None

    idx = positions.index
    n = len(idx)

    if np.isscalar(beta):
        beta_series = pd.Series(float(beta), index=idx)
    else:
        beta_series = beta.reindex(idx).astype(float)

    total_costs = total_costs.reindex(idx).fillna(0.0)
    spread = spread.reindex(idx).astype(float)
    spread_zscore = spread_zscore.reindex(idx).astype(float)

    # Precompute returns_net if not provided
    if returns_net is not None:
        returns_net = returns_net.reindex(idx).fillna(0.0)

    for i in range(1, n):
        current_pos = positions.iloc[i]
        prev_pos = positions.iloc[i - 1]

        # Close existing trade on exit or position flip
        if in_trade and (current_pos == 0 or (current_pos != 0 and current_pos != position_type)):
            exit_index = i
            exit_spread = spread.iloc[exit_index]
            exit_beta = beta_series.iloc[exit_index]

            if not np.isfinite(exit_beta):
                exit_beta = beta_series.dropna().mean() if beta_series.dropna().size > 0 else 0.0

            beta_change = exit_beta - entry_beta

            # Skip if entry notional invalid
            if entry_notional is None or entry_notional == 0 or not np.isfinite(entry_notional):
                trades.append({
                    'entry_date': idx[entry_index],
                    'exit_date': idx[exit_index],
                    'type': 'LONG' if position_type > 0 else 'SHORT',
                    'entry_z': float(spread_zscore.iloc[entry_index]),
                    'exit_z': float(spread_zscore.iloc[exit_index]),
                    'entry_spread': float(spread.iloc[entry_index]),
                    'exit_spread': exit_spread,
                    'spread_change': exit_spread - spread.iloc[entry_index],
                    'spread_return_pct': np.nan,
                    'gross_return_pct': np.nan,
                    'trade_costs_pct': np.nan,
                    'net_return_pct': np.nan,
                    'compound_return': np.nan,
                    'annualized_return': np.nan,
                    'entry_beta': entry_beta,
                    'exit_beta': exit_beta,
                    'beta_change_during_trade': beta_change,
                    'holding_period_bars': exit_index - entry_index,
                    'duration': idx[exit_index] - idx[entry_index],
                    'alignment_ratio': np.nan,
                    'max_abs_z_in_trade': np.nan,
                    'mismatch': np.nan,
                    'entry_notional': entry_notional
                })
                in_trade = False
                continue

            # Calculate P&L metrics
            entry_spread = float(spread.iloc[entry_index])
            spread_change = exit_spread - entry_spread
            spread_return_pct = (spread_change / entry_notional) * 100.0

            # Directional return (positive = profitable)
            if position_type > 0:
                gross_return_pct = spread_return_pct
            else:
                gross_return_pct = -spread_return_pct

            # Costs over trade duration
            trade_costs_pct = float(total_costs.iloc[entry_index:exit_index + 1].sum() * 100.0)
            net_return_pct = gross_return_pct - trade_costs_pct

            # Compound return (if returns_net provided)
            if returns_net is not None:
                trade_returns = returns_net.iloc[entry_index:exit_index + 1]
                compound_return = float((1 + trade_returns).prod() - 1)
            else:
                compound_return = np.nan

            # Annualized return (assuming 3-min bars for crypto)
            holding_bars = exit_index - entry_index
            if holding_bars > 0:
                # Rough annualization: 365 * 24 * 60 / 3 = 175200 bars per year for 3-min bars
                bars_per_year = 365 * 24 * 20  # Conservative: 20 bars/hour average
                periods_ratio = bars_per_year / holding_bars
                annualized_return = ((1 + net_return_pct/100) ** periods_ratio - 1) * 100
            else:
                annualized_return = np.nan

            # Diagnostics
            span_spread = spread.iloc[entry_index:exit_index + 1]
            span_z = spread_zscore.iloc[entry_index:exit_index + 1]
            alignment_ratio = float((np.sign(span_spread) == np.sign(span_z)).mean()) if span_z.size > 0 else np.nan
            max_abs_z = float(span_z.abs().max()) if span_z.size > 0 else np.nan

            z_entry = float(span_z.iloc[0]) if span_z.size > 0 else np.nan
            z_exit = float(span_z.iloc[-1]) if span_z.size > 0 else np.nan
            mismatch = bool(np.sign(spread_change) != np.sign(z_exit - z_entry))

            trades.append({
                'entry_date': idx[entry_index],
                'exit_date': idx[exit_index],
                'type': 'LONG' if position_type > 0 else 'SHORT',
                'entry_z': z_entry,
                'exit_z': z_exit,
                'entry_spread': entry_spread,
                'exit_spread': exit_spread,
                'spread_change': spread_change,
                'spread_return_pct': spread_return_pct,
                'gross_return_pct': gross_return_pct,
                'trade_costs_pct': trade_costs_pct,
                'net_return_pct': net_return_pct,
                'compound_return': compound_return,
                'annualized_return': annualized_return,
                'entry_beta': float(entry_beta),
                'exit_beta': float(exit_beta),
                'beta_change_during_trade': float(beta_change),
                'entry_notional': float(entry_notional),
                'holding_period_bars': holding_bars,
                'duration': idx[exit_index] - idx[entry_index],
                'alignment_ratio': alignment_ratio,
                'max_abs_z_in_trade': max_abs_z,
                'mismatch': mismatch
            })

            in_trade = False

        # Open new trade on entry
        if current_pos != 0 and (not in_trade or current_pos != position_type):
            in_trade = True
            entry_index = i
            position_type = current_pos
            entry_beta = beta_series.iloc[i]

            if not np.isfinite(entry_beta):
                entry_beta = beta_series.dropna().mean() if beta_series.dropna().size > 0 else 0.0

            entry_notional = abs(S1.iloc[i]) + abs(entry_beta) * abs(S2.iloc[i])

    df = pd.DataFrame(trades)

    if not df.empty:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])

        numeric_cols = ['entry_spread', 'exit_spread', 'spread_change', 'spread_return_pct',
                        'gross_return_pct', 'trade_costs_pct', 'net_return_pct',
                        'compound_return', 'annualized_return',
                        'entry_beta', 'exit_beta', 'beta_change_during_trade',
                        'entry_notional', 'alignment_ratio', 'max_abs_z_in_trade']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def calculate_max_drawdown(cumulative: pd.Series) -> float:
    """Calculate maximum drawdown from cumulative return series."""
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calculate_composite_score(returns: pd.Series,
                              tf_min: float,
                              days_per_year: int = 365,
                              weights: list = None) -> float:
    """
    Unbounded weighted composite of Sharpe, Sortino, and Calmar ratios.
    """
    if weights is None:
        weights = [0.3, 0.4, 0.3]

    r = pd.Series(returns).dropna()
    if r.empty:
        return np.nan

    periods_per_day = (24 * 60) / float(tf_min)
    periods_per_year = days_per_year * periods_per_day

    # Performance stats
    cum = (1 + r).cumprod()
    max_dd = calculate_max_drawdown(cum)

    ann_return = (1 + r.mean()) ** periods_per_year - 1
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if not downside.empty else 0.0

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    # **Unbounded** composite score
    return weights[0] * sharpe + weights[1] * sortino + weights[2] * calmar



def calculate_performance_metrics(returns: Iterable[float],
                                  tf_min: float,
                                  days_per_year: int = 365,
                                  beta: Optional[Union[float, pd.Series]] = None,
                                  beta_method: Optional[str] = None) -> Dict:
    """
    Comprehensive performance metrics for strategy returns.

    Args:
        returns: Per-bar return series
        tf_min: Minutes per bar (for annualization)
        days_per_year: Trading days per year
        beta: Hedge ratio (for diagnostics)
        beta_method: Method used for beta estimation

    Returns:
        Dictionary of performance metrics
    """
    r = pd.Series(returns).dropna()

    if r.empty:
        base = {k: np.nan for k in [
            'sharpe', 'sortino', 'calmar', 'composite',
            'profit_factor', 'total_return', 'max_drawdown', 'volatility',
            'win_rate', 'avg_trade_return', 'number_of_trades'
        ]}
        base.update({'total_return': 0.0, 'max_drawdown': 0.0, 'number_of_trades': 0})

        if beta is not None:
            base['beta_method'] = beta_method
            if np.isscalar(beta):
                base['beta_stats'] = {'mean': float(beta), 'volatility': 0.0,
                                      'min': float(beta), 'max': float(beta)}
            else:
                beta_s = beta.dropna()
                base['beta_stats'] = {
                    'mean': float(beta_s.mean()) if len(beta_s) > 0 else np.nan,
                    'volatility': float(beta_s.std()) if len(beta_s) > 0 else np.nan,
                    'min': float(beta_s.min()) if len(beta_s) > 0 else np.nan,
                    'max': float(beta_s.max()) if len(beta_s) > 0 else np.nan
                }
        return base

    periods_per_day = (24 * 60) / float(tf_min)
    periods_per_year = days_per_year * periods_per_day

    cum = (1 + r).cumprod()
    total_return = float(cum.iloc[-1] - 1.0)
    max_dd = calculate_max_drawdown(cum)

    # Correct annualized return: geometric mean
    n_periods = len(r)
    ann_return = ((1 + total_return) ** (periods_per_year / n_periods) - 1) if n_periods > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if not downside.empty else 0.0

    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    sortino = ann_return / downside_vol if downside_vol > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(abs(r[r < 0].sum()))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf

    composite = calculate_composite_score(r, tf_min=tf_min, days_per_year=days_per_year,
                                          weights=[0.3, 0.4, 0.3])

    metrics = {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'composite': composite,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'volatility': ann_vol,
        'win_rate': float((r > 0).sum()) / len(r),
        'avg_trade_return': float(r.mean()),
        'number_of_trades': int(len(r))
    }

    if beta is not None:
        metrics['beta_method'] = beta_method
        if np.isscalar(beta):
            metrics['beta_stats'] = {
                'mean': float(beta),
                'volatility': 0.0,
                'min': float(beta),
                'max': float(beta)
            }
        else:
            beta_s = beta.dropna()
            metrics['beta_stats'] = {
                'mean': float(beta_s.mean()),
                'volatility': float(beta_s.std()),
                'min': float(beta_s.min()),
                'max': float(beta_s.max())
            }

    return metrics