"""
Visualization utilities for trade analysis and diagnostics.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Union


def plot_strategy_engine(
    spread: pd.Series,
    spread_zscore: pd.Series,
    trade_df: pd.DataFrame,
    params: dict,
    beta: Optional[Union[pd.Series, float]] = None,
    plot_beta: bool = False
):
    """
    Visualize spread, z-score, and trade entries/exits, optionally with beta overlay.

    Args:
        spread: Spread time series
        spread_zscore: Normalized spread
        trade_df: DataFrame from analyze_trades_detailed()
        params: Trading parameters (for threshold lines)
        beta: Optional hedge ratio overlay (Series or scalar)
        plot_beta: Whether to overlay beta on the main chart
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Primary axis: Spread
    ax.plot(spread, color='blue', alpha=0.7, linewidth=1.3, label='Spread')
    ax.set_ylabel('Spread', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Secondary axis: Z-Score
    ax2 = ax.twinx()
    ax2.plot(spread_zscore, color='purple', alpha=0.7, linewidth=1.2, label='Z-Score')
    ax2.set_ylabel('Z-Score', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Entry/Exit thresholds on Z-Score axis
    ax2.axhline(params.get('entry_threshold_short', 2.0), color='r',
                linestyle='--', alpha=0.8, label='Entry Short')
    ax2.axhline(params.get('entry_threshold_long', -2.0), color='g',
                linestyle='--', alpha=0.8, label='Entry Long')
    ax2.axhline(params.get('exit_threshold_short', 0.0), color='orange',
                linestyle=':', alpha=0.8, label='Exit Short')
    ax2.axhline(params.get('exit_threshold_long', 0.0), color='blue',
                linestyle=':', alpha=0.8, label='Exit Long')

    # Optional third axis for beta
    if beta is not None and plot_beta:
        if np.isscalar(beta):
            beta_s = pd.Series(float(beta), index=spread.index)
        else:
            beta_s = beta.reindex(spread.index)

        # Create a third y-axis on the right, offset slightly
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(beta_s, color='black', linewidth=1.5, alpha=0.8, label='Beta')
        ax3.set_ylabel('Beta', color='black')
        ax3.tick_params(axis='y', labelcolor='black')

        # Optional ±1σ shading
        beta_mean = beta_s.mean()
        beta_std = beta_s.std()
        ax3.fill_between(beta_s.index,
                         beta_mean - beta_std,
                         beta_mean + beta_std,
                         color='gray', alpha=0.1)

    # Plot trades
    for _, trade in trade_df.iterrows():
        is_mismatch = trade.get('mismatch', False)
        entry_marker = '^' if trade['type'] == 'LONG' else 'v'

        entry_color = (
            'gray' if is_mismatch
            else ('lime' if trade['type'] == 'LONG' else 'red')
        )
        exit_color = 'cyan' if trade['type'] == 'LONG' else 'yellow'

        ax2.scatter(trade['entry_date'], trade['entry_z'],
                    color=entry_color, marker=entry_marker, s=100,
                    edgecolor='black', zorder=5)
        ax2.scatter(trade['exit_date'], trade['exit_z'],
                    color=exit_color, marker='o', s=80,
                    edgecolor='black', zorder=5)

    # Labels and legend
    ax.set_title("Pairs Trading: Spread, Z-Score, and Beta with Trades", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Combine legends cleanly
    lines, labels = [], []
    for a in [ax, ax2] + ([ax3] if beta is not None and plot_beta else []):
        lns, lbls = a.get_legend_handles_labels()
        lines += lns
        labels += lbls
    ax.legend(lines, labels, loc='upper right', framealpha=0.8, fontsize=9)

    plt.tight_layout()
    plt.show()



def plot_returns_analysis(returns: pd.Series,
                          cumulative_returns: pd.Series,
                          rolling_sharpe_bars: int,
                          figsize=(14, 8)):
    """
    Plot return distribution and cumulative performance.

    Args:
        returns: Per-bar returns
        cumulative_returns: Cumulative return series
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Cumulative returns
    axes[0, 0].plot(cumulative_returns, linewidth=1.5)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True, alpha=0.3)

    # Return distribution
    axes[0, 1].hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Return Distribution')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Rolling Sharpe (30-day)
    rolling_mean = returns.rolling(rolling_sharpe_bars).mean()
    rolling_std = returns.rolling(rolling_sharpe_bars).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized
    axes[1, 0].plot(rolling_sharpe, linewidth=1.5)
    axes[1, 0].set_title(f'Rolling {rolling_sharpe_bars}-Bar Sharpe Ratio')
    axes[1, 0].set_ylabel('Sharpe')
    axes[1, 0].grid(True, alpha=0.3)

    # Drawdown
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    axes[1, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.5, color='red')
    axes[1, 1].set_title('Drawdown (%)')
    axes[1, 1].set_ylabel('Drawdown %')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()