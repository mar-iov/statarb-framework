"""
Basic example: Pairs trading backtest with static beta.

Demonstrates minimal setup for running a backtest.
"""

import numpy as np
import pandas as pd
from statarb import (
    run_backtest,
    analyze_trades,
    calculate_performance_metrics,
    plot_strategy_engine,
    setup_logging
)

# Setup logging
setup_logging()

# Generate synthetic correlated price data for demonstration
np.random.seed(42)
n = 10000
t = pd.date_range('2023-01-01', periods=n, freq='1min')

# Create cointegrated pair
noise1 = np.random.randn(n) * 0.1
noise2 = np.random.randn(n) * 0.1
common_trend = np.cumsum(np.random.randn(n) * 0.05)

S1 = pd.Series(100 + common_trend + noise1, index=t, name='BTC')
S2 = pd.Series(50 + 0.5 * common_trend + noise2, index=t, name='ETH')

# Strategy parameters
params = {
    'entry_threshold_long': -2.0,
    'entry_threshold_short': 2.0,
    'exit_threshold_long': -0.5,
    'exit_threshold_short': 0.5,
    'stop_loss_threshold_long': -3.0,
    'stop_loss_threshold_short': 3.0,
    'z_window_long': 100,
    'cooldown_bars': 5
}

cost_params = {
    'commission_rate': 0.001,
    'slippage_bps': 5
}

# Run backtest
print("Running backtest...")
results = run_backtest(
    S1, S2,
    beta_method='static',
    params=params,
    cost_params=cost_params
)

positions, returns, spread, zscore, stops, changes, costs, betas, metadata = results

# Analyze trades
print("\nAnalyzing trades...")
trade_df = analyze_trades(positions, spread, zscore, S1, S2, betas, costs)

print(f"\nTotal trades: {len(trade_df)}")
if not trade_df.empty:
    print(f"Win rate: {(trade_df['net_return_pct'] > 0).mean():.2%}")
    print(f"Average return per trade: {trade_df['net_return_pct'].mean():.3f}%")
    print(f"Best trade: {trade_df['net_return_pct'].max():.3f}%")
    print(f"Worst trade: {trade_df['net_return_pct'].min():.3f}%")

# Performance metrics
print("\nCalculating performance metrics...")
metrics = calculate_performance_metrics(
    returns,
    tf_min=1,  # 1-minute bars
    beta=betas,
    beta_method='static'
)

print(f"\nPerformance Summary:")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"  Total Return: {metrics['total_return']*100:.2f}%")
print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"  Volatility: {metrics['volatility']*100:.2f}%")