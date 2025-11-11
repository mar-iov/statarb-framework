"""
Complete crypto pairs trading backtest example.

Demonstrates data loading, backtesting, and comprehensive analysis.
"""

import numpy as np
import pandas as pd
from statarb import (
    run_backtest,
    analyze_trades,
    calculate_performance_metrics,
    plot_strategy_engine,
    plot_returns_analysis,
    setup_logging
)

setup_logging()

# ============================================================================
# DATA LOADING
# ============================================================================

ADA_df = pd.read_csv(
    r'C:\Users\Utente\PycharmProjects\DynamicBetaStatarb\src\data\binance_ADAUSDT_FULL_3m.csv'
)
ARB_df = pd.read_csv(
    r'C:\Users\Utente\PycharmProjects\DynamicBetaStatarb\src\data\binance_ARBUSDT_FULL_3m.csv'
)

# Convert datetime with UTC
ADA_df['datetime_ny'] = pd.to_datetime(ADA_df['datetime_ny'], utc=True)
ARB_df['datetime_ny'] = pd.to_datetime(ARB_df['datetime_ny'], utc=True)

ADA_df.set_index('datetime_ny', inplace=True)
ARB_df.set_index('datetime_ny', inplace=True)

# Prefix columns to avoid conflicts
ADA_df = ADA_df.add_prefix('ADA_')
ARB_df = ARB_df.add_prefix('ARB_')

# Combine and fill gaps
combined_df = pd.concat([ADA_df, ARB_df], axis=1, join='outer')
combined_df.ffill(inplace=True)
combined_df.bfill(inplace=True)

# Filter date range
start_date = '2025-01-15'
end_date = '2025-08-15'
tf_min = 3

start_ts = pd.to_datetime(start_date, utc=True)
end_ts = pd.to_datetime(end_date, utc=True)
combined_df = combined_df.loc[start_ts:end_ts]

print(f"Data loaded: {combined_df.shape[0]} bars from {combined_df.index.min()} to {combined_df.index.max()}")

# Extract close prices
S1 = combined_df['ADA_close']
S2 = combined_df['ARB_close']

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

params = {
    'entry_threshold_long': -1.7,
    'exit_threshold_long': 0.6,
    'stop_loss_threshold_long': -3,
    'z_window_long': 103,
    'entry_threshold_short': 1.43,
    'exit_threshold_short': -0.12,
    'stop_loss_threshold_short': 3,
    'z_window_short': 103,
    'cooldown_bars': 0,
}

'''
NOTE:
Regular retail per-leg costs and latency (i.e. 5 bps commission + 5 bps slippage) will
turn many profitable parametrizations into losing ones. Brokers and exchanges offer alternative
cost structures that are more appropriate.
In-development filtering system will allow for lower frequency and higher expected returns per-trade,
downing cost impact in standard conditions.
'''

cost_params = {                 # Regular retail per-leg costs and latency (i.e. 5 bps commission + 5 bps slippage) will
    'commission_rate': 0.0000,  # turn many profitable parametrizations into losing ones. Brokers and exchanges offer alternative
    'slippage_bps': 0           # cost structures that are more appropriate. In development filtering system will allow for
}                               # profitability

# ============================================================================
# RUN BACKTEST
# ============================================================================

print("\nRunning backtest with Kalman beta...")
results = run_backtest(
    S1=S1,
    S2=S2,
    beta_method="kalman",
    beta_kwargs={"delta": 1e-5, "R": 0.001},
    params=params,
    cost_params=cost_params
)

positions, returns_net, spread, spread_zscore, stop_loss_events, pos_changes, total_costs, betas, metadata = results

print(f"Costs applied: {cost_params['commission_rate']*100:.3f}% commission + {cost_params['slippage_bps']} bps slippage per leg")

# ============================================================================
# PERFORMANCE METRICS (using the function)
# ============================================================================

print("\n" + "="*80)
print("STRATEGY PERFORMANCE METRICS")
print("="*80)

metrics = calculate_performance_metrics(
    returns=returns_net,
    tf_min=tf_min,
    days_per_year=365,  # Crypto markets are open every day.
    beta=betas,
    beta_method="kalman"
)

# Print core metrics
print(f"\nReturns:")
print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
print(f"  Avg Bar Return:      {metrics['avg_trade_return']*100:>8.4f}%")

print(f"\nRisk-Adjusted:")
print(f"  Sharpe Ratio:        {metrics['sharpe']:>8.2f}")
print(f"  Sortino Ratio:       {metrics['sortino']:>8.2f}")
print(f"  Calmar Ratio:        {metrics['calmar']:>8.2f}")
print(f"  Composite Score:     {metrics['composite']:>8.2f}")

print(f"\nRisk:")
print(f"  Annualized Vol:      {metrics['volatility']*100:>8.2f}%")
print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")

print(f"\nTrade Statistics:")
print(f"  Win Rate:            {metrics['win_rate']*100:>8.1f}%")
print(f"  Profit Factor:       {metrics['profit_factor']:>8.2f}")
print(f"  Number of Bars:      {metrics['number_of_trades']:>8d}")

if 'beta_stats' in metrics:
    print(f"\nBeta Statistics:")
    print(f"  Mean Beta:           {metrics['beta_stats']['mean']:>8.4f}")
    print(f"  Beta Volatility:     {metrics['beta_stats']['volatility']:>8.4f}")
    print(f"  Beta Range:          [{metrics['beta_stats']['min']:.4f}, {metrics['beta_stats']['max']:.4f}]")

# ============================================================================
# TRADE-BY-TRADE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TRADE-BY-TRADE BREAKDOWN")
print("="*80)

trade_df = analyze_trades(positions=positions, spread=spread, spread_zscore=spread_zscore, S1=S1, S2=S2, beta=betas,
                          total_costs=total_costs, returns_net=returns_net)

if not trade_df.empty:
    # Check for open position
    if positions.iloc[-1] != 0:
        print(f"\n  WARNING: Final position is {positions.iloc[-1]:.0f} (not flat). Unrealized P&L not analyzed.")

    print(f"\nTotal Trades: {len(trade_df)}")

    # VERIFICATION: Compare compound returns
    trade_compound_product = (1 + trade_df['compound_return']).prod() - 1
    strategy_total = metrics['total_return']

    print(f"\n🔍 Return Reconciliation:")
    print(f"  Strategy Total (bar-level):     {strategy_total*100:>8.2f}%")
    print(f"  Trade Product (compound):       {trade_compound_product*100:>8.2f}%")
    print(f"  Difference:                     {(strategy_total - trade_compound_product)*100:>8.2f}%")

    if abs(strategy_total - trade_compound_product) < 0.001:
        print("   Returns match within tolerance")
    else:
        print("   Returns differ - check for open positions or calculation issues")

    # Win/Loss breakdown
    winners = trade_df[trade_df['net_return_pct'] > 0]
    losers = trade_df[trade_df['net_return_pct'] <= 0]

    print(f"\nWin/Loss Breakdown:")
    print(f"  Winning Trades:      {len(winners):>4d} ({len(winners)/len(trade_df)*100:>5.1f}%)")
    print(f"  Losing Trades:       {len(losers):>4d} ({len(losers)/len(trade_df)*100:>5.1f}%)")

    if not winners.empty:
        print(f"  Avg Winner:          {winners['net_return_pct'].mean():>8.3f}%")
        print(f"  Largest Winner:      {winners['net_return_pct'].max():>8.3f}%")

    if not losers.empty:
        print(f"  Avg Loser:           {losers['net_return_pct'].mean():>8.3f}%")
        print(f"  Largest Loser:       {losers['net_return_pct'].min():>8.3f}%")

    # Holding period stats
    print(f"\nHolding Period:")
    print(f"  Avg Duration:        {trade_df['holding_period_bars'].mean():>8.1f} bars")
    print(f"  Median Duration:     {trade_df['holding_period_bars'].median():>8.0f} bars")
    print(f"  Range:               [{trade_df['holding_period_bars'].min():.0f}, {trade_df['holding_period_bars'].max():.0f}] bars")

    # Beta dynamics during trades
    print(f"\nBeta Change During Trades:")
    print(f"  Mean Abs Change:     {trade_df['beta_change_during_trade'].abs().mean():>8.4f}")
    print(f"  Max Abs Change:      {trade_df['beta_change_during_trade'].abs().max():>8.4f}")

    # Signal alignment diagnostics
    print(f"\nSignal Quality:")
    print(f"  Avg Alignment Ratio: {trade_df['alignment_ratio'].mean():>8.2f}")
    print(f"  Mismatched Trades:   {trade_df['mismatch'].sum():>4d} ({trade_df['mismatch'].sum()/len(trade_df)*100:>5.1f}%)")

    # Problematic trades
    big_losers = trade_df[trade_df['net_return_pct'] < -0.5]
    if not big_losers.empty:
        print(f"\n⚠️  Trades with >0.5% loss: {len(big_losers)}")
        print("\nWorst 5 Trades:")
        worst = big_losers.nsmallest(5, 'net_return_pct')[
            ['entry_date', 'exit_date', 'type', 'net_return_pct',
             'holding_period_bars', 'entry_z', 'exit_z', 'mismatch']
        ]
        print(worst.to_string(index=False))

    # Type breakdown
    print(f"\nTrade Type Distribution:")
    type_counts = trade_df['type'].value_counts()
    for trade_type, count in type_counts.items():
        avg_ret = trade_df[trade_df['type'] == trade_type]['net_return_pct'].mean()
        print(f"  {trade_type:5s}:              {count:>4d} trades, avg return: {avg_ret:>7.3f}%")

else:
    print("\nNo trades executed.")

# ============================================================================
# TRANSACTION COST ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TRANSACTION COST IMPACT")
print("="*80)

total_cost_paid = total_costs.sum()
total_legs = pos_changes.sum()

print(f"\nTotal Cost Paid:       {total_cost_paid*100:.3f}% of notional")
print(f"Total Trade Legs:      {int(total_legs)}")
print(f"Avg Cost Per Leg:      {(total_cost_paid/total_legs)*100:.4f}% of notional" if total_legs > 0 else "N/A")

# Stop loss analysis
stop_count = stop_loss_events.sum()
if stop_count > 0:
    print(f"\n⚠️  Stop Loss Triggered:   {stop_count} times")
    print(f"   Stop Loss Rate:        {(stop_count/len(trade_df))*100:.1f}% of trades" if not trade_df.empty else "N/A")

# ======================
# BENCHMARK COMPARISON
# ======================

print("\n" + "="*80)
print("BENCHMARK COMPARISON")
print("="*80)

# Buy-and-hold S1
bh_returns_S1 = S1.pct_change().fillna(0)
bh_total_S1 = (1 + bh_returns_S1).prod() - 1

# Buy-and-hold S2
bh_returns_S2 = S2.pct_change().fillna(0)
bh_total_S2 = (1 + bh_returns_S2).prod() - 1

strategy_total = metrics['total_return']

print(f"\nHold ADA (S1):         {bh_total_S1*100:>8.2f}%")
print(f"Hold ARB (S2):         {bh_total_S2*100:>8.2f}%")
print(f"Pairs Strategy:        {strategy_total*100:>8.2f}%")
print(f"\nOutperformance vs S1:  {(strategy_total - bh_total_S1)*100:>8.2f}%")
print(f"Outperformance vs S2:  {(strategy_total - bh_total_S2)*100:>8.2f}%")

# ===============
# VISUALIZATIONS
# ===============

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

# Plot 1: Trade visualization
'''
Gray triangles ("mismatches") indicate trades which satisfied the exit condition, but still yielded a loss. This is due to the
zscore being underreactive/overreactive to that underlying spread movement. These instances inspire the addition of new filters
(volume, volatility, ...) for entry confirmation.
P.S.: zoom-in.
'''

if not trade_df.empty:
    plot_strategy_engine(spread=spread, spread_zscore=spread_zscore, trade_df=trade_df, params=params, beta=betas, plot_beta=False)

# Plot 2: Returns analysis
cumulative_returns = (1 + returns_net).cumprod()
plot_returns_analysis(
    returns=returns_net,
    cumulative_returns=cumulative_returns,
    rolling_sharpe_bars=3360 #weekly
)

print("\nBacktest complete!")