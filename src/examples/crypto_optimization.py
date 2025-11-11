"""
Example: Parameter optimization with constraints, early stopping, and parallelization.

Demonstrates all three Phase 1 improvements.
"""

import numpy as np
import pandas as pd
import time
from statarb import (
    auto_optimization,
    grid_search_optimization,
    random_search_optimization,
    setup_logging
)

setup_logging()

# ============================================================================
# DATA LOADING
# ============================================================================

# Change paths according to your system setup
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
start_date = '2025-05-30'
end_date = '2025-08-30'
tf_min = 3

start_ts = pd.to_datetime(start_date, utc=True)
end_ts = pd.to_datetime(end_date, utc=True)
combined_df = combined_df.loc[start_ts:end_ts]

print(f"Data loaded: {combined_df.shape[0]} bars from {combined_df.index.min()} to {combined_df.index.max()}")

# Extract close prices
S1 = combined_df['ADA_close']
S2 = combined_df['ARB_close']

def main():                        # Necessary for Windows multiprocessing
    # =============================================================================
    # EXAMPLE 1: Grid Search with Constraints and Early Stopping
    # =============================================================================

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Grid Search with Constraints")
    print("=" * 80)

    # Define parameter ranges
    param_ranges = {
        'entry_threshold_long': [-3.0, -2.0, -1.5, -1.0],
        'exit_threshold_long': [-0.5, 0.0, 0.5, 1, 1.5],
        'stop_loss_threshold_long': [-5.0, -4.0],
        'entry_threshold_short': [1.0, 1.5, 2.0, 3.0],
        'exit_threshold_short': [-0.5, 0.0, 0.5, 1, 1.5],
        'stop_loss_threshold_short': [4.0, 5.0],
        'z_window_long': [50, 100, 150],
        'cooldown_bars': [0, 10]
    }

    # Define logical constraints
    constraints = [
        # Long position logic: stop_loss < entry < exit
        {'type': 'order', 'params': ('stop_loss_threshold_long', '<', 'entry_threshold_long')},
        {'type': 'order', 'params': ('entry_threshold_long', '<', 'exit_threshold_long')},

        # Short position logic: exit < entry < stop_loss
        {'type': 'order', 'params': ('exit_threshold_short', '<', 'entry_threshold_short')},
        {'type': 'order', 'params': ('entry_threshold_short', '<', 'stop_loss_threshold_short')},

        # Z-window must be positive
        {'type': 'order', 'params': ('z_window_long', '>', 10)},
    ]

    start_time = time.perf_counter()

    # Run optimization with early stopping
    best_params, best_value, results_df = grid_search_optimization(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=tf_min,  # 1-minute bars
        beta_method='static',
        objective_metric='sharpe',
        constraints=constraints,
        early_stop_threshold=2.0,  # Stop if Sharpe > 2.0
        early_stop_patience=50,  # Or if no improvement for 50 iterations
        n_jobs=-1  # Use all CPUs
    )

    end_time = time.perf_counter()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Optimization completed in {execution_time_minutes:.2f} minutes")

    print(f"\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest objective metric: {best_value:.4f}")
    print(f"Valid combinations tested: {len(results_df)}")

    # =============================================================================
    # EXAMPLE 2: Random Search with Parallelization
    # =============================================================================

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Random Search with Parallelization")
    print("=" * 80)

    # Continuous parameter ranges (better for random search)
    param_ranges_continuous = {
        'entry_threshold_long': [-3.0, -0.5],
        'exit_threshold_long': [-0.5, 1.0],
        'stop_loss_threshold_long': [-6.0, -2.0],
        'entry_threshold_short': [0.5, 3.0],
        'exit_threshold_short': [-1.0, 0.5],
        'stop_loss_threshold_short': [2.0, 6.0],
        'z_window_long': [30, 200],
        'z_window_short': [30, 200],
        'cooldown_bars': [0, 10]
    }

    start_time = time.perf_counter()

    # Same constraints
    best_params_rand, best_value_rand, results_rand = random_search_optimization(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges_continuous,
        tf_min=3,
        beta_method='kalman',  # Try Kalman beta
        beta_kwargs={'delta': 1e-5, 'R': 0.001},
        objective_metric='composite',
        constraints=constraints,
        n_iter=3000,
        early_stop_threshold=0.8,  # Stop if composite score > 0.8
        early_stop_patience=30,
        n_jobs=-1
    )

    end_time = time.perf_counter()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Optimization completed in {execution_time_minutes:.2f} minutes")

    print(f"\nBest Parameters:")
    for k, v in best_params_rand.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nBest objective metric: {best_value_rand:.4f}")
    print(f"Valid combinations tested: {len(results_rand)}")

    # =============================================================================
    # EXAMPLE 3: Compare Methods
    # =============================================================================

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Method Comparison")
    print("=" * 80)

    # Simplified ranges for quick comparison
    simple_ranges = {
        'entry_threshold_long': [-2.5, -1.5, -1.0],
        'exit_threshold_long': [0.0, 0.5],
        'entry_threshold_short': [1.0, 1.5, 2.5],
        'exit_threshold_short': [0.0, -0.5],
        'z_window_long': [75, 100]
    }

    simple_constraints = [
        {'type': 'order', 'params': ('entry_threshold_long', '<', 'exit_threshold_long')},
        {'type': 'order', 'params': ('entry_threshold_short', '>', 'exit_threshold_short')},
    ]

    methods = ['grid_search', 'random_search']
    results_comparison = {}

    for method in methods:
        print(f"\nTesting {method}...")

        best_p, best_v, df = auto_optimization(S1=S1, S2=S2, param_ranges=simple_ranges, tf_min=3, beta_method='static',
                                               objective_metric='sharpe', optimization_method=method,
                                               constraints=simple_constraints,
                                               n_iter=50 if method == 'random_search' else None, n_jobs=-1)

        results_comparison[method] = {
            'best_value': best_v,
            'best_params': best_p,
            'n_tested': len(df)
        }

    print("\n" + "-" * 80)
    print("Results Summary:")
    print("-" * 80)
    for method, result in results_comparison.items():
        print(f"\n{method.upper()}:")
        print(f"  Best Sharpe: {result['best_value']:.4f}")
        print(f"  Combinations tested: {result['n_tested']}")
        print(f"  Best params: {result['best_params']}")

# Necessary for Windows multiprocessing
if __name__ == '__main__':
    main()