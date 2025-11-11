"""
Example: Walk-Forward Analysis with dynamic beta selection.

Demonstrates comprehensive WFA validation for production deployment.
"""

import numpy as np
import pandas as pd
from statarb import (
    auto_wfa,
    print_wfa_summary,
    setup_logging
)

setup_logging(verbose_modules=False)

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
start_date = '2025-01-15'
end_date = '2025-08-30'
tf_min = 3

start_ts = pd.to_datetime(start_date, utc=True)
end_ts = pd.to_datetime(end_date, utc=True)
combined_df = combined_df.loc[start_ts:end_ts]

print(f"Data loaded: {combined_df.shape[0]} bars from {combined_df.index.min()} to {combined_df.index.max()}")

# Extract close prices
S1 = combined_df['ADA_close']
S2 = combined_df['ARB_close']


def main():

    param_ranges = {
        'entry_threshold_long': [-2.5, -2.0, -1.5, -1.0],
        'exit_threshold_long': [-0.5, 0.0, 0.5, 1, 1.5, 2],
        'entry_threshold_short': [1.0, 1.5, 2.0, 2.5],
        'exit_threshold_short': [-2, -1.5, -1, -0.5, 0.0, 0.5],
        'z_window_long': [50, 75, 100, 125, 150],
        'cooldown_bars': [0]
    }

    cost_params = {
        'commission_rate': 0.0000,
        'slippage_bps': 0
    }

    # ========================
    # EXAMPLE 1: Rolling WFA
    # ========================

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Rolling WFA with Static Beta")
    print("=" * 80)

    results, aggregated, full_returns = auto_wfa(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=3,
        train_days=15,
        test_days=7,
        cost_params=cost_params,
        objective_metric='composite',
        wfa_method='rolling',
        optimization_method='grid_search',
        beta_method='kalman',
        n_jobs=16 # Set according to user's CPU. -1 -> maximum computation speed (at the expense of any other process)
    )

    print_wfa_summary(aggregated, objective_metric='composite')

    # =========================
    # EXAMPLE 2: Expanding WFA
    # =========================

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Rolling WFA with Dynamic Beta Selection")
    print("=" * 80)

    results, aggregated, full_returns = auto_wfa(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=3,
        train_days=15,
        test_days=7,
        cost_params=cost_params,
        objective_metric='composite',
        wfa_method='expanding',
        optimization_method='bayesian',
        beta_method='kalman',
        beta_kwargs={'delta': 1e-5, 'R': 0.001},
        n_jobs=16
    )

    print_wfa_summary(aggregated, objective_metric='composite')

    # =========================================================
    # EXAMPLE 3: CPCV (Combinatorial Purged Cross-Validation)
    # =========================================================

    print("\n" + "=" * 80)
    print("EXAMPLE 3: CPCV")
    print("=" * 80)

    results, aggregated, full_returns = auto_wfa(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=3,
        train_days=None,  # Not used in CPCV
        test_days=None,  # Not used in CPCV
        cost_params=cost_params,
        objective_metric='composite',
        wfa_method='cpcv',
        optimization_method='bayesian',
        beta_method='kalman',
        beta_kwargs={'delta': 1e-5, 'R': 0.001},
        n_splits=7,
        embargo_pct=0.02
    )

    print_wfa_summary(aggregated, objective_metric='composite')

    # ===============================================
    # EXAMPLE 4: Anchored WFA (for regime stability)
    # ===============================================

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Anchored WFA")
    print("=" * 80)

    results, aggregated, full_returns = auto_wfa(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=3,
        train_days=60,  # Initial anchor period
        test_days=15,
        cost_params=cost_params,
        objective_metric='composite',
        wfa_method='anchored',
        optimization_method='bayesian',
        beta_method='kalman',
        beta_kwargs={'delta': 1e-5, 'R': 0.001, 'window': 100},
        n_jobs=16
    )

    print_wfa_summary(aggregated, objective_metric='composite')


if __name__ == '__main__':
    main()