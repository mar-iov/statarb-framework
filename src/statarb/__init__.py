"""
Statistical Arbitrage Framework for Crypto Pairs Trading

A modular framework for backtesting pairs trading strategies with dynamic hedge ratios.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .beta_estimation import (
    estimate_static_beta,
    estimate_rolling_beta,
    estimate_kalman_beta,
    get_beta_series
)

from .backtest import run_backtest

from .analytics import (
    analyze_trades,
    calculate_performance_metrics,
    calculate_max_drawdown,
    calculate_composite_score
)

from .plotting import plot_strategy_engine, plot_returns_analysis

from .utils import calculate_zscore, setup_logging

from .optimization import (
    auto_optimization,
    grid_search_optimization,
    random_search_optimization,
    bayesian_optimization,
    genetic_optimization,
    extract_optimizer_status
)

from .walk_forward import (
    auto_wfa,
    WFAResult,
    WFAAggregatedResults,
    print_wfa_summary
)

__all__ = [
    # Beta estimation
    'estimate_static_beta',
    'estimate_rolling_beta',
    'estimate_kalman_beta',
    'get_beta_series',
    # Backtesting
    'run_backtest',
    # Analytics
    'analyze_trades',
    'calculate_performance_metrics',
    'calculate_max_drawdown',
    'calculate_composite_score',
    # Plotting
    'plot_strategy_engine',
    'plot_returns_analysis',
    # Utils
    'calculate_zscore',
    'setup_logging',
    # Optimization
    'auto_optimization',
    'grid_search_optimization',
    'random_search_optimization',
    'bayesian_optimization',
    'genetic_optimization',
    'extract_optimizer_status',
    # Walk-Forward Analysis
    'auto_wfa',
    'WFAResult',
    'WFAAggregatedResults',
    'print_wfa_summary'
]