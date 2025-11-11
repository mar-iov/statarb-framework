"""
Walk-Forward Analysis framework with dynamic beta support.

Supports multiple WFA methodologies:
- Rolling window
- Expanding window
- Anchored window
- Combinatorial Purged Cross-Validation (CPCV)

Includes cointegration stability monitoring for pairs trading.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .optimization import auto_optimization
from .backtest import run_backtest
from .analytics import calculate_performance_metrics

logger = logging.getLogger(__name__)


@dataclass
class WFAResult:
    """Results from a single walk-forward period."""
    period: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    best_beta_method: str
    optimization_method: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_returns: pd.Series
    test_returns: pd.Series
    objective_value: float
    optimizer_iterations: int
    cointegration_pvalue: float
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WFAAggregatedResults:
    """Aggregated results across all WFA periods."""
    total_periods: int
    overall_metrics: Dict[str, float]
    avg_test_objective: float
    avg_train_objective: float
    stability_ratio: float
    parameter_distributions: Dict[str, Dict[str, float]]
    cointegration_stability: float
    beta_method_distribution: Dict[str, int]
    periods: List[WFAResult]


def auto_wfa(S1: pd.Series,
                 S2: pd.Series,
                 param_ranges: Dict,
                 tf_min: float,
                 train_days: int,
                 test_days: int,
                 cost_params: Optional[Dict] = None,
                 objective_metric: str = 'composite',
                 wfa_method: str = 'rolling',
                 optimization_method: Optional[str] = None,
                 beta_method: str = 'static',
                 beta_kwargs: Optional[Dict] = None,
                 min_data_points: int = 20,
                 cointegration_threshold: float = 0.05,
                 **kwargs) -> Tuple[List[WFAResult], WFAAggregatedResults, pd.Series]:
    """
    Enhanced Walk-Forward Analysis with dynamic beta support.

    Args:
        S1, S2: Price series
        param_ranges: Parameter search space
        tf_min: Timeframe in minutes
        train_days: Training period length
        test_days: Testing period length
        cost_params: Transaction costs
        objective_metric: Metric to optimize
        wfa_method: 'rolling', 'expanding', 'anchored', 'cpcv'
        optimization_method: Optimizer to use (None = auto-select)
        beta_method: 'static', 'rolling', 'kalman' (or auto if beta_method_selection=True)
        beta_kwargs: Parameters for beta estimation
        beta_method_selection: If True, optimizes beta method per window
        min_data_points: Minimum bars required
        cointegration_threshold: P-value threshold for cointegration test
        **kwargs: Method-specific parameters

    Returns:
        (wfa_results, aggregated_results, full_returns)
    """
    if cost_params is None:
        cost_params = {'commission_rate': 0.001, 'slippage_bps': 5}

    if beta_kwargs is None:
        beta_kwargs = {}

    # Select WFA method
    wfa_methods = {
        'rolling': _rolling_wfa,
        'expanding': _expanding_wfa,
        'anchored': _anchored_wfa,
        'cpcv': _cpcv_wfa
    }

    if wfa_method not in wfa_methods:
        raise ValueError(f"Unknown WFA method: {wfa_method}")

    logger.info(f"Starting {wfa_method} WFA with {beta_method} beta")

    # Run selected method
    wfa_func = wfa_methods[wfa_method]
    results, full_returns = wfa_func(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=tf_min,
        train_days=train_days,
        test_days=test_days,
        cost_params=cost_params,
        objective_metric=objective_metric,
        optimization_method=optimization_method,
        beta_method=beta_method,
        beta_kwargs=beta_kwargs,
        min_data_points=min_data_points,
        cointegration_threshold=cointegration_threshold,
        **kwargs
    )

    # Aggregate results
    aggregated = _aggregate_results(results, full_returns, tf_min, wfa_method, objective_metric)

    return results, aggregated, full_returns


def _rolling_wfa(S1: pd.Series,
                 S2: pd.Series,
                 param_ranges: Dict,
                 tf_min: float,
                 train_days: int,
                 test_days: int,
                 cost_params: Dict,
                 objective_metric: str,
                 optimization_method: Optional[str],
                 beta_method: str,
                 beta_kwargs: Dict,
                 min_data_points: int,
                 cointegration_threshold: float,
                 **kwargs) -> Tuple[List[WFAResult], pd.Series]:
    """
    Rolling window WFA.
    Train on fixed window, test on next period, roll forward.
    """
    dates = S1.index
    results = []
    full_returns = pd.Series(dtype=float)

    current_start = dates[0]
    period = 1

    while True:
        train_end = current_start + pd.DateOffset(days=train_days)
        test_end = train_end + pd.DateOffset(days=test_days)

        if test_end > dates[-1]:
            break

        # Masks
        train_mask = (dates >= current_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        if train_mask.sum() < min_data_points or test_mask.sum() < min_data_points:
            current_start += pd.DateOffset(days=test_days)
            continue

        # Process window
        result = _process_window(
            S1=S1[train_mask],
            S2=S2[train_mask],
            S1_test=S1[test_mask],
            S2_test=S2[test_mask],
            param_ranges=param_ranges,
            tf_min=tf_min,
            cost_params=cost_params,
            objective_metric=objective_metric,
            optimization_method=optimization_method,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            cointegration_threshold=cointegration_threshold,
            period=period,
            train_start=current_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            **kwargs
        )

        if result is not None:
            results.append(result)
            full_returns = pd.concat([full_returns, result.test_returns])
            period += 1

        current_start += pd.DateOffset(days=test_days)

    return results, full_returns


def _expanding_wfa(S1: pd.Series,
                   S2: pd.Series,
                   param_ranges: Dict,
                   tf_min: float,
                   train_days: int,
                   test_days: int,
                   cost_params: Dict,
                   objective_metric: str,
                   optimization_method: Optional[str],
                   beta_method: str,
                   beta_kwargs: Dict,
                   min_data_points: int,
                   cointegration_threshold: float,
                   **kwargs) -> Tuple[List[WFAResult], pd.Series]:
    """
    Expanding window WFA.
    Training window grows, test window fixed.
    """
    dates = S1.index
    results = []
    full_returns = pd.Series(dtype=float)

    anchor_start = dates[0]
    train_end = anchor_start + pd.DateOffset(days=train_days)
    period = 1

    while True:
        test_end = train_end + pd.DateOffset(days=test_days)

        if test_end > dates[-1]:
            break

        train_mask = (dates >= anchor_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        if train_mask.sum() < min_data_points or test_mask.sum() < min_data_points:
            train_end += pd.DateOffset(days=test_days)
            continue

        result = _process_window(
            S1=S1[train_mask],
            S2=S2[train_mask],
            S1_test=S1[test_mask],
            S2_test=S2[test_mask],
            param_ranges=param_ranges,
            tf_min=tf_min,
            cost_params=cost_params,
            objective_metric=objective_metric,
            optimization_method=optimization_method,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            cointegration_threshold=cointegration_threshold,
            period=period,
            train_start=anchor_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            **kwargs
        )

        if result is not None:
            results.append(result)
            full_returns = pd.concat([full_returns, result.test_returns])
            period += 1

        train_end += pd.DateOffset(days=test_days)

    return results, full_returns


def _anchored_wfa(S1: pd.Series,
                  S2: pd.Series,
                  param_ranges: Dict,
                  tf_min: float,
                  train_days: int,
                  test_days: int,
                  cost_params: Dict,
                  objective_metric: str,
                  optimization_method: Optional[str],
                  beta_method: str,
                  beta_kwargs: Dict,
                  min_data_points: int,
                  cointegration_threshold: float,
                  **kwargs) -> Tuple[List[WFAResult], pd.Series]:
    """
    Anchored window WFA.
    Training always starts from beginning, test window rolls forward.
    """
    dates = S1.index
    results = []
    full_returns = pd.Series(dtype=float)

    anchor_start = dates[0]
    current_test_start = anchor_start + pd.DateOffset(days=train_days)
    period = 1

    while True:
        test_end = current_test_start + pd.DateOffset(days=test_days)

        if test_end > dates[-1]:
            break

        train_mask = (dates >= anchor_start) & (dates < current_test_start)
        test_mask = (dates >= current_test_start) & (dates < test_end)

        if train_mask.sum() < min_data_points or test_mask.sum() < min_data_points:
            current_test_start += pd.DateOffset(days=test_days)
            continue

        result = _process_window(
            S1=S1[train_mask],
            S2=S2[train_mask],
            S1_test=S1[test_mask],
            S2_test=S2[test_mask],
            param_ranges=param_ranges,
            tf_min=tf_min,
            cost_params=cost_params,
            objective_metric=objective_metric,
            optimization_method=optimization_method,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            cointegration_threshold=cointegration_threshold,
            period=period,
            train_start=anchor_start,
            train_end=current_test_start,
            test_start=current_test_start,
            test_end=test_end,
            **kwargs
        )

        if result is not None:
            results.append(result)
            full_returns = pd.concat([full_returns, result.test_returns])
            period += 1

        current_test_start += pd.DateOffset(days=test_days)

    return results, full_returns


def _cpcv_wfa(S1: pd.Series,
              S2: pd.Series,
              param_ranges: Dict,
              tf_min: float,
              train_days: int,  # Not used directly, n_splits controls divisions
              test_days: int,   # Not used directly
              cost_params: Dict,
              objective_metric: str,
              optimization_method: Optional[str],
              beta_method: str,
              beta_kwargs: Dict,
              min_data_points: int,
              cointegration_threshold: float,
              n_splits: int = 5,
              embargo_pct: float = 0.01,
              **kwargs) -> Tuple[List[WFAResult], pd.Series]:
    """
    Combinatorial Purged Cross-Validation.

    Key differences from standard k-fold:
    - Purges overlapping periods between train/test
    - Embargo period prevents leakage
    - Tests on non-contiguous blocks

    Args:
        n_splits: Number of folds
        embargo_pct: Percentage of data to embargo around test set
    """
    dates = S1.index
    n = len(dates)

    if n < min_data_points * n_splits:
        raise ValueError(f"Insufficient data for {n_splits} splits")

    # Create fold boundaries
    fold_size = n // n_splits
    fold_bounds = []
    for i in range(n_splits):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_splits - 1 else n
        fold_bounds.append((start_idx, end_idx))

    embargo_size = int(n * embargo_pct)
    results = []
    full_returns = pd.Series(dtype=float)

    for k, (test_start_idx, test_end_idx) in enumerate(fold_bounds):
        # Test indices
        test_dates = dates[test_start_idx:test_end_idx]
        test_mask = S1.index.isin(test_dates)

        # Train indices (everything else with embargo)
        purge_start = max(0, test_start_idx - embargo_size)
        purge_end = min(n, test_end_idx + embargo_size)

        train_indices = list(range(0, purge_start)) + list(range(purge_end, n))
        if len(train_indices) < min_data_points:
            logger.warning(f"Fold {k+1}: insufficient train data after purge/embargo")
            continue

        train_dates = dates[train_indices]
        train_mask = S1.index.isin(train_dates)

        if train_mask.sum() < min_data_points or test_mask.sum() < min_data_points:
            continue

        result = _process_window(
            S1=S1[train_mask],
            S2=S2[train_mask],
            S1_test=S1[test_mask],
            S2_test=S2[test_mask],
            param_ranges=param_ranges,
            tf_min=tf_min,
            cost_params=cost_params,
            objective_metric=objective_metric,
            optimization_method=optimization_method,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            cointegration_threshold=cointegration_threshold,
            period=k + 1,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            **kwargs
        )

        if result is not None:
            results.append(result)
            full_returns = pd.concat([full_returns, result.test_returns])

    return results, full_returns


def _process_window(S1: pd.Series,
                    S2: pd.Series,
                    S1_test: pd.Series,
                    S2_test: pd.Series,
                    param_ranges: Dict,
                    tf_min: float,
                    cost_params: Dict,
                    objective_metric: str,
                    optimization_method: Optional[str],
                    beta_method: str,
                    beta_kwargs: Dict,
                    cointegration_threshold: float,
                    period: int,
                    train_start: pd.Timestamp,
                    train_end: pd.Timestamp,
                    test_start: pd.Timestamp,
                    test_end: pd.Timestamp,
                    **kwargs) -> Optional[WFAResult]:
    """
    Process a single train/test window.

    Returns None if cointegration test fails or optimization fails.
    """
    # Check cointegration
    _, p_value, _ = coint(S1, S2)

    if p_value > cointegration_threshold:
        logger.warning(f"Period {period}: Cointegration FAILED (p={p_value:.4f})")


    try:
        final_params, final_value, diagnostics = auto_optimization(
            S1=S1,
            S2=S2,
            param_ranges=param_ranges,
            tf_min=tf_min,
            cost_params=cost_params,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            objective_metric=objective_metric,
            optimization_method=optimization_method,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Period {period}: Optimization failed - {e}")
        return None

    # Backtest on train
    train_results = run_backtest(
        S1=S1,
        S2=S2,
        beta=None,
        beta_method=beta_method,
        beta_kwargs=beta_kwargs,
        params=final_params,
        cost_params=cost_params
    )
    train_returns = train_results[1]

    # Backtest on test
    test_results = run_backtest(
        S1=S1_test,
        S2=S2_test,
        beta=None,
        beta_method=beta_method,
        beta_kwargs=beta_kwargs,
        params=final_params,
        cost_params=cost_params
    )
    test_returns = test_results[1]

    # Calculate metrics
    train_metrics = calculate_performance_metrics(
        returns=train_returns,
        tf_min=tf_min,
        beta=train_results[7],
        beta_method=beta_method
    )

    test_metrics = calculate_performance_metrics(
        returns=test_returns,
        tf_min=tf_min,
        beta=test_results[7],
        beta_method=beta_method
    )

    logger.info(f"Period {period}: Train {objective_metric}={train_metrics[f'{objective_metric}']:.2f}, "
                f"Test {objective_metric}={test_metrics[f'{objective_metric}']:.2f}, Beta={beta_method}")

    return WFAResult(
        period=period,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        best_params=final_params,
        best_beta_method=beta_method,
        optimization_method=optimization_method or 'auto',
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_returns=train_returns,
        test_returns=test_returns,
        objective_value=final_value,
        optimizer_iterations=0,  # Could track from diagnostics
        cointegration_pvalue=p_value
    )


def _aggregate_results(results: List[WFAResult],
                       full_returns: pd.Series,
                       tf_min: float,
                       wfa_method: str,
                       objective_metric: str) -> WFAAggregatedResults:
    """Aggregate results across all periods."""
    if not results:
        raise ValueError("No valid WFA results to aggregate")

    # Overall metrics
    overall_metrics = calculate_performance_metrics(
        returns=full_returns,
        tf_min=tf_min
    )
    overall_metrics['wfa_method'] = wfa_method
    overall_metrics['n_periods'] = len(results)

    # Averages
    test_objective = [r.test_metrics[f'{objective_metric}'] for r in results]
    train_objective = [r.train_metrics[f'{objective_metric}'] for r in results]

    avg_test_objective = np.mean(test_objective)
    avg_train_objective = np.mean(train_objective)
    stability_ratio = avg_test_objective / avg_train_objective if avg_train_objective != 0 else 0

    # Parameter distributions
    param_dist = defaultdict(list)
    for r in results:
        for k, v in r.best_params.items():
            if isinstance(v, (int, float)):
                param_dist[k].append(v)

    param_distributions = {
        k: {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'min': float(np.min(v)),
            'max': float(np.max(v))
        }
        for k, v in param_dist.items()
    }

    # Cointegration stability
    coint_pvalues = [r.cointegration_pvalue for r in results]
    cointegration_stability = np.mean([p < 0.05 for p in coint_pvalues])

    # Beta method distribution
    beta_methods = [r.best_beta_method for r in results]
    beta_method_dist = {
        method: beta_methods.count(method)
        for method in set(beta_methods)
    }

    return WFAAggregatedResults(
        total_periods=len(results),
        overall_metrics=overall_metrics,
        avg_test_objective=avg_test_objective,
        avg_train_objective=avg_train_objective,
        stability_ratio=stability_ratio,
        parameter_distributions=param_distributions,
        cointegration_stability=cointegration_stability,
        beta_method_distribution=beta_method_dist,
        periods=results
    )


def print_wfa_summary(aggregated: WFAAggregatedResults,
                      objective_metric: str):
    """Print comprehensive WFA summary."""
    print("\n" + "="*80)
    print("WALK-FORWARD ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nMethod: {aggregated.overall_metrics['wfa_method']}")
    print(f"Total Periods: {aggregated.total_periods}")

    print(f"\nOverall Performance:")
    print(f"  Total Return:        {aggregated.overall_metrics['total_return']*100:>8.2f}%")
    print(f"  Objective Metric:        {aggregated.overall_metrics[f'{objective_metric}']:>8.2f}")
    print(f"  Max Drawdown:        {aggregated.overall_metrics['max_drawdown']*100:>8.2f}%")

    print(f"\nOut-of-Sample Metrics:")
    print(f"  Avg Test Objective Metric:     {aggregated.avg_test_objective:>8.2f}")
    print(f"  Avg Train Objective Metric:    {aggregated.avg_train_objective:>8.2f}")
    print(f"  Stability Ratio:     {aggregated.stability_ratio:>8.2f}")

    print(f"\nCointegration Stability: {aggregated.cointegration_stability*100:.1f}% of periods")

    print(f"\nBeta Method Distribution:")
    for method, count in aggregated.beta_method_distribution.items():
        pct = count / aggregated.total_periods * 100
        print(f"  {method:10s}: {count:>3d} ({pct:>5.1f}%)")

    print(f"\nParameter Stability:")
    for param, stats in aggregated.parameter_distributions.items():
        print(f"  {param:25s}: μ={stats['mean']:>7.3f}, σ={stats['std']:>6.3f}")