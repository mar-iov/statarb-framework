"""
Utility functions and logging setup for the statistical arbitrage framework.
"""

import logging
import pandas as pd


def setup_logging(level=logging.INFO, verbose_modules=False):
    """
    Configure logging for the package.

    Args:
        level: Logging level (logging.INFO, logging.WARNING, etc.)
        verbose_modules: If False, reduces verbosity of specific modules
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Reduce verbosity for specific modules
    if not verbose_modules:
        # Reduce beta estimation logging to WARNING
        logging.getLogger('statarb.beta_estimation').setLevel(logging.WARNING)
        # Reduce backtest logging
        logging.getLogger('statarb.backtest').setLevel(logging.WARNING)
        # Keep optimization and WFA at INFO level
        logging.getLogger('statarb.optimization').setLevel(logging.INFO)
        logging.getLogger('statarb.walk_forward').setLevel(logging.INFO)


def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling z-score with look-ahead bias prevention.

    Mean and std are computed on past data only (shifted by 1 bar).

    Args:
        series: Input time series
        window: Rolling window size

    Returns:
        Z-score series
    """
    mean = series.rolling(window=window).mean().shift(1)
    std = series.rolling(window=window).std(ddof=0).shift(1)
    zscore = (series - mean) / std
    return zscore