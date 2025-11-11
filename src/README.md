# Crypto Statistical Arbitrage Framework

A Python framework for backtesting pairs trading strategies on cryptocurrency markets with support for dynamic hedge ratios and comprehensive performance analytics.

## IN DEVELOPMENT:  
1. Risk Management and Leverage architecture
2. Market Regime analysis
3. Volume filters

## Features

- **Multiple Beta Estimation Methods**
  - Static OLS regression
  - Rolling window OLS
  - Kalman filter (adaptive)

- **Flexible Trading Logic**
  - Z-score based entry/exit signals
  - Stop-loss protection with cooldown periods
  - Separate long/short z-score windows

- **Transaction Cost Modeling**
  - Commission rates
  - Slippage (basis points)

- **Comprehensive Analytics**
  - Trade-by-trade P&L breakdown
  - Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
  - Signal alignment diagnostics

## Installation

```bash
git clone https://github.com/mar-iov/crypto-statarb.git
cd crypto-statarb
pip install -r requirements.txt
```

For development:
```bash
pip install -e .
```


## Beta Estimation Methods

### Static OLS
Fixed hedge ratio estimated over entire sample.
```python
beta_method='static'
```

### Rolling Window
Time-varying beta using moving window OLS.
```python
beta_method='rolling'
beta_kwargs={'window': 100, 'min_periods': 50}
```

### Kalman Filter
Adaptive beta tracking with state-space model.
```python
beta_method='kalman'
beta_kwargs={'delta': 1e-5, 'R': 0.001}
```

delta: controls how fast beta is allowed to change — higher values -> more responsive but noisier.
R: observation noise variance — higher values -> smoother but slower adaptation.

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside deviation-adjusted return
- **Calmar Ratio**: Return vs max drawdown
- **Profit Factor**: Gross profit / gross loss
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline

## Walk-Forward Analysis Framework - Complete Guide

### What Was Built

A production-grade WFA framework with:
-  **4 WFA methodologies** (rolling, expanding, anchored, CPCV)
-  **Dynamic beta support** (static, rolling, Kalman + auto-selection)
-  **Cointegration monitoring** (pairs-trading specific)
-  **Smart optimizer integration** (grid, random, Bayesian, genetic)
-  **Comprehensive diagnostics** (stability, parameter distributions)

### WFA Methods Explained

#### Rolling Window
```
Train: [====]     Test: [==]
Train:      [====] Test:  [==]
Train:           [====] Test: [==]
```
**Use for**: Standard validation, parameter stability over time
**Best when**: Market conditions relatively stable

#### Expanding Window
```
Train: [====]           Test: [==]
Train: [========]       Test:  [==]
Train: [============]   Test:   [==]
```
**Use for**: Regime detection, increasing sample size
**Best when**: Want to see if more data improves performance

### Anchored Window
```
Train: [==========================]     Test: [==]
Train: [==========================]       Test:  [==]
Train: [==========================]         Test:   [==]
```
**Use for**: Comparing current vs historical performance
**Best when**: Want consistent reference point

### CPCV (Combinatorial Purged Cross-Validation)
```
Fold 1: Train: [====]  [====]  [====]  [====]   Test: [====]
Fold 2: Train: [====]  [====]  [====]  Test: [====] [====]
Fold 3: Train: [====]  [====]  Test: [====] [====]  [====]
                      (with purge/embargo gaps)
```
**Use for**: Best variance estimates, avoid sequential bias
**Best when**: Need robust out-of-sample metrics

---

### Understanding the Results

#### Stability Ratio
```python
stability_ratio = avg_test_sharpe / avg_train_sharpe
```

**-Interpretation**:
- `> 0.8`: Excellent - parameters are robust
- `0.5 - 0.8`: Good - some degradation expected
- `< 0.5`: Poor - severe overfitting

#### Cointegration Stability
```python
cointegration_stability = % of windows with p-value < 0.05
```

**-Interpretation**:
- `> 80%`: Excellent - pair is stable
- `60-80%`: Acceptable - monitor closely
- `< 60%`: Warning - pair relationship unstable

#### Parameter Distributions
```python
parameter_distributions = {
    'entry_threshold_long': {
        'mean': -1.8,
        'std': 0.3,  # Low std = stable parameter
        'min': -2.5,
        'max': -1.0
    }
}
```

**Low std**: Parameter is consistently optimal → robust
**High std**: Parameter jumps around → sensitive/unstable

---

### Usage Patterns

#### Pattern 1: Quick Validation (Rolling + Static Beta)
```python
results, aggregated, returns = enhanced_wfa(
    S1, S2,
    param_ranges=ranges,
    tf_min=3,
    train_days=60,
    test_days=30,
    wfa_method='rolling',
    beta_method='static',
    optimization_method='random_search'
)
```
**Use**: Initial strategy validation

#### Pattern 2: Robust Testing (CPCV + Rolling/Kalman Beta)
```python
results, aggregated, returns = enhanced_wfa(
    S1, S2,
    param_ranges=ranges,
    tf_min=3,
    wfa_method='cpcv',
    n_splits=5,
    embargo_pct=0.02,
    beta_method_selection=True,  # Auto-select best beta
    optimization_method='bayesian'
)
```
**Use**: Final validation before production

#### Pattern 3: Regime Analysis (Anchored + Kalman)
```python
results, aggregated, returns = enhanced_wfa(
    S1, S2,
    param_ranges=ranges,
    tf_min=3,
    train_days=90,
    test_days=30,
    wfa_method='anchored',
    beta_method='kalman',
    beta_kwargs={'delta': 1e-5},
    optimization_method='grid_search'
)
```

---

### 📈 Interpreting WFA Output

#### Good Results Example:
```
WALK-FORWARD ANALYSIS SUMMARY
================================================================================
Method: rolling
Total Periods: 10

Overall Performance:
  Total Return:         15.24%
  Sharpe Ratio:          1.85
  Max Drawdown:         -8.32%

Out-of-Sample Metrics:
  Avg Test Sharpe:       1.42
  Avg Train Sharpe:      1.68
  Stability Ratio:       0.85  ← GOOD (>0.8)

Cointegration Stability: 90.0% of periods  ← EXCELLENT (>80%)

Beta Method Distribution:
  kalman    :   6 ( 60.0%)  ← Adaptive beta often wins
  static    :   3 ( 30.0%)
  rolling   :   1 ( 10.0%)

Parameter Stability:
  entry_threshold_long     : μ= -1.800, σ=  0.250  ← Low std = stable
  z_window_long            : μ= 75.000, σ= 12.500
```

**-Interpretation**: **Ready for production**
- Stability ratio > 0.8 → Robust parameters
- Cointegration > 80% → Stable pair
- Test Sharpe > 1.0 → Has edge
- Low parameter std → Consistent optimization

#### Warning Signs Example:
```
Out-of-Sample Metrics:
  Avg Test Sharpe:       0.45  ← LOW
  Avg Train Sharpe:      2.10
  Stability Ratio:       0.21  ← BAD (<0.5)

Cointegration Stability: 45.0% of periods  ← WARNING (<60%)

Parameter Stability:
  entry_threshold_long     : μ= -1.500, σ=  1.850  ← High std = unstable
```

**Interpretation**: **NOT ready for production**
- Severe overfitting (train Sharpe 2.1, test 0.45)
- Cointegration breaks frequently
- Parameters unstable across windows

---

### Typical Beta Distribution:
- **Trending market**: Static can be viable (stable relationship, simple assumptions)
- **Volatile market**: Kalman often wins (adapts to changes)
- **Sideways market**: Rolling often wins (medium-term average)


---

### Decision Matrix: When to Deploy

| Metric | Threshold | Status |
|--------|-----------|--------|
| Stability Ratio | > 0.7 | ✅ Good |
| Cointegration Stability | > 70% | ✅ Good |
| Avg Test Sharpe | > 1.0 | ✅ Good |
| Parameter Std (normalized) | < 0.3 | ✅ Good |

**If ALL check** → Proceed to paper trading
**If 3/4 green** → Proceed with caution
**If <3 green** → Re-optimize or abandon pair

---

### Common Pitfalls

#### 1. **Insufficient Data**
```python
# BAD: Only 3 months of data, 10 windows
train_days=7, test_days=3  # Each window is tiny
```
**Fix**: Use longer windows (60/30 days minimum for crypto)

#### 2. **No Embargo in CPCV**
```python
# BAD: No embargo
enhanced_wfa(..., wfa_method='cpcv', embargo_pct=0.0)
```
**Fix**: Always use 1-2% embargo

#### 3. **Ignoring Cointegration**
If cointegration_stability < 60%, **don't deploy** even if other metrics look good.

#### 4. **Over-optimizing on Train**
If train Sharpe > 3.0 and test Sharpe < 1.0 → Pure overfitting

---

### Example Workflow

#### Step 1: Initial Validation
```python
# Quick scan with random search + rolling
enhanced_wfa(..., wfa_method='rolling', optimization_method='random_search')
```
**Decision**: If stability_ratio < 0.5 → Stop here

#### Step 2: Robust Validation
```python
# Thorough test with CPCV + Bayesian
enhanced_wfa(..., wfa_method='cpcv', optimization_method='bayesian')
```
**Decision**: If all metrics green → Proceed

#### Step 3: Regime Analysis
```python
# Check if performance degrading over time
enhanced_wfa(..., wfa_method='anchored')
```
**Decision**: If recent performance << historical → Re-evaluate

### Step 4: Paper Trading (30 days)
Deploy with best parameters + beta method, monitor live.

---

## Contributing

This is a personal project for educational purposes. Feel free to fork and adapt for your own use.

## Disclaimer

This framework is for educational and research purposes only. Not financial advice. Use at your own risk.