"""
Parameter optimization methods for pairs trading strategies.

Supports grid search, random search, Bayesian, and genetic algorithms
with dynamic beta estimation (static, rolling, Kalman).
"""

import logging
import random
import itertools
import time
from typing import Dict, Tuple, Optional, Any, List, Callable
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from .backtest import run_backtest
from .analytics import calculate_performance_metrics

logger = logging.getLogger(__name__)

def validate_params(params: Dict, constraints: Optional[List[Dict]] = None) -> bool:
    """
    Validate parameter combinations against logical constraints.
    
    Args:
        params: Parameter dictionary to validate
        constraints: List of constraint dictionaries

    Returns:
        True if valid, False otherwise
    """
    if constraints is None or not constraints:
        return True

    for constraint in constraints:
        ctype = constraint.get('type', 'order')
        rule = constraint['params']

        if ctype == 'order':
            param1, op, param2 = rule
            val1 = params.get(param1)
            val2 = params.get(param2) if isinstance(param2, str) else param2

            if val1 is None or val2 is None:
                continue

            if op == '<' and not (val1 < val2):
                return False
            elif op == '>' and not (val1 > val2):
                return False
            elif op == '<=' and not (val1 <= val2):
                return False
            elif op == '>=' and not (val1 >= val2):
                return False
            elif op == '==' and not (val1 == val2):
                return False
            elif op == '!=' and not (val1 != val2):
                return False

        elif ctype == 'range':
            param, min_val, max_val = rule
            val = params.get(param)
            if val is None:
                continue
            if not (min_val <= val <= max_val):
                return False

    return True


def _evaluate_single_params(params: Dict,
                            S1: pd.Series,
                            S2: pd.Series,
                            tf_min: float,
                            cost_params: Dict,
                            beta_method: str,
                            beta_kwargs: Dict,
                            min_data_points: int,
                            target_column: str) -> Optional[Dict]:
    """Worker function for parallel evaluation."""
    try:
        bt_results = run_backtest(
            S1=S1,
            S2=S2,
            beta=None,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            params=params,
            cost_params=cost_params
        )

        positions, returns_net = bt_results[0], bt_results[1]

        if len(returns_net) < min_data_points or returns_net.std() == 0:
            return None

        metrics = calculate_performance_metrics(
            returns=returns_net,
            tf_min=tf_min,
            beta=bt_results[7],
            beta_method=beta_method
        )

        if not np.isfinite(metrics.get(target_column, np.nan)):
            return None

        num_trades = int((positions.diff().abs() > 0).sum())

        return {
            **params,
            **metrics,
            'num_trades': num_trades,
            'beta_method': beta_method
        }

    except Exception as e:
        logger.debug(f"Evaluation failed: {e}")
        return None


def auto_optimization(S1: pd.Series,                          #WIP
                          S2: pd.Series,
                          param_ranges: Dict,
                          tf_min: float,
                          cost_params: Optional[Dict] = None,
                          beta_method: str = 'static',
                          beta_kwargs: Optional[Dict] = None,
                          objective_metric: str = 'composite',
                          optimization_method: str = 'grid_search',
                          min_data_points: int = 20,
                          constraints: Optional[List[Dict]] = None,
                          n_jobs: int = 1,
                          **kwargs) -> Tuple[Dict, float, pd.DataFrame]:
    """
    Unified optimization interface with automatic method selection.
    """
    if cost_params is None:
        cost_params = {'commission_rate': 0.000, 'slippage_bps': 0}

    if beta_kwargs is None:
        beta_kwargs = {}

    optimizers = {
        'grid_search': grid_search_optimization,
        'random_search': random_search_optimization,
        'bayesian': bayesian_optimization,
        'genetic': genetic_optimization
    }

    if optimization_method not in optimizers:
        raise ValueError(f"Unknown method: {optimization_method}")

    logger.info(f"Starting {optimization_method} optimization with {beta_method} beta")

    optimizer_func = optimizers[optimization_method]

    return optimizer_func(
        S1=S1,
        S2=S2,
        param_ranges=param_ranges,
        tf_min=tf_min,
        cost_params=cost_params,
        beta_method=beta_method,
        beta_kwargs=beta_kwargs,
        objective_metric=objective_metric,
        min_data_points=min_data_points,
        constraints=constraints,
        n_jobs=n_jobs,
        **kwargs
    )


def grid_search_optimization(S1: pd.Series,
                             S2: pd.Series,
                             param_ranges: Dict,
                             tf_min: float,
                             cost_params: Optional[Dict] = None,
                             beta_method: str = 'static',
                             beta_kwargs: Optional[Dict] = None,
                             objective_metric: str = 'composite',
                             min_data_points: int = 20,
                             constraints: Optional[List[Dict]] = None,
                             early_stop_threshold: Optional[float] = None,
                             early_stop_patience: int = 100,
                             n_jobs: int = 1,
                             **kwargs) -> Tuple[Dict, float, pd.DataFrame]:
    """
    Exhaustive grid search over parameter space.
    """
    if cost_params is None:
        cost_params = {'commission_rate': 0.000, 'slippage_bps': 0}

    if cost_params is "standard":
        cost_params = {'commission_rate': 0.005, 'slippage_bps': 5}

    if beta_kwargs is None:
        beta_kwargs = {}

    metric_to_column = {
        'sharpe': 'sharpe',
        'sortino': 'sortino',
        'calmar': 'calmar',
        'composite': 'composite',
        'profit_factor': 'profit_factor',
        'total_return': 'total_return'
    }
    target_column = metric_to_column.get(objective_metric)
    if target_column is None:
        raise ValueError(f"Invalid objective_metric: {objective_metric}")

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combinations = list(itertools.product(*param_values))

    if constraints:
        original_count = len(all_combinations)
        all_combinations = [
            combo for combo in all_combinations
            if validate_params(dict(zip(param_names, combo)), constraints)
        ]
        filtered_count = original_count - len(all_combinations)
        if filtered_count > 0:
            print(f"Filtered {filtered_count}/{original_count} invalid combinations")

    print(f"Grid search: Testing {len(all_combinations)} combinations ({beta_method} beta, {n_jobs} jobs)...")

    if n_jobs == -1:
        n_jobs = cpu_count()

    if n_jobs > 1 and len(all_combinations) > 10:
        worker = partial(
            _evaluate_single_params,
            S1=S1,
            S2=S2,
            tf_min=tf_min,
            cost_params=cost_params,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            min_data_points=min_data_points,
            target_column=target_column
        )

        param_dicts = [dict(zip(param_names, combo)) for combo in all_combinations]

        with Pool(n_jobs) as pool:
            results = pool.map(worker, param_dicts)

        results = [r for r in results if r is not None]

    else:
        results = []
        best_value = -np.inf
        no_improvement_count = 0

        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))

            result = _evaluate_single_params(
                params, S1, S2, tf_min, cost_params,
                beta_method, beta_kwargs, min_data_points, target_column
            )

            if result is not None:
                results.append(result)

                if result[target_column] > best_value:
                    best_value = result[target_column]
                    no_improvement_count = 0

                    if early_stop_threshold and best_value >= early_stop_threshold:
                        print(f"  Early stop: threshold reached at iteration {i+1}")
                        break
                else:
                    no_improvement_count += 1

                    if no_improvement_count >= early_stop_patience:
                        print(f"  Early stop: no improvement for {early_stop_patience} iterations")
                        break

            if (i + 1) % max(1, len(all_combinations) // 10) == 0:
                pct = (i + 1) / len(all_combinations) * 100
                current_best = max([r[target_column] for r in results]) if results else np.nan
                print(f"  Progress: {pct:.1f}% | Best {objective_metric}: {current_best:.4f}")

    if not results:
        raise ValueError("No valid parameter combinations found!")

    results_df = pd.DataFrame(results)
    best_idx = results_df[target_column].idxmax()
    best_params = results_df.loc[best_idx, param_names].to_dict()
    best_value = float(results_df.loc[best_idx, target_column])

    print(f"Grid search complete: Best {objective_metric} = {best_value:.4f}")

    return best_params, best_value, results_df


def random_search_optimization(S1: pd.Series,
                               S2: pd.Series,
                               param_ranges: Dict,
                               tf_min: float,
                               cost_params: Optional[Dict] = None,
                               beta_method: str = 'static',
                               beta_kwargs: Optional[Dict] = None,
                               objective_metric: str = 'composite',
                               min_data_points: int = 20,
                               constraints: Optional[List[Dict]] = None,
                               n_iter: int = 100,
                               early_stop_threshold: Optional[float] = None,
                               early_stop_patience: int = 20,
                               n_jobs: int = 1,
                               **kwargs) -> Tuple[Dict, float, pd.DataFrame]:
    """
    Random sampling of parameter space.
    """
    if cost_params is None:
        cost_params = {'commission_rate': 0.000, 'slippage_bps': 0}

    if beta_kwargs is None:
        beta_kwargs = {}

    metric_to_column = {
        'sharpe': 'sharpe',
        'sortino': 'sortino',
        'calmar': 'calmar',
        'composite': 'composite',
        'profit_factor': 'profit_factor',
        'total_return': 'total_return'
    }
    target_column = metric_to_column.get(objective_metric)
    if target_column is None:
        raise ValueError(f"Invalid objective_metric: {objective_metric}")

    param_names = list(param_ranges.keys())

    def sample_params():
        max_attempts = 1000
        for _ in range(max_attempts):
            params = {}
            for name, values in param_ranges.items():
                if len(values) == 2 and all(isinstance(v, (int, np.integer)) for v in values):
                    params[name] = random.randint(min(values), max(values))
                elif len(values) == 2 and all(isinstance(v, (float, int)) for v in values):
                    params[name] = random.uniform(min(values), max(values))
                else:
                    params[name] = random.choice(values)

            if validate_params(params, constraints):
                return params

        raise ValueError("Could not generate valid params")

    print(f"Random search: Testing {n_iter} combinations ({beta_method} beta, {n_jobs} jobs)...")

    if n_jobs == -1:
        n_jobs = cpu_count()

    if n_jobs > 1:
        all_params = []
        for _ in range(n_iter):
            try:
                all_params.append(sample_params())
            except ValueError:
                continue

        worker = partial(
            _evaluate_single_params,
            S1=S1,
            S2=S2,
            tf_min=tf_min,
            cost_params=cost_params,
            beta_method=beta_method,
            beta_kwargs=beta_kwargs,
            min_data_points=min_data_points,
            target_column=target_column
        )

        with Pool(n_jobs) as pool:
            results = pool.map(worker, all_params)

        results = [r for r in results if r is not None]

    else:
        results = []
        best_so_far = -np.inf
        no_improvement_count = 0

        for i in range(n_iter):
            try:
                params = sample_params()
            except ValueError:
                continue

            result = _evaluate_single_params(
                params, S1, S2, tf_min, cost_params,
                beta_method, beta_kwargs, min_data_points, target_column
            )

            if result is not None:
                results.append(result)

                if result[target_column] > best_so_far:
                    best_so_far = result[target_column]
                    no_improvement_count = 0

                    if early_stop_threshold and best_so_far >= early_stop_threshold:
                        print(f"  Early stop: threshold reached")
                        break
                else:
                    no_improvement_count += 1

                    if no_improvement_count >= early_stop_patience:
                        print(f"  Early stop: no improvement")
                        break

            if (i + 1) % max(1, n_iter // 10) == 0:
                print(f"  Progress: {i+1}/{n_iter} | Best: {best_so_far:.4f}")

    if not results:
        raise ValueError("No valid parameter combinations found!")

    results_df = pd.DataFrame(results)
    best_idx = results_df[target_column].idxmax()
    best_params = results_df.loc[best_idx, param_names].to_dict()
    best_value = float(results_df.loc[best_idx, target_column])

    print(f"Random search complete: Best {objective_metric} = {best_value:.4f}")

    return best_params, best_value, results_df


def bayesian_optimization(S1: pd.Series,
                          S2: pd.Series,
                          param_ranges: Dict,
                          tf_min: float,
                          cost_params: Optional[Dict] = None,
                          beta_method: str = 'static',
                          beta_kwargs: Optional[Dict] = None,
                          objective_metric: str = 'composite',
                          min_data_points: int = 20,
                          constraints: Optional[List[Dict]] = None,
                          n_iter: int = 50,
                          penalty_value: float = 1e6,
                          **kwargs) -> Tuple[Dict, float, pd.DataFrame]:
    """Bayesian optimization using Gaussian Processes."""
    if cost_params is None:
        cost_params = {'commission_rate': 0.000, 'slippage_bps': 0}

    if beta_kwargs is None:
        beta_kwargs = {}

    metric_to_column = {
        'sharpe': 'sharpe',
        'sortino': 'sortino',
        'calmar': 'calmar',
        'composite': 'composite',
        'profit_factor': 'profit_factor',
        'total_return': 'total_return'
    }
    target_column = metric_to_column.get(objective_metric, 'composite')

    dimensions = []
    param_names = []
    int_params = set()

    for name, values in param_ranges.items():
        unique_vals = sorted(set(values))

        if all(isinstance(x, (int, np.integer)) for x in unique_vals):
            if len(unique_vals) > 2:
                is_contiguous = all((unique_vals[i+1] - unique_vals[i]) == 1
                                   for i in range(len(unique_vals)-1))
                if is_contiguous:
                    dimensions.append(Integer(min(unique_vals), max(unique_vals), name=name))
                else:
                    dimensions.append(Categorical(unique_vals, name=name))
            else:
                dimensions.append(Categorical(unique_vals, name=name))
            int_params.add(name)

        elif all(isinstance(x, (int, float, np.integer, np.floating)) for x in unique_vals):
            if len(unique_vals) > 2:
                dimensions.append(Real(min(unique_vals), max(unique_vals), name=name))
            else:
                dimensions.append(Categorical(unique_vals, name=name))

        else:
            dimensions.append(Categorical(unique_vals, name=name))

        param_names.append(name)

    valid_trials = []

    @use_named_args(dimensions=dimensions)
    def objective_function(**params):
        for k in int_params:
            if k in params:
                try:
                    params[k] = int(round(params[k]))
                except Exception:
                    pass

        if not validate_params(params, constraints):
            return float(penalty_value)

        try:
            bt_results = run_backtest(
                S1=S1, S2=S2, beta=None,
                beta_method=beta_method,
                beta_kwargs=beta_kwargs,
                params=params,
                cost_params=cost_params
            )

            positions, returns_net = bt_results[0], bt_results[1]

            if len(returns_net) < min_data_points or returns_net.std() == 0:
                return float(penalty_value)

            metrics = calculate_performance_metrics(
                returns=returns_net,
                tf_min=tf_min,
                beta=bt_results[7],
                beta_method=beta_method
            )

            target_value = metrics.get(target_column)

            if target_value is None or not np.isfinite(target_value):
                return float(penalty_value)

            num_trades = int((positions.diff().abs() > 0).sum())
            valid_trials.append({**params, **metrics, 'num_trades': num_trades, 'beta_method': beta_method})

            return float(-target_value)

        except Exception:
            return float(penalty_value)

    print(f"Bayesian optimization: {n_iter} iterations...")

    result = gp_minimize(
        objective_function,
        dimensions,
        n_calls=n_iter,
        random_state=42,
        verbose=False
    )

    results_df = pd.DataFrame(valid_trials)

    if results_df.empty:
        return {}, -np.inf, pd.DataFrame()

    if target_column in results_df.columns:
        best_idx = results_df[target_column].idxmax()
        best_params = results_df.loc[best_idx, param_names].to_dict()
        best_value = float(results_df.loc[best_idx, target_column])
    else:
        best_params = dict(zip(param_names, result.x))
        best_value = float(-result.fun) if np.isfinite(result.fun) else -np.inf

    print(f"Bayesian complete: Best {objective_metric} = {best_value:.4f}")

    return best_params, best_value, results_df


def genetic_optimization(S1: pd.Series,
                        S2: pd.Series,
                        param_ranges: Dict,
                        tf_min: float,
                        cost_params: Optional[Dict] = None,
                        beta_method: str = 'static',
                        beta_kwargs: Optional[Dict] = None,
                        objective_metric: str = 'composite',
                        min_data_points: int = 20,
                        constraints: Optional[List[Dict]] = None,
                        population_size: int = 30,
                        generations: int = 20,
                        mutation_rate: float = 0.2,
                        elite_fraction: float = 0.2,
                        patience: int = 5,
                        tol: float = 1e-4,
                        **kwargs) -> Tuple[Dict, float, pd.DataFrame]:
    """Genetic algorithm optimization."""
    if cost_params is None:
        cost_params = {'commission_rate': 0.000, 'slippage_bps': 0}

    if beta_kwargs is None:
        beta_kwargs = {}

    metric_to_column = {
        'sharpe': 'sharpe',
        'sortino': 'sortino',
        'calmar': 'calmar',
        'composite': 'composite',
        'profit_factor': 'profit_factor',
        'total_return': 'total_return'
    }
    target_column = metric_to_column[objective_metric]

    param_names = list(param_ranges.keys())
    int_params = {k for k, v in param_ranges.items()
                  if isinstance(v[0], (int, np.integer))}

    def sample_individual():
        max_attempts = 1000
        for _ in range(max_attempts):
            params = {}
            for k, v in param_ranges.items():
                if k in int_params:
                    params[k] = random.randint(min(v), max(v))
                else:
                    params[k] = random.uniform(min(v), max(v))

            if validate_params(params, constraints):
                return params

        raise ValueError("Could not generate valid individual")

    def evaluate(params):
        try:
            bt_results = run_backtest(
                S1=S1, S2=S2, beta=None,
                beta_method=beta_method,
                beta_kwargs=beta_kwargs,
                params=params,
                cost_params=cost_params
            )

            positions, returns_net = bt_results[0], bt_results[1]

            if len(returns_net) < min_data_points or returns_net.std() == 0:
                return -np.inf, None

            metrics = calculate_performance_metrics(
                returns=returns_net,
                tf_min=tf_min,
                beta=bt_results[7],
                beta_method=beta_method
            )

            score = metrics.get(target_column, -np.inf)
            if not np.isfinite(score):
                return -np.inf, None

            metrics['num_trades'] = int((positions.diff().abs() > 0).sum())
            metrics['beta_method'] = beta_method
            return score, metrics

        except Exception:
            return -np.inf, None

    def crossover(parent1, parent2):
        max_attempts = 10
        for _ in range(max_attempts):
            child = {}
            for k in param_ranges.keys():
                child[k] = parent1[k] if random.random() < 0.5 else parent2[k]

            if validate_params(child, constraints):
                return child

        return {k: parent1[k] for k in param_ranges.keys()}

    def mutate(individual):
        max_attempts = 10
        for _ in range(max_attempts):
            mutated = individual.copy()
            for k, v in param_ranges.items():
                if random.random() < mutation_rate:
                    if k in int_params:
                        mutated[k] = random.randint(min(v), max(v))
                    else:
                        mutated[k] = random.uniform(min(v), max(v))

            if validate_params(mutated, constraints):
                return mutated

        return individual

    population = []
    for _ in range(population_size):
        try:
            population.append(sample_individual())
        except ValueError:
            break

    if not population:
        raise ValueError("Could not generate any valid individuals")

    results = []
    best_value, best_params = -np.inf, None
    no_improvement = 0

    print(f"Genetic optimization: {len(population)} individuals x {generations} generations...")

    for gen in range(generations):
        evaluated = []
        for ind in population:
            score, metrics = evaluate(ind)
            if metrics is not None:
                evaluated.append({**ind, **metrics, 'fitness': score})

        if not evaluated:
            continue

        gen_df = pd.DataFrame(evaluated)
        results.append(gen_df)

        gen_best_idx = gen_df['fitness'].idxmax()
        gen_best_val = gen_df.loc[gen_best_idx, 'fitness']

        if gen_best_val > best_value * (1 + tol):
            best_value = gen_best_val
            best_params = gen_df.loc[gen_best_idx, param_names].to_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        print(f"  Gen {gen + 1}/{generations} | Best: {best_value:.4f}")

        if no_improvement >= patience:
            print("  Early stopping")
            break

        elite_size = max(1, int(elite_fraction * population_size))
        elites = gen_df.nlargest(elite_size, 'fitness').to_dict('records')

        new_population = [{k: ind[k] for k in param_names} for ind in elites]

        while len(new_population) < population_size:
            parents = random.sample(elites, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)

        population = new_population

    all_results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    print(f"Genetic complete: Best {objective_metric} = {best_value:.4f}")

    return best_params, best_value, all_results


def extract_optimizer_status(result: Any, optimization_method: str) -> str:
    """Extract standardized status from optimizer results."""
    if optimization_method == 'bayesian':
        if hasattr(result, 'success') and result.success:
            return 'converged'
        elif hasattr(result, 'n_calls'):
            max_calls = getattr(result, 'max_calls', result.n_calls)
            if result.n_calls >= max_calls:
                return 'max_iter_reached'
            return 'incomplete'
        return 'unknown'

    elif optimization_method in ['genetic', 'random_search', 'grid_search']:
        if isinstance(result, pd.DataFrame) and not result.empty:
            return 'completed'
        return 'failed'

    if isinstance(result, dict) and 'error' in result:
        return 'failed'

    return 'unknown'