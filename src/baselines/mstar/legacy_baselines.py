import numpy as np
from typing import Sequence, Tuple, Callable, Optional, Dict, Any
import optuna
from optuna import Trial, logging
from optuna.logging import disable_default_handler, set_verbosity, enable_default_handler
from optuna.samplers import TPESampler
from scipy.stats import truncnorm
from itertools import product
from itertools import islice
from optuna.logging import get_logger
from scipy.special import gamma

class RandomSearch:

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.bounds = bounds
        self.objective_function = objective_function
        self.minimize = minimize
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        self.n_iter = kwargs.get('n_iter', 100)
        self.budget = kwargs.get('budget', 1000)
        self.samples_per_iter = max(1, self.budget // self.n_iter)
        if random_state is not None:
            np.random.seed(random_state)
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_params = None

    def _generate_random_point(self) -> Sequence[float]:
        point = []
        for low, high in self.bounds:
            point.append(np.random.uniform(low, high))
        return point

    def optimize(self) -> Dict[str, Any]:
        if self.verbose:
            print(f'开始随机搜索，总迭代次数: {self.n_iter}')
        for i in range(self.n_iter):
            for _ in range(self.samples_per_iter):
                params = self._generate_random_point()
                value = self.objective_function(params)
                self.history['params'].append(params)
                self.history['values'].append(value)
                if self.minimize and value < self.best_value or (not self.minimize and value > self.best_value):
                    self.best_value = value
                    self.best_params = params.copy()
            self.history['best_values'].append(self.best_value)
            self.history['best_params'].append(self.best_params)
            if self.verbose and (i + 1) % max(1, self.n_iter // 10) == 0:
                print(f'迭代 {i + 1}/{self.n_iter}，当前最佳值: {self.best_value:.6f}')
        if self.verbose:
            print(f'优化完成，最佳值: {self.best_value:.6f}')
            print(f'最佳参数: {self.best_params}')
        results = {'best_params': self.best_params, 'best_value': self.best_value, 'history': self.history}
        return results

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.best_params, self.best_value)

class BayesianOptimizationWithTPE:

    def __init__(self, n_initial_points: int, n_points_per_generation: int, n_iter: int, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, show_progress_bar: bool=False, **kwargs):
        if optuna is None:
            raise ImportError('Optuna is required. Install with: pip install optuna')
        self.n_initial_points = n_initial_points
        self.n_points_per_generation = n_points_per_generation
        self.n_iter = n_iter
        self.bounds = bounds
        self.objective_function = objective_function
        self.minimize = minimize
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self._configure_logging()
        self._initialize_optimizer()

    def _configure_logging(self):
        if not self.verbose:
            disable_default_handler()
            set_verbosity(logging.WARNING)
        else:
            enable_default_handler()
            set_verbosity(logging.INFO)

    def _initialize_optimizer(self):
        sampler_kwargs = {'n_startup_trials': self.n_initial_points, 'seed': self.random_state}
        valid_kwargs = {}
        for key in ['consider_prior', 'prior_weight', 'multivariate']:
            if key in self.kwargs:
                valid_kwargs[key] = self.kwargs[key]
        if valid_kwargs:
            sampler_kwargs.update(valid_kwargs)
        self.sampler = TPESampler(**sampler_kwargs)
        direction = 'minimize' if self.minimize else 'maximize'
        self.study = optuna.create_study(sampler=self.sampler, direction=direction)

    def _objective(self, trial: Trial) -> float:
        params = []
        for i, (low, high) in enumerate(self.bounds):
            param_name = f'x{i}'
            param_value = trial.suggest_float(param_name, low, high)
            params.append(param_value)
        value = self.objective_function(params)
        self.history['params'].append(params.copy())
        self.history['values'].append(value)
        return value

    def optimize(self) -> Dict[str, Any]:
        self.study.optimize(self._objective, n_trials=self.n_initial_points + self.n_iter * self.n_points_per_generation, show_progress_bar=self.show_progress_bar)
        best_trial = self.study.best_trial
        best_params = [best_trial.params[f'x{i}'] for i in range(len(self.bounds))]
        best_value = best_trial.value
        current_best = float('inf') if self.minimize else float('-inf')
        current_best_params = None
        self.history['best_values'] = []
        self.history['best_params'] = []
        for g in range(self.n_iter):
            end_idx = self.n_initial_points + (g + 1) * self.n_points_per_generation
            current_values = self.history['values'][:end_idx]
            current_params = self.history['params'][:end_idx]
            if self.minimize:
                idx = np.argmin(current_values)
                val = current_values[idx]
            else:
                idx = np.argmax(current_values)
                val = current_values[idx]
            self.history['best_values'].append(val)
            self.history['best_params'].append(current_params[idx].copy())
        results = {'best_params': best_params, 'best_value': best_value, 'history': self.history, 'study': self.study}
        return results

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        best_trial = self.study.best_trial
        best_params = [best_trial.params[f'x{i}'] for i in range(len(self.bounds))]
        best_value = best_trial.value
        return (best_params, best_value)

class EvolutionaryStrategyWithCMA:

    def __init__(self, n_iter: int, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, population_size: Optional[int]=None, initial_sigma: Optional[float]=None, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.n_iter = n_iter
        self.bounds = bounds
        self.objective_function = objective_function
        self.minimize = minimize
        self.population_size = population_size
        self.initial_sigma = initial_sigma
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        self.dim = len(bounds)
        if random_state is not None:
            np.random.seed(random_state)
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])
        self.initial_mean = (self.lower_bounds + self.upper_bounds) / 2
        if initial_sigma is None:
            self.initial_sigma = np.mean(self.upper_bounds - self.lower_bounds) / 3
        else:
            self.initial_sigma = initial_sigma
        if population_size is None:
            self.population_size = 4 + int(3 * np.log(self.dim))
        else:
            self.population_size = population_size
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': [], 'sigma': [], 'mean': []}
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_params = None

    def _initialize_cma_es(self):
        self.mean = self.initial_mean.copy()
        self.sigma = self.initial_sigma
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.mu = self.population_size // 2
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.B, self.D = (None, None)
        self._update_eigen_decomposition()
        self.iteration = 0
        if self.verbose:
            print(f'CMA-ES初始化: 维度={self.dim}, 种群大小={self.population_size}, 初始步长={self.sigma:.4f}')

    def _update_eigen_decomposition(self):
        self.C = (self.C + self.C.T) / 2
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.abs(self.D))

    def _sample_population(self):
        z = np.random.randn(self.population_size, self.dim)
        y = np.dot(z, self.B * self.D)
        population = self.mean + self.sigma * y
        for i in range(self.population_size):
            for j in range(self.dim):
                if population[i, j] < self.lower_bounds[j]:
                    population[i, j] = 2 * self.lower_bounds[j] - population[i, j]
                elif population[i, j] > self.upper_bounds[j]:
                    population[i, j] = 2 * self.upper_bounds[j] - population[i, j]
        return (population, y)

    def _update_distribution(self, population, y, fitness):
        sorted_indices = np.argsort(fitness)
        if not self.minimize:
            sorted_indices = sorted_indices[::-1]
        parents = population[sorted_indices[:self.mu]]
        y_parents = y[sorted_indices[:self.mu]]
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights.reshape(-1, 1) * parents, axis=0)
        yw = np.sum(self.weights.reshape(-1, 1) * y_parents, axis=0)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.iteration + 1))) < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.mean - old_mean) / self.sigma
        artmp = 1 / self.sigma * (parents - old_mean)
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * np.dot(artmp.T, self.weights.reshape(-1, 1) * artmp)
        self.sigma = self.sigma * np.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / np.linalg.norm(np.random.randn(self.dim)) - 1))
        if (self.iteration + 1) % (1 / (self.c1 + self.cmu) / self.dim / 10) == 0:
            self._update_eigen_decomposition()

    def optimize(self) -> Dict[str, Any]:
        self._initialize_cma_es()
        if self.verbose:
            print(f'开始CMA-ES优化，总迭代次数: {self.n_iter}')
        for iteration in range(self.n_iter):
            self.iteration = iteration
            population, y = self._sample_population()
            fitness = np.array([self.objective_function(ind.tolist()) for ind in population])
            for i in range(self.population_size):
                self.history['params'].append(population[i].tolist())
                self.history['values'].append(fitness[i])
            current_best_idx = np.argmin(fitness) if self.minimize else np.argmax(fitness)
            current_best_value = fitness[current_best_idx]
            current_best_params = population[current_best_idx].tolist()
            if self.minimize and current_best_value < self.best_value or (not self.minimize and current_best_value > self.best_value):
                self.best_value = current_best_value
                self.best_params = current_best_params
                if self.verbose:
                    print(f'迭代 {iteration + 1}: 发现新的最佳值 {current_best_value:.6f}')
            self.history['best_values'].append(self.best_value)
            self.history['best_params'].append(self.best_params.copy() if self.best_params is not None else [])
            self.history['sigma'].append(self.sigma)
            self.history['mean'].append(self.mean.tolist())
            self._update_distribution(population, y, fitness)
            if self.verbose and (iteration + 1) % max(1, self.n_iter // 10) == 0:
                print(f'已完成 {iteration + 1}/{self.n_iter} 次迭代，当前最佳值: {self.best_value:.6f}, 步长: {self.sigma:.4f}')
        if self.verbose:
            print(f'优化完成，最佳值: {self.best_value:.6f}')
            print(f'最佳参数: {self.best_params}')
        results = {'best_params': self.best_params, 'best_value': self.best_value, 'history': self.history}
        return results

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.best_params, self.best_value)

class GridSearchWithBudget:

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.objective_func = objective_function
        self.dim = len(bounds)
        self.random_state = random_state
        self.bounds = bounds
        self.verbose = verbose
        self.minimize = minimize
        self.kwargs = kwargs
        self.budget = kwargs.get('budget', 1000)
        n_points = int(np.floor(self.budget ** (1 / self.dim)))
        self.n_points_per_dim = max(n_points, 2)
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_params = None
        if random_state is not None:
            np.random.seed(random_state)

    def _generate_grid(self) -> Sequence[Sequence[float]]:
        grids = []
        for low, high in self.bounds:
            dim_grid = np.linspace(low, high, self.n_points_per_dim).tolist()
            grids.append(dim_grid)
        full_grid = product(*grids)
        grid_points = list(islice(full_grid, self.budget))
        return grid_points

    def optimize(self, n_iter=None, n_samples=None):
        if self.verbose:
            print(f'开始带预算的网格搜索，总预算: {self.budget} 次评估')
        grid_points = self._generate_grid()
        if self.verbose:
            print(f'生成网格点数: {len(grid_points)} (不超过预算 {self.budget})')
        for i, params in enumerate(grid_points):
            value = self.objective_func(params)
            params_list = list(params)
            self.history['params'].append(params_list)
            self.history['values'].append(value)
            if self.minimize and value < self.best_value or (not self.minimize and value > self.best_value):
                self.best_value = value
                self.best_params = params_list
                if self.verbose:
                    print(f'评估 {i + 1}/{len(grid_points)}: 新最佳值 {value:.6f}')
            self.history['best_values'].append(self.best_value)
            self.history['best_params'].append(self.best_params.copy() if self.best_params else None)
        if self.verbose:
            print(f'网格搜索完成，最佳值: {self.best_value:.6f}')
        return {'best_params': self.best_params, 'best_value': self.best_value, 'history': self.history}

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.best_params, self.best_value)

class ASHAOptimizer:

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, show_progress_bar: bool=False, **kwargs):
        self.n_initial_points = kwargs.get('n_initial_points', 100)
        self.n_points_per_generation = kwargs.get('n_points_per_generation', 9)
        self.n_iter = kwargs.get('n_iter', 9)
        self.total_budget = self.n_iter * self.n_points_per_generation
        self.bounds = bounds
        self.objective_function = objective_function
        self.minimize = minimize
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.kwargs = kwargs
        self.history = {'best_values': [], 'best_params': []}
        self.global_best_value = float('inf') if self.minimize else float('-inf')
        self.global_best_params = None
        self._configure_logging()
        self._initialize_optimizer()

    def _configure_logging(self):
        if not self.verbose:
            disable_default_handler()
            set_verbosity(logging.WARNING)
        else:
            enable_default_handler()
            set_verbosity(logging.INFO)

    def _initialize_optimizer(self):
        direction = 'minimize' if self.minimize else 'maximize'
        reduction_factor = self.kwargs.get('reduction_factor', 3)
        self.pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=self.n_initial_points, min_early_stopping_rate=self.kwargs.get('min_early_stopping_rate', 0), reduction_factor=reduction_factor)
        self.study = optuna.create_study(direction=direction, pruner=self.pruner, sampler=optuna.samplers.RandomSampler(seed=self.random_state))

    def _objective(self, trial: Trial) -> float:
        params = [trial.suggest_float(f'x{i}', low, high) for i, (low, high) in enumerate(self.bounds)]
        value = self.objective_function(params)
        if self.minimize and value < self.global_best_value or (not self.minimize and value > self.global_best_value):
            self.global_best_value = value
            self.global_best_params = params
        return value

    def optimize(self) -> Dict[str, Any]:
        self.study.optimize(self._objective, n_trials=self.total_budget)
        all_trials = self.study.trials
        self.history['best_values'] = []
        self.history['best_params'] = []
        current_best = float('inf') if self.minimize else float('-inf')
        for i in range(self.n_iter):
            cutoff = (i + 1) * self.n_points_per_generation
            current_trials = [t for t in all_trials[:cutoff] if t.value is not None]
            if current_trials:
                if self.minimize:
                    best_trial_so_far = min(current_trials, key=lambda t: t.value)
                else:
                    best_trial_so_far = max(current_trials, key=lambda t: t.value)
                self.history['best_values'].append(best_trial_so_far.value)
                best_p_list = [best_trial_so_far.params[f'x{j}'] for j in range(len(self.bounds))]
                self.history['best_params'].append(best_p_list)
        return {'best_params': self.global_best_params, 'best_value': self.global_best_value, 'history': self.history}

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        bt = self.study.best_trial
        best_params = [bt.params[f'x{i}'] for i in range(len(self.bounds))]
        return (best_params, bt.value)

class MomentumBasedRandomSearch:

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.objective_func = objective_function
        self.bounds = np.array(bounds, dtype=np.float64)
        self.dim = self.bounds.shape[0]
        self.minimize = minimize
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        self.initial_step_size = kwargs.get('initial_step_size', 0.1)
        self.momentum_factor = kwargs.get('momentum_factor', 0.8)
        self.step_size_decay = kwargs.get('step_size_decay', 0.99)
        self.window_size = kwargs.get('window_size', 5)
        self.n_iter = kwargs.get('n_iter', 100)
        self.n_samples = kwargs.get('n_samples', 10)
        self.current_step_size = self.initial_step_size
        self.momentum = np.zeros(self.dim)
        self.history_performances = []
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._configure_logging()

    def _configure_logging(self):
        if not self.verbose:
            disable_default_handler()
            set_verbosity(logging.WARNING)
        else:
            enable_default_handler()
            set_verbosity(logging.INFO)

    def _update_momentum(self, current_perf):
        self.history_performances.append(current_perf)
        if len(self.history_performances) > self.window_size:
            self.history_performances.pop(0)
        if len(self.history_performances) >= 2:
            perf_mean = np.mean(self.history_performances[:-1])
            perf_delta = current_perf - perf_mean
            self.momentum = self.momentum_factor * self.momentum + (1 - self.momentum_factor) * perf_delta
            if self.verbose:
                get_logger(__name__).info(f'动量更新: {self.momentum}，性能变化: {perf_delta:.4f}')
        else:
            self.momentum = np.zeros(self.dim)

    def _adjust_step_size(self):
        if len(self.history_performances) >= 2:
            prev_perf = self.history_performances[-2]
            curr_perf = self.history_performances[-1]
            if self.minimize and curr_perf < prev_perf or (not self.minimize and curr_perf > prev_perf):
                self.current_step_size *= 1.1
                get_logger(__name__).info(f'性能提升，步长放大至: {self.current_step_size:.4f}')
            else:
                self.current_step_size *= 0.9
                get_logger(__name__).info(f'性能下降，步长缩小至: {self.current_step_size:.4f}')
        self.current_step_size = np.clip(self.current_step_size, 0.001, 0.5)

    def optimize(self) -> Dict[str, Any]:
        best_value = np.inf if self.minimize else -np.inf
        best_params = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=self.dim)
        initial_value = self.objective_func(best_params.tolist())
        self.history_performances.append(initial_value)
        self.history['params'].append(best_params.tolist())
        self.history['values'].append(initial_value)
        best_value = initial_value
        self.history['best_values'].append(best_value)
        self.history['best_params'].append(best_params.tolist())
        if self.verbose:
            get_logger(__name__).info(f'初始参数: {best_params.tolist()}, 初始值: {best_value:.4f}')
        for iter_idx in range(1, self.n_iter + 1):
            for _ in range(self.n_samples):
                momentum_norm = self.momentum / (np.linalg.norm(self.momentum) + 1e-08)
                random_dir = np.random.normal(0, 1, self.dim)
                search_dir = 0.7 * momentum_norm + 0.3 * random_dir
                search_dir = search_dir / (np.linalg.norm(search_dir) + 1e-08)
                new_params = best_params + self.current_step_size * search_dir
                new_params = np.clip(new_params, self.bounds[:, 0], self.bounds[:, 1])
                try:
                    new_value = self.objective_func(new_params.tolist())
                except Exception as e:
                    get_logger(__name__).warning(f'目标函数评估失败: {e}，跳过该参数')
                    continue
                self.history['params'].append(new_params.tolist())
                self.history['values'].append(new_value)
                if self.minimize and new_value < best_value or (not self.minimize and new_value > best_value):
                    best_value = new_value
                    best_params = new_params.copy()
                    if self.verbose:
                        get_logger(__name__).info(f'迭代 {iter_idx} 发现新最优: {best_value:.4f}')
                self._update_momentum(new_value)
                self._adjust_step_size()
            self.history['best_values'].append(best_value)
            self.history['best_params'].append(best_params.tolist())
        if self.verbose:
            get_logger(__name__).info(f'优化完成，最佳值: {best_value:.4f}, 最佳参数: {best_params.tolist()}')
        return {'best_params': best_params.tolist(), 'best_value': best_value, 'history': self.history}

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.history['best_params'][-1], self.history['best_values'][-1])

class AdamHPO:

    def __init__(self, bounds, objective_function, minimize=False, **kwargs):
        self.objective_func = objective_function
        self.bounds = np.array(bounds, dtype=np.float64)
        self.dim = self.bounds.shape[0]
        self.minimize = minimize
        self.budget = kwargs.get('budget', 1000)
        self.evals_per_iter = self.dim + 1
        self.n_iter = max(1, (self.budget - 1) // self.evals_per_iter)
        self.lr = kwargs.get('lr', 0.05)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.eps = kwargs.get('eps', 1e-08)
        self.grad_eps = kwargs.get('grad_eps', 1e-05)
        self.m = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.t = 0
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}

    def optimize(self):
        curr_params = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
        curr_value = self.objective_func(curr_params)
        best_params = curr_params.copy()
        best_value = curr_value
        self._record_history(curr_params, curr_value, best_params, best_value)
        for _ in range(self.n_iter):
            self.t += 1
            grad = np.zeros(self.dim)
            for d in range(self.dim):
                p_temp = curr_params.copy()
                p_temp[d] += self.grad_eps
                p_temp = np.clip(p_temp, self.bounds[:, 0], self.bounds[:, 1])
                val_perturbed = self.objective_func(p_temp)
                diff = val_perturbed - curr_value
                grad[d] = diff / self.grad_eps if self.minimize else -diff / self.grad_eps
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            curr_params = np.clip(curr_params - step, self.bounds[:, 0], self.bounds[:, 1])
            curr_value = self.objective_func(curr_params)
            is_improved = curr_value < best_value if self.minimize else curr_value > best_value
            if is_improved:
                best_value = curr_value
                best_params = curr_params.copy()
            self._record_history(curr_params, curr_value, best_params, best_value)
        target_len = 100
        current_best_vals = self.history['best_values']
        current_best_params = self.history['best_params']
        indices = np.linspace(0, len(current_best_vals) - 1, target_len).astype(int)
        self.history['best_values'] = [current_best_vals[i] for i in indices]
        self.history['best_params'] = [current_best_params[i] for i in indices]
        return {'best_value': best_value, 'best_params': best_params, 'history': self.history}

    def _record_history(self, p, v, bp, bv):
        self.history['params'].append(p.tolist())
        self.history['values'].append(v)
        self.history['best_params'].append(bp.tolist())
        self.history['best_values'].append(bv)

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.history['best_params'][-1], self.history['best_values'][-1])

class EvolutionaryHPO:

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, crossover_rate: float=0.9, mutation_rate: float=0.1, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.population_size = kwargs.get('population_size', 10)
        self.n_iter = kwargs.get('n_iter', 100)
        self.bounds = bounds
        self.objective_function = objective_function
        self.minimize = minimize
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.default_rng(self.random_state)
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self.dim = len(bounds)

    def _initialize_population(self) -> np.ndarray:
        pop = np.zeros((self.population_size, self.dim))
        for d, (low, high) in enumerate(self.bounds):
            pop[:, d] = self.rng.uniform(low, high, size=self.population_size)
        return pop

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        values = np.array([self.objective_function(ind.tolist()) for ind in population])
        for ind, v in zip(population, values):
            self.history['params'].append(ind.tolist())
            self.history['values'].append(v)
        return values

    def _select(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        selected = []
        for _ in range(self.population_size):
            i, j = self.rng.integers(0, self.population_size, size=2)
            if fitness[i] < fitness[j]:
                selected.append(population[i])
            else:
                selected.append(population[j])
        return np.array(selected)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return (parent1.copy(), parent2.copy())
        point = self.rng.integers(1, self.dim)
        c1 = np.concatenate([parent1[:point], parent2[point:]])
        c2 = np.concatenate([parent2[:point], parent1[point:]])
        return (c1, c2)

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        for d in range(self.dim):
            if self.rng.random() < self.mutation_rate:
                low, high = self.bounds[d]
                individual[d] = self.rng.uniform(low, high)
        return individual

    def optimize(self) -> Dict[str, Any]:
        population = self._initialize_population()
        fitness = self._evaluate(population)
        best = fitness.min() if self.minimize else fitness.max()
        best_ind = population[fitness.argmin()] if self.minimize else population[fitness.argmax()]
        current_best = best
        current_best_params = best_ind.copy()
        for gen in range(self.n_iter):
            selected = self._select(population, fitness)
            next_population = []
            for i in range(0, self.population_size, 2):
                p1 = selected[i]
                p2 = selected[(i + 1) % self.population_size]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                next_population.append(c1)
                next_population.append(c2)
            population = np.array(next_population[:self.population_size])
            fitness = self._evaluate(population)
            if self.minimize:
                gen_best = fitness.min()
                if gen_best < current_best:
                    current_best = gen_best
                    current_best_params = population[fitness.argmin()].copy()
            else:
                gen_best = fitness.max()
                if gen_best > current_best:
                    current_best = gen_best
                    current_best_params = population[fitness.argmax()].copy()
            self.history['best_values'].append(current_best)
            self.history['best_params'].append(current_best_params.tolist())
            if self.verbose:
                print(f'Gen {gen + 1}/{self.n_iter} | Best = {current_best}')
        results = {'best_params': current_best_params.tolist(), 'best_value': current_best, 'history': self.history}
        return results

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        if len(self.history['best_values']) == 0:
            raise RuntimeError('Call optimize() first.')
        return (self.history['best_params'][-1], self.history['best_values'][-1])

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

class GWO(object):

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.minimize = minimize
        self.verbose = verbose
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        self.dim = len(self.upper_bound)
        self.pop_size = kwargs.get('pop_size', 10)
        self.max_iter = kwargs.get('max_iter', 100)
        if random_state is not None:
            np.random.seed(random_state)
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.alpha = None
        self.beta = None
        self.delta = None
        self.objective_function = objective_function
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_params = None
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self.update_leaders()
        self.a = 2

    def update_leaders(self):
        fitness = np.asarray([self.objective_function(position.tolist()) for position in self.pop])
        for i in range(self.pop_size):
            self.history['params'].append(self.pop[i].tolist())
            self.history['values'].append(fitness[i])
        sorted_index = np.argsort(fitness)
        if not self.minimize:
            sorted_index = sorted_index[::-1]
        self.pop = self.pop[sorted_index]
        self.alpha = self.pop[0].copy()
        self.beta = self.pop[1].copy()
        self.delta = self.pop[2].copy()
        current_best_value = fitness[sorted_index[0]]
        current_best_params = self.pop[0].tolist()
        if self.minimize and current_best_value < self.best_value or (not self.minimize and current_best_value > self.best_value):
            self.best_value = current_best_value
            self.best_params = current_best_params

    def get_position(self, individual, leader):
        r1 = np.random.rand()
        r2 = np.random.rand()
        A = 2 * self.a * r1 - self.a
        C = 2 * r2
        D = np.abs(C * leader - individual)
        return leader - A * D

    def evolve(self):
        new_pop = np.zeros_like(self.pop)
        for i in range(3, self.pop_size):
            position = self.pop[i]
            X1 = self.get_position(position, self.alpha)
            X2 = self.get_position(position, self.beta)
            X3 = self.get_position(position, self.delta)
            position = (X1 + X2 + X3) / 3
            position = np.clip(position, self.lower_bound, self.upper_bound)
            new_pop[i] = position
        new_pop[0] = self.alpha
        new_pop[1] = self.beta
        new_pop[2] = self.delta
        self.pop = new_pop
        self.a = self.a - 2 / self.max_iter

    def optimize(self) -> Dict[str, Any]:
        if self.verbose:
            print(f'开始灰狼优化，总迭代次数: {self.max_iter}, 种群大小: {self.pop_size}')
        self.history['best_values'].append(self.best_value)
        self.history['best_params'].append(self.best_params.copy() if self.best_params else None)
        for iteration in range(self.max_iter):
            self.evolve()
            self.update_leaders()
            self.history['best_values'].append(self.best_value)
            self.history['best_params'].append(self.best_params.copy() if self.best_params else None)
            if self.verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                print(f'已完成 {iteration + 1}/{self.max_iter} 次迭代，当前最佳值: {self.best_value:.6f}')
        if self.verbose:
            print(f'优化完成，最佳值: {self.best_value:.6f}')
            print(f'最佳参数: {self.best_params}')
        return {'best_params': self.best_params, 'best_value': self.best_value, 'history': self.history}

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.best_params, self.best_value)

class PSO(object):

    def __init__(self, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, **kwargs):
        self.w = kwargs.get('w', 0.8)
        self.c1 = kwargs.get('c1', 2.0)
        self.c2 = kwargs.get('c2', 2.0)
        self.lower_bound = np.array([b[0] for b in bounds])
        self.upper_bound = np.array([b[1] for b in bounds])
        self.dim = len(self.lower_bound)
        self.minimize = minimize
        self.verbose = verbose
        self.pop_size = kwargs.get('pop_size', 10)
        self.max_iter = kwargs.get('max_iter', 100)
        if random_state is not None:
            np.random.seed(random_state)
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.pop = np.round(self.pop, decimals=6)
        self.objective_function = objective_function
        differ = abs(self.upper_bound - self.lower_bound)
        self.v = np.random.uniform(-differ, differ, (self.pop_size, self.dim))
        self.pbest = self.pop.copy()
        self.gbest = self.pop[0]
        self.fitness_best = np.min(np.asarray([self.objective_function(p) for p in self.pop]))
        self.update_best()
        self.history = {'params': [], 'values': [], 'best_values': [], 'best_params': []}
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_params = None
        self._init_history_record()

    def _init_history_record(self):
        init_fitness = np.asarray([self.objective_function(p) for p in self.pop])
        for i in range(self.pop_size):
            self.history['params'].append(self.pop[i].tolist())
            self.history['values'].append(init_fitness[i])
        self._sync_best_result()
        self.history['best_values'].append(self.best_value)
        self.history['best_params'].append(self.best_params)

    def _sync_best_result(self):
        current_gbest_value = self.objective_function(self.gbest)
        if self.minimize:
            if current_gbest_value < self.best_value:
                self.best_value = current_gbest_value
                self.best_params = self.gbest.tolist()
        elif current_gbest_value > self.best_value:
            self.best_value = current_gbest_value
            self.best_params = self.gbest.tolist()

    def update_v(self):
        k_1 = self.c1 * np.random.rand(self.pop_size).reshape(self.pop_size, 1) * (self.pbest - self.pop)
        k_2 = self.c2 * np.random.rand(self.pop_size).reshape(self.pop_size, 1) * (self.gbest - self.pop)
        self.v = self.w * self.v + k_1 + k_2

    def update_p(self):
        self.pop += self.v
        self.pop = np.clip(self.pop, self.lower_bound, self.upper_bound)

    def update_best(self):
        fitness = np.asarray([self.objective_function(p) for p in self.pop])
        pbest_fitness = np.asarray([self.objective_function(p) for p in self.pbest])
        better_indices = fitness < pbest_fitness
        self.pbest[better_indices] = self.pop[better_indices].copy()
        pbest_fitness[better_indices] = fitness[better_indices]
        gbest_fitness = self.objective_function(self.gbest)
        if np.min(pbest_fitness) < gbest_fitness:
            index = np.argmin(pbest_fitness)
            self.gbest = self.pbest[index].copy()

    def evolve(self):
        self.update_v()
        self.update_p()
        self.update_best()

    def optimize(self) -> Dict[str, Any]:
        if self.verbose:
            print(f'开始粒子群优化，总迭代次数: {self.max_iter}, 种群大小: {self.pop_size}')
            print(f'PSO超参：惯性权重w={self.w}, 个体学习因子c1={self.c1}, 全局学习因子c2={self.c2}')
        for iteration in range(self.max_iter):
            self.evolve()
            self._sync_best_result()
            current_fitness = np.asarray([self.objective_function(p) for p in self.pop])
            for i in range(self.pop_size):
                self.history['params'].append(self.pop[i].tolist())
                self.history['values'].append(current_fitness[i])
            self.history['best_values'].append(self.best_value)
            self.history['best_params'].append(self.best_params)
            if self.verbose and (iteration + 1) % max(1, self.max_iter // 10) == 0:
                print(f'已完成 {iteration + 1}/{self.max_iter} 次迭代，当前最佳值: {self.best_value:.6f}')
        if self.verbose:
            print(f'粒子群优化完成，最终最佳值: {self.best_value:.6f}')
            print(f'最终最佳参数: {self.best_params}')
        return {'best_params': self.best_params, 'best_value': self.best_value, 'history': self.history}

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.best_params, self.best_value)
