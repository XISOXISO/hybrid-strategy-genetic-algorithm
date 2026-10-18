import numpy as np
from typing import Sequence, Tuple, Callable, Optional, Dict, Any
from optuna import logging
from optuna.logging import disable_default_handler, set_verbosity, enable_default_handler
from optuna.logging import get_logger

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
