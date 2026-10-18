import numpy as np
from typing import Sequence, Tuple, Callable, Optional, Dict, Any
from scipy.special import gamma

class BasicGeneticAlgorithm:

    def __init__(self, n_iter: int, bounds: Sequence[Tuple[float, float]], objective_function: Callable[[Sequence[float]], float], population_size: int=50, minimize: bool=True, random_state: Optional[int]=None, verbose: bool=False, use_strategy1: bool=False, use_strategy2: bool=False, use_strategy3: bool=False, use_strategy4: bool=False, use_strategy5: bool=False, strategy1_bias: float=1e-08, strategy2_bias: float=1e-08, strategy3_params: Optional[Dict[str, float]]=None, strategy4_params: Optional[Dict[str, Any]]=None, strategy5_params: Optional[Dict[str, Any]]=None, **kwargs):
        self.n_iter = n_iter
        self.bounds = bounds
        self.objective_function = objective_function
        self.population_size = population_size
        self.minimize = minimize
        self.random_state = random_state
        self.verbose = verbose
        self.use_strategy1 = use_strategy1
        self.use_strategy2 = use_strategy2
        self.use_strategy3 = use_strategy3
        self.use_strategy4 = use_strategy4
        self.use_strategy5 = use_strategy5
        self.strategy1_bias = strategy1_bias
        self.strategy2_bias = strategy2_bias
        self.mapping_type = kwargs.get('mapping_type', 'logarithmic')
        self.strategy3_params = strategy3_params or {'mr_min': 0.05, 'cr_max': 0.8, 'bias': 1e-08}
        self.strategy4_params = strategy4_params or {'T': 10, 'bias': 1e-08}
        self.strategy5_params = strategy5_params or {'m': 5, 'k': 2, 'n': 10}
        self.dim = len(bounds)
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])
        if random_state is not None:
            np.random.seed(random_state)
        if self.use_strategy5:
            self.m = self.strategy5_params['m']
            self.k = self.strategy5_params['k']
            self.n = self.strategy5_params['n']
            self.subpop_size = self.population_size // self.m
            self.population_size = self.subpop_size * self.m
        else:
            self.m = 1
            self.subpop_size = self.population_size
        if self.use_strategy4:
            self.stagnation_counter = 0
            self.history_best_value = float('inf') if minimize else float('-inf')
        self.history = {'best_params': [], 'best_values': [], 'mean_values': [], 'worst_values': [], 'mutation_rates': [], 'crossover_rates': [], 'population_diversity': []}
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_individual = None
        self._validate_parameters()

    def _validate_parameters(self):
        if self.population_size < 2:
            raise ValueError('种群大小必须至少为2')
        if self.use_strategy5 and self.population_size < self.m * 2:
            raise ValueError('使用多种群策略时，种群大小必须至少是种群数量的2倍')

    def _initialize_population(self):
        if self.use_strategy5:
            populations = []
            for _ in range(self.m):
                subpop = np.zeros((self.subpop_size, self.dim))
                for i in range(self.subpop_size):
                    for j in range(self.dim):
                        low, high = self.bounds[j]
                        subpop[i, j] = np.random.uniform(low, high)
                populations.append(subpop)
            return populations
        else:
            population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                for j in range(self.dim):
                    low, high = self.bounds[j]
                    population[i, j] = np.random.uniform(low, high)
            return [population]

    def _evaluate_population(self, population):
        if self.use_strategy5:
            objectives = []
            for subpop in population:
                sub_obj = np.zeros(len(subpop))
                for i, individual in enumerate(subpop):
                    sub_obj[i] = self.objective_function(individual.tolist())
                objectives.append(sub_obj)
            return objectives
        else:
            objectives = np.zeros(len(population[0]))
            for i, individual in enumerate(population[0]):
                objectives[i] = self.objective_function(individual.tolist())
            return [objectives]

    def _strategy1_fitness_transform(self, objectives):
        if not self.use_strategy1:
            return objectives if self.minimize else -objectives
        o_min = np.min(objectives)
        o_max = np.max(objectives)
        denominator = o_max - o_min + self.strategy1_bias
        if denominator <= 0:
            denominator = self.strategy1_bias
        normalized = (objectives - o_min + self.strategy1_bias) / denominator
        if self.mapping_type == 'logarithmic':
            fitness = np.log2(1 + normalized)
        elif self.mapping_type == 'exponential':
            fitness = np.exp(normalized)
        else:
            fitness = normalized
        if not self.minimize:
            fitness = -fitness
        return fitness

    def _strategy2_roulette_probability(self, fitness):
        if not self.use_strategy2:
            min_fitness = np.min(fitness)
            if min_fitness < 0:
                adjusted_fitness = fitness - min_fitness + 1e-08
            else:
                adjusted_fitness = fitness + 1e-08
            return adjusted_fitness / np.sum(adjusted_fitness)
        f_max = np.max(fitness)
        numerator = f_max - fitness + self.strategy2_bias
        denominator = np.sum(numerator)
        if denominator <= 0:
            return np.ones_like(fitness) / len(fitness)
        return numerator / denominator

    def _strategy3_adaptive_rates(self, fitness, mr_min, cr_max, diversity_ratio):
        if not self.use_strategy3:
            return (np.full_like(fitness, mr_min), np.full_like(fitness, cr_max))
        f_avg = np.mean(fitness)
        f_min = np.min(fitness)
        mutation_rates = np.zeros_like(fitness)
        crossover_rates = np.zeros_like(fitness)
        bias = self.strategy3_params['bias']
        adaptive_mr_min = mr_min * (1.5 - 0.5 * diversity_ratio)
        for i, f_val in enumerate(fitness):
            if f_avg > f_val:
                r = (f_avg - f_val + bias) / (f_avg - f_min + bias)
                mutation_rates[i] = adaptive_mr_min * np.exp(r)
                crossover_rates[i] = cr_max * (1 - r)
            else:
                mutation_rates[i] = adaptive_mr_min
                crossover_rates[i] = cr_max
        mutation_rates = np.clip(mutation_rates, 0.05, 0.3)
        crossover_rates = np.clip(crossover_rates, 0, 0.8)
        return (mutation_rates, crossover_rates)

    def levy_flight(self, dim):
        beta = 1.5
        sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def _strategy4_maintenance(self, population, objectives, best_individual, best_value):
        if not self.use_strategy4:
            return (population, False)
        if self.minimize and best_value <= self.history_best_value or (not self.minimize and best_value >= self.history_best_value):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.history_best_value = best_value
        if self.stagnation_counter >= self.strategy4_params['T']:
            if self.verbose:
                print(f'执行维护策略，停滞代数: {self.stagnation_counter}')
            self.stagnation_counter = 0
            new_population = []
            for subpop in population:
                new_subpop = [best_individual.copy()]
                for _ in range(int(len(subpop) * 0.4)):
                    step = self.levy_flight(self.dim) * (self.upper_bounds - self.lower_bounds) * 0.01
                    new_ind = np.clip(best_individual + step, self.lower_bounds, self.upper_bounds)
                    new_subpop.append(new_ind)
                while len(new_subpop) < len(subpop):
                    random_ind = np.random.uniform(self.lower_bounds, self.upper_bounds)
                    new_subpop.append(random_ind)
                new_population.append(np.array(new_subpop))
            return (new_population, True)
        return (population, False)

    def _strategy5_population_exchange(self, populations, objectives, generation):
        if not self.use_strategy5 or generation % self.n != 0:
            return populations
        if self.verbose:
            print(f'执行多种群交换策略，代数: {generation}')
        elite_individuals = []
        elite_objectives = []
        for i, (subpop, obj) in enumerate(zip(populations, objectives)):
            if self.minimize:
                indices = np.argsort(obj)[:self.k]
            else:
                indices = np.argsort(obj)[-self.k:]
            elite_individuals.extend(subpop[indices])
            elite_objectives.extend(obj[indices])
        elite_population = np.array(elite_individuals[:self.subpop_size])
        if len(elite_population) < self.subpop_size:
            additional = self.subpop_size - len(elite_population)
            for _ in range(additional):
                individual = np.zeros(self.dim)
                for j in range(self.dim):
                    low, high = self.bounds[j]
                    individual[j] = np.random.uniform(low, high)
                elite_population = np.vstack([elite_population, individual])
        populations[0] = elite_population
        return populations

    def _crossover(self, parent1, parent2, crossover_rate):
        child = parent1.copy()
        for i in range(self.dim):
            if np.random.rand() < crossover_rate:
                child[i] = parent2[i]
        return child

    def _mutation(self, individual, mutation_rate):
        mutated = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < mutation_rate:
                scale = (self.upper_bounds[i] - self.lower_bounds[i]) * 0.1
                mutation = np.random.normal(0, scale)
                mutated[i] += mutation
                mutated[i] = np.clip(mutated[i], self.lower_bounds[i], self.upper_bounds[i])
        return mutated

    def _calculate_diversity(self, population):
        if self.use_strategy5:
            all_individuals = np.vstack(population)
            diversity = 0
            count = 0
            n_total = len(all_individuals)
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    diversity += np.linalg.norm(all_individuals[i] - all_individuals[j])
                    count += 1
            return diversity / count if count > 0 else 0
        else:
            diversity = 0
            count = 0
            n_individuals = len(population[0])
            for i in range(n_individuals):
                for j in range(i + 1, n_individuals):
                    diversity += np.linalg.norm(population[0][i] - population[0][j])
                    count += 1
            return diversity / count if count > 0 else 0

    def optimize(self) -> Dict[str, Any]:
        populations = self._initialize_population()
        if self.verbose:
            strategy_status = [f"策略{i + 1}: {('启用' if getattr(self, f'use_strategy{i + 1}') else '禁用')}" for i in range(5)]
            print(f"开始混合遗传算法优化，配置: {', '.join(strategy_status)}")
            if self.use_strategy5:
                print(f'多种群配置: {self.m}个种群，每个种群{self.subpop_size}个个体')
        for generation in range(self.n_iter):
            all_objectives = self._evaluate_population(populations)
            if self.use_strategy5:
                global_best_value = float('inf') if self.minimize else float('-inf')
                global_best_individual = None
                for subpop, objectives in zip(populations, all_objectives):
                    if self.minimize:
                        best_idx = np.argmin(objectives)
                        best_value = objectives[best_idx]
                    else:
                        best_idx = np.argmax(objectives)
                        best_value = objectives[best_idx]
                    if self.minimize and best_value < global_best_value or (not self.minimize and best_value > global_best_value):
                        global_best_value = best_value
                        global_best_individual = subpop[best_idx].copy()
            else:
                objectives = all_objectives[0]
                if self.minimize:
                    best_idx = np.argmin(objectives)
                else:
                    best_idx = np.argmax(objectives)
                global_best_value = objectives[best_idx]
                global_best_individual = populations[0][best_idx].copy()
            if self.minimize and global_best_value < self.best_value or (not self.minimize and global_best_value > self.best_value):
                self.best_value = global_best_value
                self.best_individual = global_best_individual.copy()
                if self.verbose:
                    print(f'代数 {generation + 1}: 发现新的最佳值 {global_best_value:.6f}')
            self.history['best_params'].append(self.best_individual.copy())
            self.history['best_values'].append(self.best_value)
            if self.use_strategy5:
                mean_value = np.mean([np.mean(obj) for obj in all_objectives])
                worst_value = np.max([np.max(obj) for obj in all_objectives]) if self.minimize else np.min([np.min(obj) for obj in all_objectives])
            else:
                mean_value = np.mean(all_objectives[0])
                worst_value = np.max(all_objectives[0]) if self.minimize else np.min(all_objectives[0])
            self.history['mean_values'].append(mean_value)
            self.history['worst_values'].append(worst_value)
            self.history['population_diversity'].append(self._calculate_diversity(populations))
            populations, maintenance_performed = self._strategy4_maintenance(populations, all_objectives, self.best_individual, self.best_value)
            if maintenance_performed:
                all_objectives = self._evaluate_population(populations)
            populations = self._strategy5_population_exchange(populations, all_objectives, generation)
            new_populations = []
            avg_mutation_rates = []
            avg_crossover_rates = []
            for i, (subpop, objectives) in enumerate(zip(populations, all_objectives)):
                fitness = self._strategy1_fitness_transform(objectives)
                selection_probs = self._strategy2_roulette_probability(fitness)
                current_diversity = self._calculate_diversity(populations)
                if generation == 0:
                    self.initial_diversity = current_diversity if current_diversity > 0 else 1.0
                diversity_ratio = np.clip(current_diversity / self.initial_diversity, 0.1, 1.0)
                mr_min = self.strategy3_params['mr_min']
                cr_max = self.strategy3_params['cr_max']
                mutation_rates, crossover_rates = self._strategy3_adaptive_rates(fitness, mr_min, cr_max, diversity_ratio)
                avg_mutation_rates.append(np.mean(mutation_rates))
                avg_crossover_rates.append(np.mean(crossover_rates))
                new_subpop = np.zeros_like(subpop)
                parent_indices = np.random.choice(len(subpop), len(subpop), p=selection_probs)
                selected_parents = subpop[parent_indices]
                for j in range(len(subpop)):
                    if j % 2 == 0 and j + 1 < len(subpop):
                        parent1 = selected_parents[j]
                        parent2 = selected_parents[j + 1]
                        avg_cr = (crossover_rates[j] + crossover_rates[j + 1]) / 2
                        child1 = self._crossover(parent1, parent2, avg_cr)
                        child2 = self._crossover(parent2, parent1, avg_cr)
                        child1 = self._mutation(child1, mutation_rates[j])
                        child2 = self._mutation(child2, mutation_rates[j + 1])
                        new_subpop[j] = child1
                        new_subpop[j + 1] = child2
                    elif j % 2 == 0:
                        parent = selected_parents[j]
                        new_subpop[j] = self._mutation(parent, mutation_rates[j])
                new_populations.append(new_subpop)
            self.history['mutation_rates'].append(np.mean(avg_mutation_rates))
            self.history['crossover_rates'].append(np.mean(avg_crossover_rates))
            populations = new_populations
            if self.verbose and (generation + 1) % max(1, self.n_iter // 10) == 0:
                print(f'代数 {generation + 1}: 最佳值 {self.best_value:.6f}, 平均适应度 {mean_value:.6f}')
        if self.verbose:
            print(f'优化完成，最佳值: {self.best_value:.6f}')
            print(f'最佳个体: {self.best_individual}')
        results = {'best_params': self.best_individual, 'best_value': self.best_value, 'history': self.history}
        return results

    def get_optimization_history(self) -> Dict[str, Any]:
        return self.history

    def get_best_result(self) -> Tuple[Sequence[float], float]:
        return (self.best_individual, self.best_value)
