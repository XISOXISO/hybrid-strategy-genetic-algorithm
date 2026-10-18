from __future__ import annotations
import math
import numpy as np
if __package__:
    from . import legacy_baselines as legacy
    from .ga_basic import BasicGeneticAlgorithm
    from .de_baselines import DifferentialEvolution
    from .strict_adam import StrictBudgetAdamHPO
else:
    import legacy_baselines as legacy
    from ga_basic import BasicGeneticAlgorithm
    from de_baselines import DifferentialEvolution
    from strict_adam import StrictBudgetAdamHPO

def _canonical_name(name):
    canonical = ALIASES.get(name, name)
    if canonical not in ALGORITHMS:
        raise ValueError(f'Unsupported baseline: {name}')
    return canonical
ALGORITHMS = ['RandomSearch', 'BayesianOptimization', 'CMA_ES', 'GridSearchWithBudget', 'ASHAOptimizer', 'AdamHPO', 'EvolutionaryHPO', 'MomentumBasedRandomSearch', 'GWO', 'PSO', 'GA_Basic', 'LSHADE', 'JSO']
ALIASES = {'ASHA': 'ASHAOptimizer', 'TPE': 'BayesianOptimization', 'jSO': 'JSO', 'L-SHADE': 'LSHADE', 'jSO_src2017_v4': 'JSO', 'LSHADE_src2014_v4': 'LSHADE'}

class BoundedLegacyCMA(legacy.EvolutionaryStrategyWithCMA):

    def _sample_population(self):
        population, y = super()._sample_population()
        outside = (population < self.lower_bounds) | (population > self.upper_bounds)
        if np.any(outside):
            width = self.upper_bounds - self.lower_bounds
            remainder = np.mod(population - self.lower_bounds, 2 * width)
            reflected = self.lower_bounds + np.where(remainder <= width, remainder, 2 * width - remainder)
            population = np.where(outside, reflected, population)
        return (population, y)

def configuration(name, dim, budget):
    name = _canonical_name(name)
    generations = max(1, math.ceil(budget / dim))
    if name == 'RandomSearch':
        return {'n_iter': generations, 'budget': budget}
    if name == 'BayesianOptimization':
        initial = min(10, budget)
        return {'n_initial_points': initial, 'n_points_per_generation': 1, 'n_iter': budget - initial, 'show_progress_bar': False}
    if name == 'CMA_ES':
        return {'n_iter': generations, 'population_size': dim}
    if name == 'GridSearchWithBudget':
        return {'budget': budget}
    if name == 'ASHAOptimizer':
        return {'n_initial_points': 1, 'n_points_per_generation': dim, 'n_iter': generations, 'show_progress_bar': False}
    if name == 'AdamHPO':
        return {'budget': budget}
    if name == 'EvolutionaryHPO':
        return {'population_size': dim, 'n_iter': generations}
    if name == 'MomentumBasedRandomSearch':
        return {'n_samples': dim, 'n_iter': generations}
    if name in ('GWO', 'PSO'):
        return {'pop_size': dim, 'max_iter': generations}
    if name == 'GA_Basic':
        enabled = name != 'GA_Basic'
        return {'n_iter': generations, 'population_size': dim, 'use_strategy1': False, 'use_strategy2': enabled, 'use_strategy3': enabled, 'use_strategy4': enabled, 'use_strategy5': False, 'strategy3_params': {'mr_min': 0.05, 'cr_max': 0.8, 'bias': 1e-08}, 'strategy4_params': {'T': 10, 'bias': 1e-08}}
    if name in ('LSHADE', 'JSO'):
        return {'budget': budget, 'variant': 'LSHADE' if name == 'LSHADE' else 'jSO'}
    raise ValueError(name)

def create(name, dim, budget, seed, objective):
    name = _canonical_name(name)
    classes = {'RandomSearch': legacy.RandomSearch, 'BayesianOptimization': legacy.BayesianOptimizationWithTPE, 'CMA_ES': BoundedLegacyCMA, 'GridSearchWithBudget': legacy.GridSearchWithBudget, 'ASHAOptimizer': legacy.ASHAOptimizer, 'AdamHPO': StrictBudgetAdamHPO, 'EvolutionaryHPO': legacy.EvolutionaryHPO, 'MomentumBasedRandomSearch': legacy.MomentumBasedRandomSearch, 'GWO': legacy.GWO, 'PSO': legacy.PSO, 'GA_Basic': BasicGeneticAlgorithm, 'LSHADE': DifferentialEvolution, 'JSO': DifferentialEvolution}
    cls = classes[name]
    algorithm = cls(bounds=[(-100.0, 100.0)] * dim, objective_function=objective, minimize=True, random_state=seed, verbose=False, **configuration(name, dim, budget))
    return algorithm
