from __future__ import annotations

import math
import numpy as np
if __package__:
    from .legacy_hga import HybridGeneticAlgorithm
else:
    from legacy_hga import HybridGeneticAlgorithm

ALGORITHMS = [
    'RM_HGA_T10', 'CM_HGA_T10', 'SW50_HGA_T10', 'SW60_HGA_T10',
]
ALIASES = {'HGA': 'RM_HGA_T10', 'HGA-RM': 'RM_HGA_T10',
           'HGA-CM': 'CM_HGA_T10', 'HGA-SW50': 'SW50_HGA_T10',
           'HGA-SW60': 'SW60_HGA_T10'}

class HGAVariant(HybridGeneticAlgorithm):
    def _strategy3_adaptive_rates(self, fitness, mr_min, cr_max, diversity_ratio):
        rates, crossover = super()._strategy3_adaptive_rates(
            fitness, mr_min, cr_max, diversity_ratio)
        mode = self.variant_mode
        if mode.startswith('SW'):
            progress = self.objective_function.fe / self.objective_function.budget
            mode = 'RM' if progress < self.switch_fraction else 'CM'
        if mode == 'CM':

            assigned = np.empty_like(rates)
            assigned[np.argsort(fitness, kind='stable')] = np.sort(rates)
            rates = assigned
        return rates, crossover

def _canonical_name(name):
    canonical = ALIASES.get(name, name)
    if canonical not in ALGORITHMS:
        raise ValueError(f'Unsupported HGA variant: {name}')
    return canonical

def configuration(name, dim, budget):
    _canonical_name(name)
    if dim <= 0 or budget <= 0:
        raise ValueError('dim and budget must be positive')
    generations = max(1, math.ceil(budget / dim))
    return {'n_iter': generations, 'population_size': dim,
            'use_strategy1': False, 'use_strategy2': True,
            'use_strategy3': True, 'use_strategy4': True,
            'use_strategy5': False,
            'strategy3_params': {'mr_min': .05, 'cr_max': .8, 'bias': 1e-8},
            'strategy4_params': {'T': 10, 'bias': 1e-8}}

def create(name, dim, budget, seed, objective):
    name = _canonical_name(name)
    algorithm = HGAVariant(bounds=[(-100., 100.)] * dim, objective_function=objective,
                    minimize=True, random_state=seed, verbose=False,
                    **configuration(name, dim, budget))
    algorithm.variant_mode = name.split('_')[0]
    algorithm.switch_fraction = .5 if name.startswith('SW50') else .6
    return algorithm
