from __future__ import annotations
import math
import numpy as np
if __package__:
    from .legacy_hga import HybridGeneticAlgorithm
else:
    from legacy_hga import HybridGeneticAlgorithm

FACTORS = ('cr_max', 'mr_min', 'T', 'guided_ratio')
LEVELS = {'cr_max': [.8, .6, 1.0], 'mr_min': [.05, .01, .10],
          'T': [10, 5, 15], 'guided_ratio': [.4, .2, .6]}
ROWS = [(0,0,0,0), (0,1,1,1), (0,2,2,2), (1,0,1,2), (1,1,2,0),
        (1,2,0,1), (2,0,2,1), (2,1,0,2), (2,2,1,0)]
COMBINATIONS = {f'L{i+1:02d}': {k: LEVELS[k][v] for k,v in zip(FACTORS,row)}
                for i,row in enumerate(ROWS)}

class SensitivityHGA(HybridGeneticAlgorithm):
    def __init__(self, *args, guided_ratio=.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.guided_ratio = guided_ratio
        self.reconstruction_count = 0

    def _strategy3_adaptive_rates(self, fitness, mr_min, cr_max, diversity_ratio):
        avg, best = np.mean(fitness), np.min(fitness)
        bias = self.strategy3_params['bias']
        baseline = mr_min * (1.5 - .5 * diversity_ratio)
        r = np.clip((avg - fitness + bias) / (avg - best + bias), 0., 1.)
        better = fitness < avg
        mutation = np.where(better, baseline * np.exp(r), baseline)
        crossover = np.where(better, cr_max * (1-r), cr_max)
        return np.clip(mutation, mr_min, .30), np.clip(crossover, 0., cr_max)

    def _evaluate_population(self, population):
        values = super()._evaluate_population(population)

        if self.best_individual is not None:
            for subpop, vals in zip(population, values):
                worst = int(np.argmax(vals))
                if self.best_value < vals[worst]:
                    subpop[worst] = self.best_individual.copy()
                    vals[worst] = self.best_value
        return values

    def _strategy4_maintenance(self, population, objectives, best_individual, best_value):
        if best_value < self.history_best_value:
            self.history_best_value = best_value
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        if self.stagnation_counter < self.strategy4_params['T']:
            return population, False
        self.stagnation_counter = 0
        self.reconstruction_count += 1
        result = []
        for subpop in population:
            n = len(subpop)

            guided = min(n-1, int(math.floor((n-1)*self.guided_ratio + .5)))
            generated = [best_individual.copy()]
            for _ in range(guided):
                step = self.levy_flight(self.dim) * (self.upper_bounds-self.lower_bounds)*.01
                generated.append(np.clip(best_individual+step, self.lower_bounds, self.upper_bounds))
            while len(generated) < n:
                generated.append(np.random.uniform(self.lower_bounds, self.upper_bounds))
            result.append(np.asarray(generated))
        return result, True

def configuration(name, dim, budget, parameters):
    return {'population_size': dim, 'n_iter': math.ceil(budget/dim)+1,
            'use_strategy1': False, 'use_strategy2': True, 'use_strategy3': True,
            'use_strategy4': True, 'use_strategy5': False,
            'strategy3_params': {'mr_min': parameters['mr_min'],
                                 'cr_max': parameters['cr_max'], 'bias': 1e-8},
            'strategy4_params': {'T': parameters['T'], 'bias': 1e-8},
            'guided_ratio': parameters['guided_ratio']}

def create(name, dim, budget, seed, objective, parameters):
    return SensitivityHGA(bounds=[(-100.,100.)]*dim, objective_function=objective,
                          minimize=True, random_state=seed, verbose=False,
                          **configuration(name,dim,budget,parameters))
