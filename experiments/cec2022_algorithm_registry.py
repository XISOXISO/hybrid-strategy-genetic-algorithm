from pathlib import Path
import hashlib

import experiment_paths
from src.HGA.cec2022 import algorithm_factory as hga
from src.baselines.cec2022 import algorithm_factory as baselines


ALGORITHMS = [
    'RandomSearch', 'BayesianOptimization', 'CMA_ES', 'GridSearchWithBudget',
    'ASHAOptimizer', 'AdamHPO', 'EvolutionaryHPO', 'MomentumBasedRandomSearch',
    'GWO', 'PSO', 'GA_Basic', 'RM_HGA_T10', 'CM_HGA_T10',
    'SW50_HGA_T10', 'SW60_HGA_T10', 'LSHADE', 'JSO',
]
ALIASES = {**baselines.ALIASES, **hga.ALIASES}


def canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in ALGORITHMS:
        raise ValueError(f'Unsupported CEC2022 algorithm: {name}')
    return name


def configuration(name, dim, budget):
    name = canonical_name(name)
    factory = hga if name in hga.ALGORITHMS else baselines
    return factory.configuration(name, dim, budget)


def create(name, dim, budget, seed, objective):
    name = canonical_name(name)
    factory = hga if name in hga.ALGORITHMS else baselines
    return factory.create(name, dim, budget, seed, objective)


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = experiment_paths.algorithm_source_paths('cec2022')
    for name in ('run_cec2022.py', 'summarize_cec2022.py',
                 'cec2022_cec_evaluator.py', 'cec2022_algorithm_registry.py', 'experiment_paths.py'):
        paths['experiments/' + name] = root / name
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
