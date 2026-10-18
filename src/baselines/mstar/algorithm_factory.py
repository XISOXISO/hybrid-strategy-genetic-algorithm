from __future__ import annotations

if __package__:
    from . import legacy_baselines as legacy
    from .ga_basic import BasicGeneticAlgorithm
    from .de_optimizers import SuccessHistoryDE
else:
    import legacy_baselines as legacy
    from ga_basic import BasicGeneticAlgorithm
    from de_optimizers import SuccessHistoryDE

LEGACY_ALGORITHMS = (
    'RandomSearch', 'BayesianOptimization', 'CMA_ES', 'GridSearchWithBudget',
    'ASHAOptimizer', 'AdamHPO', 'EvolutionaryHPO', 'MomentumBasedRandomSearch',
    'GWO', 'PSO', 'GA_Basic',
)
JOURNAL_ALGORITHMS = ('JSO', 'LSHADE')
ALGORITHMS = LEGACY_ALGORITHMS + JOURNAL_ALGORITHMS
ALIASES = {
    'TPE': 'BayesianOptimization',
    'ASHA': 'ASHAOptimizer',
    'GA': 'GA_Basic',
    'BasicGA': 'GA_Basic',
    'jSO': 'JSO',
    'L-SHADE': 'LSHADE',
}


def _canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in ALGORITHMS:
        raise ValueError(f'Unsupported MSTAR baseline: {name}')
    return name


def create(name, bounds, objective, seed=None, minimize=False, verbose=False, **kwargs):
    name = _canonical_name(name)
    if name not in LEGACY_ALGORITHMS:
        raise ValueError('JSO and LSHADE use run_optimizer with an evaluation journal')
    classes = {
        'RandomSearch': legacy.RandomSearch,
        'BayesianOptimization': legacy.BayesianOptimizationWithTPE,
        'CMA_ES': legacy.EvolutionaryStrategyWithCMA,
        'GridSearchWithBudget': legacy.GridSearchWithBudget,
        'ASHAOptimizer': legacy.ASHAOptimizer,
        'AdamHPO': legacy.AdamHPO,
        'EvolutionaryHPO': legacy.EvolutionaryHPO,
        'MomentumBasedRandomSearch': legacy.MomentumBasedRandomSearch,
        'GWO': legacy.GWO,
        'PSO': legacy.PSO,
        'GA_Basic': BasicGeneticAlgorithm,
    }
    if name == 'GA_Basic':
        kwargs.update({f'use_strategy{i}': False for i in range(1, 6)})
    return classes[name](
        bounds=bounds,
        objective_function=objective,
        random_state=seed,
        minimize=minimize,
        verbose=verbose,
        **kwargs,
    )


def run_optimizer(name, journal, seed, budget, population=10):
    name = _canonical_name(name)
    if name not in JOURNAL_ALGORITHMS:
        raise ValueError('This baseline uses create with the model experiment objective')
    engine = SuccessHistoryDE(name, journal, seed, budget)
    engine.run()
    return engine
