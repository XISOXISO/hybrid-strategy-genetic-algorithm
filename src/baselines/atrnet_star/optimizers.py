import math
import random
import numpy as np
ALGORITHMS = ['RandomSearch', 'TPE', 'CMA_ES', 'GridSearchWithBudget', 'ASHA', 'EvolutionaryHPO', 'MomentumBasedRandomSearch', 'AdamHPO', 'GWO', 'PSO', 'GA_Basic', 'JSO', 'LSHADE']
ALIASES = {'BayesianOptimization': 'TPE', 'ASHAOptimizer': 'ASHA',
           'GA': 'GA_Basic', 'BasicGA': 'GA_Basic', 'jSO': 'JSO', 'L-SHADE': 'LSHADE'}

def run_optimizer(name, objective, seed, budget, population=5):
    name = ALIASES.get(name, name)
    if name not in ALGORITHMS:
        raise ValueError(f'Unsupported ATRNet-STAR baseline: {name}')
    if name in ('JSO', 'LSHADE'):
        if __package__:
            from .de_optimizers import SuccessHistoryDE
        else:
            from de_optimizers import SuccessHistoryDE
        controller = SuccessHistoryDE(name, objective, seed, budget)
        controller.run()
        return controller
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if name == 'RandomSearch':
        while True:
            objective(rng.integers(0, 3, size=8))
    elif name == 'GridSearchWithBudget':
        order = np.random.default_rng(271828).permutation(6561)
        for cid in order:
            objective(np.array(np.unravel_index(int(cid), (3,) * 8)))
    elif name in ('TPE', 'ASHA'):
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10) if name == 'TPE' else optuna.samplers.RandomSampler(seed=seed)
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3) if name == 'ASHA' else optuna.pruners.NopPruner()
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        while True:
            trial = study.ask()
            x = [trial.suggest_int('g' + str(i), 0, 2) for i in range(8)]
            effective, _, changed = objective.resolve(np.asarray(x, dtype=float), True)
            original = x if changed else None
            if changed:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.enqueue_trial({'g' + str(i): int(effective[i]) for i in range(8)})
                trial = study.ask()
                x = [trial.suggest_int('g' + str(i), 0, 2) for i in range(8)]
            value = objective(x, trial=trial if name == 'ASHA' else None, repair=False, sampler_proposal=original)
            if objective.last_pruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, value)
    elif name == 'CMA_ES':
        import cma
        es = cma.CMAEvolutionStrategy(np.ones(8), 0.8, {'bounds': [0.0, 2.0], 'seed': seed + 1, 'popsize': population, 'verbose': -9, 'verb_disp': 0})
        while True:
            xs = es.ask()
            ys = [objective(x) for x in xs]
            es.tell(xs, ys)
    elif name == 'AdamHPO':
        while True:
            x = rng.integers(0, 3, size=8).astype(float)
            m = np.zeros(8)
            v = np.zeros(8)
            stale = 0
            for step in range(1, 201):
                before = objective.fes
                g = np.rint(x).astype(int)
                y = objective(g, repair=False)
                grad = np.zeros(8)
                for d in range(8):
                    z = g.copy()
                    delta = 1 if g[d] < 2 else -1
                    z[d] += delta
                    grad[d] = (objective(z, repair=False) - y) / delta
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad ** 2
                x = np.clip(x - 0.05 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-08), 0, 2)
                objective(np.rint(x), repair=False)
                stale = stale + 1 if objective.fes == before else 0
                if stale >= 10:
                    break
    else:
        if __package__:
            from . import legacy_baselines as legacy
            from .ga_basic import BasicGeneticAlgorithm
        else:
            import legacy_baselines as legacy
            from ga_basic import BasicGeneticAlgorithm
        common = dict(bounds=[(0.0, 2.0)] * 8, objective_function=objective, minimize=True, random_state=seed, verbose=False)
        generations = max(1, math.ceil(budget / population))
        if name == 'PSO':
            alg = legacy.PSO(**common, pop_size=population, max_iter=generations)
        elif name == 'GWO':
            alg = legacy.GWO(**common, pop_size=population, max_iter=generations)
        elif name == 'EvolutionaryHPO':
            alg = legacy.EvolutionaryHPO(**common, population_size=population, n_iter=generations)
        elif name == 'MomentumBasedRandomSearch':
            alg = legacy.MomentumBasedRandomSearch(**common, n_samples=population, n_iter=generations)
        elif name == 'GA_Basic':
            alg = BasicGeneticAlgorithm(**common, population_size=population, n_iter=generations, use_strategy1=False, use_strategy2=False, use_strategy3=False, use_strategy4=False, use_strategy5=False)
        else:
            raise ValueError(name)
        alg.optimize()
    if objective.fes < objective.target:
        raise RuntimeError(f'{name} stopped at {objective.fes}, target {objective.target}; no random fill-in is allowed')
