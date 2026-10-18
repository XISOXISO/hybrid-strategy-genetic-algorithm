# CEC2022 Baselines

## Files

| File | Purpose |
| --- | --- |
| `algorithm_factory.py` | Provides a common construction interface and resolves baseline identifiers and aliases. |
| `legacy_baselines.py` | Contains the random search, Bayesian optimization, CMA-ES, grid search, ASHA, evolutionary HPO, momentum-based random search, GWO, and PSO implementations. |
| `ga_basic.py` | Provides the basic GA implementation with the additional HGA mechanisms disabled by the baseline factory. |
| `de_baselines.py` | Contains the Python implementations of L-SHADE and JSO. |
| `strict_adam.py` | Provides the Adam-inspired HPO implementation and its evaluation-budget accounting. |

## Algorithms

| Name | Identifier |
| --- | --- |
| Random search | `RandomSearch` |
| Bayesian optimization with TPE | `BayesianOptimization` |
| CMA-ES | `CMA_ES` |
| Grid search with budget | `GridSearchWithBudget` |
| ASHA | `ASHAOptimizer` |
| Adam-inspired HPO | `AdamHPO` |
| Evolutionary HPO | `EvolutionaryHPO` |
| Momentum-based random search | `MomentumBasedRandomSearch` |
| Grey wolf optimizer | `GWO` |
| Particle swarm optimization | `PSO` |
| Basic genetic algorithm | `GA_Basic` |
| L-SHADE | `LSHADE` |
| JSO | `JSO` |
