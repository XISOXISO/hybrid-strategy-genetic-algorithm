# MSTAR Baselines

## Files

| File | Purpose |
| --- | --- |
| `algorithm_factory.py` | Constructs the original model-experiment baselines through `create` and dispatches the journal-based JSO and L-SHADE implementations through `run_optimizer`. |
| `legacy_baselines.py` | Contains the random search, Bayesian optimization, CMA-ES, grid search, ASHA, Adam-inspired HPO, evolutionary HPO, momentum-based random search, GWO, and PSO implementations from the original MSTAR model experiments. |
| `ga_basic.py` | Provides the basic GA implementation with the additional HGA mechanisms disabled by the baseline factory. |
| `de_optimizers.py` | Implements JSO and L-SHADE for discrete candidate evaluation through a journal. |
| `journal.py` | Records candidate evaluations, handles repeated candidates, counts function evaluations, and supports deterministic replay. |
| `common.py` | Provides MSTAR candidate configuration decoding, file operations, checksums, and process locks. |

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
| JSO | `JSO` |
| L-SHADE | `LSHADE` |
