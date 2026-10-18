# ATRNet-STAR Baselines

## Files

| File | Purpose |
| --- | --- |
| `optimizers.py` | Provides the common `run_optimizer` entry point and the random search, TPE, CMA-ES, grid search, ASHA, and discrete Adam-inspired HPO execution logic. |
| `legacy_baselines.py` | Contains the evolutionary HPO, momentum-based random search, GWO, and PSO implementations used by the experiment runner. |
| `ga_basic.py` | Provides the basic GA implementation with the additional HGA mechanisms disabled by the experiment runner. |
| `de_optimizers.py` | Implements JSO and L-SHADE for discrete candidate evaluation through a journal. |
| `journal.py` | Records candidate evaluations, handles repeated candidates, counts function evaluations, and supports deterministic replay. |
| `common.py` | Provides ATRNet-STAR candidate configuration decoding, file operations, checksums, and process locks. |

## Algorithms

| Name | Identifier |
| --- | --- |
| Random search | `RandomSearch` |
| Bayesian optimization with TPE | `TPE` |
| CMA-ES | `CMA_ES` |
| Grid search with budget | `GridSearchWithBudget` |
| ASHA | `ASHA` |
| Adam-inspired HPO | `AdamHPO` |
| Evolutionary HPO | `EvolutionaryHPO` |
| Momentum-based random search | `MomentumBasedRandomSearch` |
| Grey wolf optimizer | `GWO` |
| Particle swarm optimization | `PSO` |
| Basic genetic algorithm | `GA_Basic` |
| JSO | `JSO` |
| L-SHADE | `LSHADE` |
