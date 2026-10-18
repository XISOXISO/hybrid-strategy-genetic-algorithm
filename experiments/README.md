# Experiments

This directory contains the experiment entry points and support files for CEC2022, MSTAR, and ATRNet-STAR. All files are stored directly in this directory. Task prefixes distinguish support modules with similar purposes.

## Entry points

| File | Purpose |
| --- | --- |
| `run.py` | Dispatches an experiment to its task-specific entry point. |
| `run_cec2022.py` | Runs the CEC2022 comparisons and records optimization histories. |
| `run_mstar.py` | Runs the original MSTAR experiment with JSO, L-SHADE, and the HGA variants. |
| `run_atrnet_star.py` | Runs the ATRNet-STAR comparisons and records candidate evaluations. |
| `summarize_cec2022.py` | Aggregates CEC2022 results and generates tables and figures. |
| `report_mstar.py` | Validates and aggregates MSTAR search and test results. |
| `report_atrnet_star.py` | Validates and aggregates ATRNet-STAR search and test results. |
| `pack_mstar_results.py` | Packages MSTAR experiment results. |
| `pack_atrnet_star_results.py` | Packages ATRNet-STAR experiment results. |
| `atrnet_star_monitor.py` | Displays ATRNet-STAR progress from recorded status files. |

## Support files

| File or group | Purpose |
| --- | --- |
| `experiment_paths.py` | Resolves repository locations and inventories algorithm source files. |
| `cec2022_algorithm_registry.py` | Connects the CEC2022 entry point to the HGA and baseline factories. |
| `cec2022_cec_evaluator.py` | Provides the CEC2022 objective evaluators and support-data verification. |
| `cec2022_*.npz` | Stores the archived support data required by the CEC2022 evaluator. |
| `mstar_cnn_core.py`, `atrnet_star_cnn_core.py` | Implements model construction, image processing, training, and validation for each recognition task. |
| `mstar_dataset.py`, `atrnet_star_dataset.py`, `atrnet_star_atr_image.py` | Loads and checks the corresponding fixed dataset manifests and image files. |
| `mstar_backend.py`, `atrnet_star_backend.py` | Connects optimizer candidates to model evaluation and checkpoint selection. |
| `mstar_common.py`, `atrnet_star_common.py` | Provides task-specific candidate decoding, file operations, and source fingerprints. |
| `*_resilient_scheduler.py`, `*_dashboard.py` | Schedules independent runs and displays progress. |
| `*_final_evaluation.py` | Evaluates selected models on the isolated test partitions and exports test results. |
| `mstar_candidate_engine.py`, `mstar_tensor_cache.py`, `mstar_training_prefetch.py` | Supports candidate evaluation, exact preprocessing caches, and training-batch transfer. |
| `mstar_run_audit.py`, `mstar_checkpoint_archive.py`, `mstar_paper_outputs.py` | Records resource use, archives selected checkpoints, and generates experiment summaries. |
| `mstar_equivalence.py`, `mstar_v1_*` | Retains the relocated reference implementation and its integrity checks for training-equivalence verification. |
| `mstar_migration_v1.py`, `atrnet_star_migration.py`, `atrnet_star_v2_approved_sources.json` | Checks historical result identities before any supported migration. |
| `mstar_prepare_weights.py` | Downloads and verifies the pretrained backbone weights. |
| `mstar_shard_results.py` | Exports selected MSTAR run records. |
| `*_suite.json` | Identifies the algorithms belonging to each recognition experiment. |
| `*_frozen_dataset_anchor.json` | Identifies the fixed dataset expected by each recognition experiment. |
| `SOURCE_MAP.json` | Records the source locations and original fingerprints of the reorganized files. |
| `requirements.txt` | Lists the Python dependencies. |
| `VALIDATION.json` | Records the completed checks and their execution scope. |

## Repository integration

The active algorithm implementations are imported from the existing `src/HGA/` and `src/baselines/` directories. Keep the previously supplied algorithm packages in their corresponding task directories. The dataset manifests and preparation scripts belong in `datasets/mstar/` and `datasets/atrnet_star/`.

The recognition experiments preserve separate training, validation, and test roles. The relocated MSTAR reference files are used for verification rather than as the active optimizer implementations. Use new result directories for the reorganized code; existing code-identity and integrity checks remain enabled.
