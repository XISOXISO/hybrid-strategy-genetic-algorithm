# Hybrid Strategy Genetic Algorithm (HGA)

This repository provides the HGA implementations and experiment code for CEC2022 numerical optimization, MSTAR military target recognition, and ATRNet-STAR vehicle recognition. It includes HGA variants, baseline algorithms, fixed dataset manifests, data preparation utilities, and experiment entry points.

## Repository structure

| Directory | Purpose |
| --- | --- |
| [src/HGA/](src/HGA/) | HGA implementations for CEC2022, MSTAR, and ATRNet-STAR. |
| [src/baselines/](src/baselines/) | Baseline implementations organized by task. |
| [datasets/mstar/](datasets/mstar/) | MSTAR sample records, fixed partitions, class mapping, and preparation and verification code. |
| [datasets/atrnet_star/](datasets/atrnet_star/) | ATRNet-STAR sample records, fixed partitions, class mapping, imaging-condition summaries, and preparation and verification code. |
| [experiments/](experiments/) | Experiment entry points, model training and evaluation, progress tracking, checkpoint handling, and result aggregation. |

The task directories under `src/HGA/` and `src/baselines/` are named `cec2022/`, `mstar/`, and `atrnet_star/`. All experiment entry points and their support files are stored directly under `experiments/`.

## HGA variants

| Variant | Description |
| --- | --- |
| RM | Reverse mutation assignment: better-ranked individuals receive higher mutation rates. |
| CM | Conventional mutation assignment: worse-ranked individuals receive higher mutation rates. |
| SW50 | Switches from RM to CM during the search. |
| SW60 | Switches from RM to CM using a different switching schedule from SW50. |

## Baseline algorithms

The baseline modules include random search, Bayesian optimization with TPE, CMA-ES, grid search with budget, ASHA, evolutionary HPO, momentum-based random search, Adam-inspired HPO, GWO, PSO, basic GA, JSO, and L-SHADE.

CEC2022 and ATRNet-STAR have comparison suites covering the HGA variants and baseline algorithms. The original MSTAR experiment entry point runs JSO, L-SHADE, and the HGA variants. See the task-specific source files and [experiment documentation](experiments/README.md) for the corresponding implementations.

## Data availability

Dataset images and original XML annotations are obtained from the official providers. This repository supplies the retained-sample manifests, fixed partition records, class mappings, checksums, and preparation scripts. The scripts preserve the recorded memberships and sample order when reconstructing the experiment subsets from matching local files.

### MSTAR

- [Official MSTAR dataset page — AFRL SDMS](https://www.sdms.afrl.af.mil/index.php?collection=mstar)
- [Official download access and account registration](https://www.sdms.afrl.af.mil/index.php?collection=registration)
- [MSTAR subset documentation](datasets/mstar/README.md)

The MSTAR manifests describe the subset and partitions used by this experiment. Reconstruction requires the prepared JPEG files matching the recorded checksums; native Phoenix-to-JPEG conversion is not included.

### ATRNet-STAR

- [Official ATRNet-STAR project](https://github.com/waterdisappear/ATRNet-STAR)
- [Official dataset download and access terms](https://huggingface.co/datasets/waterdisappear/ATRNet-STAR)
- [ATRNet-STAR subset documentation](datasets/atrnet_star/README.md)

The ATRNet-STAR experiment uses a custom ten-class subset of the official SOC-40 release. Its fixed sample lists identify the required ground-range amplitude images and matching XML files. The custom subset is distinct from an official benchmark configuration.

Follow the providers' access procedures and usage terms, and cite the original dataset publications when using their data.

## Experiment entry points

| File | Purpose |
| --- | --- |
| [experiments/run.py](experiments/run.py) | Dispatches a selected task to its experiment entry point. |
| [experiments/run_cec2022.py](experiments/run_cec2022.py) | Runs CEC2022 optimization experiments. |
| [experiments/run_mstar.py](experiments/run_mstar.py) | Runs the original MSTAR recognition experiments. |
| [experiments/run_atrnet_star.py](experiments/run_atrnet_star.py) | Runs ATRNet-STAR recognition experiments. |

The recognition experiments use ResNet18 for candidate evaluation and keep training, validation, and test roles separate. Selected models are evaluated on the held-out test partitions. The experiment modules import algorithms from `src/HGA/` and `src/baselines/`.

Python dependencies are listed in [experiments/requirements.txt](experiments/requirements.txt). See the [experiment documentation](experiments/README.md) for support-file descriptions and [validation record](experiments/VALIDATION.json) for the scope of the completed checks.
