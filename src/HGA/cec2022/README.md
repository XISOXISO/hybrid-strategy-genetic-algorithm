# HGA for CEC2022

## Files

| File | Purpose |
| --- | --- |
| `legacy_hga.py` | Defines the base `HybridGeneticAlgorithm` class and its strategy switches. |
| `algorithm_factory.py` | Creates the RM, CM, SW50, and SW60 variants using the base class. |
| `sensitivity_model.py` | Defines `SensitivityHGA` and the nine L9 parameter combinations, `L01`–`L09`. |

## HGA variants

| Name | Identifier | Difference |
| --- | --- | --- |
| RM | `RM_HGA_T10` | Uses the base HGA mutation-rate assignment. |
| CM | `CM_HGA_T10` | Reverses the assignment direction while retaining the base mutation-rate multiset. |
| SW50 | `SW50_HGA_T10` | Uses RM before 50% of the evaluation budget, then CM. |
| SW60 | `SW60_HGA_T10` | Uses RM before 60% of the evaluation budget, then CM. |

