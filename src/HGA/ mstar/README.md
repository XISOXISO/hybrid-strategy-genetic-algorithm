# HGA for MSTAR

## Files

| File | Purpose |
| --- | --- |
| `hga_optimizer.py` | Implements HGA and its RM, CM, SW50, and SW60 variants. |
| `hga_reuse.py` | Exports and restores optimizer states, validates shared search prefixes, and reuses compatible search histories. |
| `common.py` | Provides configuration decoding, file operations, checksums, and process locks. |
| `journal.py` | Records candidate evaluations, counts function evaluations, and supports deterministic replay for interrupted searches. |
| `shared_resources.py` | Manages shared model weights, reference tracking, and resource locks. |
| `checkpoints.py` | Saves, loads, and validates model checkpoints and their selection identities. |

## HGA variants

| Name | Description |
| --- | --- |
| RM | Uses reverse mutation assignment, assigning higher mutation rates to better-ranked individuals. |
| CM | Uses conventional mutation assignment, assigning higher mutation rates to worse-ranked individuals. |
| SW50 | Switches from RM to CM during the search. |
| SW60 | Switches from RM to CM using a different switching schedule from SW50. |

