# Experiment matrix

The central ablation compares four retinal field-of-view preprocessing strategies under otherwise fixed experimental conditions.

## Preprocessing conditions

| ID | Strategy | Repository name |
|---|---|---|
| P0 | No masking | `no_mask` |
| P1 | Hard masking | `hard_mask` |
| P2 | Fixed feathering | `fixed_feather` |
| P3 | Adaptive feathering | `adaptive_feather` |

## Development datasets

- APTOS 2019
- EyePACS

## External dataset

- Messidor-2

## Required experiment groups

For each development dataset and preprocessing condition:

1. train using the fixed architecture and training protocol;
2. evaluate on the internal hold-out set;
3. evaluate on Messidor-2;
4. calculate classification metrics and QWK;
5. calculate calibration metrics (including ECE and NLL);
6. generate Grad-CAM outputs using the manuscript-defined protocol.

## Recommended run identifiers

```text
aptos2019_no_mask
aptos2019_hard_mask
aptos2019_fixed_feather
aptos2019_adaptive_feather

eyepacs_no_mask
eyepacs_hard_mask
eyepacs_fixed_feather
eyepacs_adaptive_feather
```

## Configuration policy

Every run should be reconstructable from a machine-readable configuration file stored under `configs/`.

At minimum, store:

```yaml
experiment_id:
dataset:
preprocessing:
seed:
image_size:
architecture:
weights:
batch_size:
epochs:
optimizer:
learning_rate:
augmentation:
checkpoint_metric:
```

Do not rely on undocumented constants embedded in notebooks.
