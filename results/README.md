# Results
This directory contains the compact manuscript-facing release of the experimental outputs.
## Storage policy
The release keeps machine-readable CSV/JSON artifacts required to audit the manuscript while avoiding large redundant binaries.
- All discovered seed-level aggregate metrics are retained.
- `predictions_hold_out.csv` and `predictions_external.csv` are retained for per-image metric reconstruction.
- Full per-image Grad-CAM summary CSVs are retained for quantitative attribution analysis.
- Bulk Grad-CAM overlay PNGs are **not** committed. Only manuscript example images are retained.
- Per-image `val_eval` and `int_test` prediction files are omitted to reduce repository size; their aggregate metrics remain available.
- Model checkpoints, TFLite exports, raw datasets, TensorBoard data, and training logs are excluded.

## Strategy naming
The repository directory `hard_mask` corresponds to **binary masking** in the manuscript. `adaptive_feather` corresponds to **P50-adaptive feathering**.

## Experiment matrix
### APTOS
- No masking (`no_mask`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
- Binary masking (`hard_mask`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
- Fixed feathering (`fixed_feather`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
- P50-adaptive feathering (`adaptive_feather`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
### EYEPACS
- No masking (`no_mask`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
- Binary masking (`hard_mask`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
- Fixed feathering (`fixed_feather`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432
- P50-adaptive feathering (`adaptive_feather`): 22, 42, 44, 73, 123, 1012, 1729, 2024, 4060, 5432

## Per-seed structure
```text
seed*/
├── experiment_config.json
├── compute_device.json                 # when available
├── split_counts.json                   # when available
├── internal/
│   ├── calibration/calibration_summary.json
│   ├── metrics/*.csv
│   └── gradcam/
│       ├── summaries/*.csv|*.json
│       └── overlays/...                # manuscript examples only
└── external/messidor2/
    ├── metrics/*.csv|*.json
    └── gradcam/
        ├── summaries/*.csv|*.json
        └── overlays/...                # manuscript examples only
```

## Validation
See `_manifests/validation_summary.json`, `_manifests/experiment_matrix.csv`, and `_manifests/results_inventory.csv`.
