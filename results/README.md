# Results

Store machine-readable summaries and manuscript-facing figures here.

Recommended structure:

```text
results/
├── aptos2019/
│   ├── internal/
│   ├── external/
│   ├── calibration/
│   └── gradcam/
└── eyepacs/
    ├── internal/
    ├── external/
    ├── calibration/
    └── gradcam/
```

Prefer CSV/JSON for numerical summaries. Avoid committing large training logs or model checkpoints unless essential.
