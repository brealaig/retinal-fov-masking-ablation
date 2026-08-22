# Executable scripts

Place command-line entry points here.

Recommended organization:

```text
preprocess_dataset.py
train_model.py
evaluate_internal.py
evaluate_external.py
evaluate_calibration.py
generate_gradcam.py
```

Scripts should import reusable functions from `src/retinal_fov_masking/` instead of duplicating implementation logic.
