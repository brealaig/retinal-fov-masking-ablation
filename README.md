# Retinal Field-of-View Masking Ablation

Supplementary code and reproducibility materials for the study of retinal field-of-view (FOV) masking strategies in multiclass diabetic retinopathy grading.

The repository is designed to accompany the manuscript comparing four preprocessing strategies:

1. **No masking**
2. **Hard masking**
3. **Fixed feathering**
4. **Adaptive feathering**

Experiments are organized around **APTOS 2019** and **EyePACS** as development datasets, with **Messidor-2** used for independent external evaluation. The repository also provides space for classification, calibration, and Grad-CAM analyses.

---

## Repository structure

```text
retinal-fov-masking-ablation/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── reproducibility.yml
│   └── pull_request_template.md
├── configs/
│   ├── README.md
│   └── example_experiment.yaml
├── docs/
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   └── REPRODUCIBILITY.md
├── paper/
│   ├── README.md
│   └── result_manifest.csv
├── results/
│   ├── README.md
│   ├── aptos2019/
│   │   ├── adaptive_feather/
│   │   │   ├── seed42/
│   │   │   ├── seed123/
│   │   │   └── seed2024/
│   │   ├── fixed_feather/
│   │   │   ├── seed42/
│   │   │   ├── seed123/
│   │   │   └── seed2024/
│   │   ├── hard_mask/
│   │   │   ├── seed42/
│   │   │   ├── seed123/
│   │   │   └── seed2024/
│   │   └── no_mask/
│   │       ├── seed42/
│   │       ├── seed123/
│   │       └── seed2024/
│   └── eyepacs/
│       ├── adaptive_feather/
│       │   ├── seed42/
│       │   ├── seed123/
│       │   └── seed2024/
│       ├── fixed_feather/
│       │   ├── seed42/
│       │   ├── seed123/
│       │   └── seed2024/
│       ├── hard_mask/
│       │   ├── seed42/
│       │   ├── seed123/
│       │   └── seed2024/
│       └── no_mask/
│           ├── seed42/
│           ├── seed123/
│           └── seed2024/
├── scripts/
│   ├── Pipeline/
│   │   ├── rd_on_the_fly_mask_ablation.py
│   │   └── train_mask_ablation.py
│   ├── data_preprocessing/
│   │   ├── README.md
│   │   ├── calculate_p50.py
│   │   ├── generate_ablation_variants.py
│   │   └── resize_dataset.py
│   ├── gradcamtools/
│   │   ├── analyze_gradcam_alignment.py
│   │   ├── batch_explain_external_dataset.py
│   │   ├── batch_explain_mask_ablation.py
│   │   ├── batch_explain_val.py
│   │   ├── make_gradcam.py
│   │   ├── quantify_explanations.py
│   │   └── triage_report.py
│   └── tools/
│       ├── check_double_clahe.py
│       ├── eval_metrics.py
│       ├── from_results_to_manifest.py
│       ├── from_results_to_manifest_overlay.py
│       ├── generate_manifest.py
│       ├── make_eval_dir_from_csv.py
│       ├── make_eval_dir_from_csv_bylabel.py
│       ├── make_labels_from_csv.py
│       └── merge_results_with_labels.py
├── src/
│   └── retinal_fov_masking/
│       ├── __init__.py
│       ├── preprocessing/
│       │   └── __init__.py
│       ├── training/
│       │   └── __init__.py
│       ├── evaluation/
│       │   └── __init__.py
│       ├── interpretability/
│       │   └── __init__.py
│       ├── calibration/
│       │   └── __init__.py
│       └── utils/
│           └── __init__.py
├── .gitignore
├── CITATION.cff
├── CODE_AVAILABILITY.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

Each experiment seed directory under `results/<dataset>/<strategy>/` follows the same organization:

```text
seed*/
├── experiment_config.json
├── internal/
│   ├── calibration/
│   │   └── calibration_summary.json
│   ├── gradcam/
│   │   ├── overlays/
│   │   └── summaries/
│   └── metrics/
│       └── *.csv
└── external/
    └── messidor2/
        ├── gradcam/
        │   ├── overlays/
        │   └── summaries/
        └── metrics/
            └── *.csv
```
## Quick start: end-to-end reproduction tutorial

This section describes how the repository components fit together and provides a practical workflow for reproducing the retinal field-of-view masking experiments.

The core workflow is:

```text
Raw retinal images and labels
            │
            ▼
Optional image resizing
resize_dataset.py
            │
            ▼
Brightness characterization
calculate_p50.py
            │
            ▼
FOV preprocessing variants
generate_ablation_variants.py
            │
            ▼
Dataset organization
no_mask / hard_mask / fixed_feather / adaptive_feather
            │
            ▼
Training, internal evaluation and calibration
train_mask_ablation.py
            │
            ├───────────────┐
            ▼               ▼
Classification results   Grad-CAM analysis
                         batch_explain_mask_ablation.py
                                │
                                ▼
                         analyze_gradcam_alignment.py
                                │
                                ▼
                    Tables, summaries and overlays
```

The preprocessing and experiment scripts are designed to be used together, but the raw retinal datasets themselves are **not included in the repository**.

---

### 1. Clone the repository

```bash
git clone https://github.com/brealaig/retinal-fov-masking-ablation.git
cd retinal-fov-masking-ablation
```

All commands in this guide assume that they are executed from the repository root unless stated otherwise.

---

### 2. Create a Python environment

A dedicated environment is strongly recommended because the experimental pipeline uses a fixed TensorFlow/Keras stack.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the repository dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The provided dependency file reproduces the TensorFlow 2.10-based environment used by the experimental pipeline.

> **Platform note:** some packages in `requirements.txt`, particularly the TensorFlow Intel/DirectML components, are Windows-oriented. Users reproducing the experiments on another operating system may need to adapt the accelerator-specific TensorFlow packages while preserving the main package versions whenever possible.

To confirm that TensorFlow is available:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

---

### 3. Understand which scripts are entry points

The repository contains executable scripts as well as modules that are mainly imported by other scripts.

The main workflow uses:

| Script | Role |
|---|---|
| `scripts/data_preprocessing/resize_dataset.py` | Optional aspect-ratio-preserving image resizing |
| `scripts/data_preprocessing/calculate_p50.py` | Computes retinal brightness (`p50`) and brightness categories |
| `scripts/data_preprocessing/generate_ablation_variants.py` | Generates hard-mask, fixed-feather, and adaptive-feather images |
| `scripts/Pipeline/rd_on_the_fly_mask_ablation.py` | Dataset loading and preprocessing module used by the training pipeline |
| `scripts/Pipeline/train_mask_ablation.py` | Main training, calibration, and internal-evaluation entry point |
| `scripts/gradcamtools/batch_explain_mask_ablation.py` | Recommended Grad-CAM entry point for internal experiment splits |
| `scripts/gradcamtools/analyze_gradcam_alignment.py` | Compares Grad-CAM behavior across masking variants |
| `scripts/gradcamtools/batch_explain_external_dataset.py` | Grad-CAM analysis for an independently labeled external dataset |

The remaining files under `scripts/gradcamtools/` and `scripts/tools/` provide lower-level, post-processing, compatibility, or specialized analysis functionality.

To inspect the arguments accepted by any command-line script, use:

```bash
python path/to/script.py --help
```

For example:

```bash
python scripts/Pipeline/train_mask_ablation.py --help
```

---

### 4. Obtain the datasets

The experiments use:

- **APTOS 2019**
- **EyePACS**
- **Messidor-2** for independent external evaluation

The raw images are not redistributed by this repository. Follow the access instructions in [`docs/DATASETS.md`](docs/DATASETS.md).

A convenient local organization is:

```text
data/
├── aptos2019/
│   ├── raw/
│   └── labels_source.csv
├── eyepacs/
│   ├── raw/
│   └── labels_source.csv
└── messidor2/
    ├── raw/
    └── labels_source.csv
```

`data/` is intended to remain local and should not contain redistributed third-party images in commits.

---

### 5. Prepare the experiment labels

The training pipeline expects a CSV that can resolve every image and define its experimental split.

The recommended schema is:

```text
filename,label,dataset,usage,brightness_category,p50
```

For example:

```csv
filename,label,dataset,usage,brightness_category,p50
image_0001.png,0,APTOS,train,Normal,118.2
image_0002.png,2,APTOS,val_calib,Dark,96.4
image_0003.png,1,APTOS,val_eval,Bright,143.8
image_0004.png,3,APTOS,int_test,Normal,121.1
image_0005.png,4,APTOS,hold_out,VeryDark,78.6
```

The required experimental `usage` values are:

```text
train
val_calib
val_eval
int_test
hold_out
```

These splits have different purposes:

- `train`: model optimization;
- `val_calib`: temperature-scaling calibration;
- `val_eval`: validation evaluation;
- `int_test`: internal test split;
- `hold_out`: final internal hold-out evaluation.

The training pipeline does **not** randomly recreate these splits when `usage` is already defined. This is important for reproducibility.

If the source CSV contains a `level` column instead of `label`, the current pipeline can internally interpret `level` as the class label.

---

### 6. Optional: resize the retinal images

`scripts/data_preprocessing/resize_dataset.py` can generate square images using letterboxing while preserving the original aspect ratio.

For example:

```bash
python scripts/data_preprocessing/resize_dataset.py --images data/aptos2019/raw --output work/aptos2019/resized --sizes 224
```

The generated images will be stored under:

```text
work/
└── aptos2019/
    └── resized/
        └── r224/
```

Multiple resolutions can be generated in one run:

```bash
python scripts/data_preprocessing/resize_dataset.py --images data/aptos2019/raw --output work/aptos2019/resized --sizes 224 512
```

This step is optional for the current training loader because the model input pipeline also resizes images to the selected `--img_size`.

If preprocessing variants are generated from resized images, make sure that the **same source resolution** is used for all four ablation conditions.

---

### 7. Calculate retinal brightness (`p50`)

The adaptive preprocessing strategy uses image brightness information.

Run:

```bash
python scripts/data_preprocessing/calculate_p50.py --images data/aptos2019/raw --csv data/aptos2019/labels_source.csv --output work/aptos2019/labels_with_brightness.csv
```

The script adds or updates:

```text
brightness_category
p50
```

The brightness categories used by the preprocessing pipeline are:

```text
ExtremelyDark
VeryDark
Dark
Normal
Bright
VeryBright
```

The resulting CSV should be used in the next preprocessing stage.

You can quickly inspect the result with:

```bash
python -c "import pandas as pd; df=pd.read_csv('work/aptos2019/labels_with_brightness.csv'); print(df.head()); print(df['brightness_category'].value_counts())"
```

---

### 8. Generate the FOV-masking variants

Use the brightness-enriched CSV to generate the three processed conditions:

```bash
python scripts/data_preprocessing/generate_ablation_variants.py --images data/aptos2019/raw --labels work/aptos2019/labels_with_brightness.csv --output work/aptos2019/variants
```

The script generates:

```text
work/aptos2019/variants/
├── images_hard_mask_black/
├── images_fixed_feather/
├── images_adaptive_feather/
├── qa_outline_hr/
├── qa_mask_hr/
└── _meta_delineado_tres_salidas.csv
```

The processed variants correspond to:

| Generated directory | Experimental variant |
|---|---|
| original images | `no_mask` |
| `images_hard_mask_black/` | `hard_mask` |
| `images_fixed_feather/` | `fixed_feather` |
| `images_adaptive_feather/` | `adaptive_feather` |

The QA directories are useful for visually inspecting FOV segmentation before starting computationally expensive model training.

---

### 9. Build the dataset directory expected by the training pipeline

`train_mask_ablation.py` identifies a preprocessing condition using:

```text
<dataset_root>/<variant>/
```

Therefore, the final local dataset should follow:

```text
data/
└── aptos2019/
    ├── labels.csv
    ├── no_mask/
    ├── hard_mask/
    ├── fixed_feather/
    └── adaptive_feather/
```

For example:

```text
data/aptos2019/
├── labels.csv
├── no_mask/
│   ├── image_0001.png
│   ├── image_0002.png
│   └── ...
├── hard_mask/
│   └── ...
├── fixed_feather/
│   └── ...
└── adaptive_feather/
    └── ...
```

The generated preprocessing directories can therefore be moved, copied, renamed, or linked as follows:

```text
original images
        → no_mask/

images_hard_mask_black/
        → hard_mask/

images_fixed_feather/
        → fixed_feather/

images_adaptive_feather/
        → adaptive_feather/
```

The preprocessing generator may preserve brightness-category subdirectories. They do not necessarily need to be flattened: the experiment loader can recursively resolve image filenames.

However, filenames should be unique inside a variant. Duplicate filenames can make image resolution ambiguous and may cause the pipeline to stop intentionally.

Finally, copy the brightness-enriched experiment CSV as:

```text
data/aptos2019/labels.csv
```

Repeat the same organization for EyePACS.

---

### 10. Verify the dataset before training

Before launching a full experiment, verify that all four variants exist:

```bash
python -c "from pathlib import Path; root=Path('data/aptos2019'); print({v:(root/v).exists() for v in ['no_mask','hard_mask','fixed_feather','adaptive_feather']})"
```

Expected result:

```text
{
    'no_mask': True,
    'hard_mask': True,
    'fixed_feather': True,
    'adaptive_feather': True
}
```

Also inspect the split distribution:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/aptos2019/labels.csv'); print(df['usage'].value_counts()); print(pd.crosstab(df['usage'], df['label']))"
```

This is a useful reproducibility check before any training begins.

---

### 11. Train one experiment

The main experiment entry point is:

```text
scripts/Pipeline/train_mask_ablation.py
```

For example, to train **adaptive feathering on APTOS with seed 42**:

```bash
python scripts/Pipeline/train_mask_ablation.py --dataset_root data/aptos2019 --dataset_name APTOS --labels_csv data/aptos2019/labels.csv --variant adaptive_feather --seed 42 --exp_root experiments
```

The valid variants are:

```text
no_mask
hard_mask
fixed_feather
adaptive_feather
```

The default model is EfficientNet-B0 with an input size of 224 × 224.

By default, training is performed in multiple stages:

```text
Phase A → frozen backbone / classification head
Phase B → selective fine-tuning
Phase C → deeper fine-tuning
Phase D → lightweight final fine-tuning
```

After training, the pipeline automatically performs temperature calibration using `val_calib` and evaluates the requested internal splits.

The experiment directory is created automatically using:

```text
<dataset>_<variant>_seed<seed>
```

For the previous example:

```text
experiments/
└── APTOS_adaptive_feather_seed42/
```

---

### 12. Inspect the experiment outputs

A completed experiment may contain artifacts such as:

```text
experiments/
└── APTOS_adaptive_feather_seed42/
    ├── experiment_config.json
    ├── compute_device.json
    ├── split_counts.json
    ├── run.log
    ├── best_phaseA.weights.h5
    ├── best_phaseB.weights.h5
    ├── best_phaseC.weights.h5
    ├── best_phaseD.weights.h5
    ├── final_all_phases.weights.h5
    ├── calibration_summary.json
    ├── metrics_all_requested_splits.csv
    └── tflite_export/
```

Not every intermediate checkpoint is guaranteed to exist if a training phase is disabled or stopped differently.

The most important reproducibility artifacts are:

- `experiment_config.json`;
- `compute_device.json`;
- `split_counts.json`;
- final/best model weights;
- calibration summary;
- evaluation metrics.

---

### 13. Run the complete ablation matrix

The reported experimental design combines:

```text
4 preprocessing variants
×
multiple random seeds
```

The repository results currently use:

```text
42
123
2024
```

A complete dataset-level matrix is therefore:

```text
no_mask × 42
no_mask × 123
no_mask × 2024

hard_mask × 42
hard_mask × 123
hard_mask × 2024

fixed_feather × 42
fixed_feather × 123
fixed_feather × 2024

adaptive_feather × 42
adaptive_feather × 123
adaptive_feather × 2024
```

On Windows PowerShell, this can be automated:

```powershell
$variants = @(
    "no_mask",
    "hard_mask",
    "fixed_feather",
    "adaptive_feather"
)

$seeds = @(42, 123, 2024)

foreach ($variant in $variants) {
    foreach ($seed in $seeds) {
        python scripts/Pipeline/train_mask_ablation.py `
            --dataset_root data/aptos2019 `
            --dataset_name APTOS `
            --labels_csv data/aptos2019/labels.csv `
            --variant $variant `
            --seed $seed `
            --exp_root experiments
    }
}
```

For EyePACS, change:

```text
data/aptos2019 → data/eyepacs
APTOS          → EYEPACS
```

Keeping the training configuration fixed across variants is essential: preprocessing strategy should remain the primary experimental factor.

---

### 14. Use CPU or GPU explicitly

The training pipeline supports compute-device selection.

The default behavior can be inspected with:

```bash
python scripts/Pipeline/train_mask_ablation.py --help
```

When the corresponding device utility is available, experiments can be launched using options such as:

```text
--device auto
--device cpu
--gpu_index 0
```

For reproducibility, the selected compute device is stored in:

```text
compute_device.json
```

---

### 15. Re-evaluate an existing model without retraining

The same entry point can evaluate existing weights:

```bash
python scripts/Pipeline/train_mask_ablation.py --dataset_root data/aptos2019 --dataset_name APTOS --labels_csv data/aptos2019/labels.csv --variant adaptive_feather --seed 42 --exp_root experiments --evaluate_only --weights experiments/APTOS_adaptive_feather_seed42/final_all_phases.weights.h5
```

Specific internal splits can also be requested:

```bash
python scripts/Pipeline/train_mask_ablation.py --dataset_root data/aptos2019 --dataset_name APTOS --labels_csv data/aptos2019/labels.csv --variant adaptive_feather --seed 42 --exp_root experiments --evaluate_only --weights experiments/APTOS_adaptive_feather_seed42/final_all_phases.weights.h5 --eval_splits val_eval int_test hold_out
```

This is useful when validating archived checkpoints without repeating model optimization.

---

### 16. Generate internal Grad-CAM results

After the model has been trained, Grad-CAM maps and attention metrics can be generated with:

```text
scripts/gradcamtools/batch_explain_mask_ablation.py
```

For example:

```bash
python scripts/gradcamtools/batch_explain_mask_ablation.py --dataset_root data/aptos2019 --dataset_name APTOS --labels_csv data/aptos2019/labels.csv --variant adaptive_feather --exp_dir experiments/APTOS_adaptive_feather_seed42 --seed 42 --splits hold_out --save_overlays
```

The script loads the experiment weights and generates per-image attention measurements including:

- retinal attention energy;
- extraretinal attention energy;
- border concentration;
- peripheral attention;
- prediction correctness;
- prediction confidence.

Typical outputs include:

```text
gradcam_summary_hold_out.csv
gradcam_by_class_hold_out.csv
gradcam_by_brightness_hold_out.csv
```

When `--save_overlays` is enabled, visual overlays are also stored under the experiment's Grad-CAM directory.

For a quick test before processing the full split:

```bash
python scripts/gradcamtools/batch_explain_mask_ablation.py --dataset_root data/aptos2019 --dataset_name APTOS --labels_csv data/aptos2019/labels.csv --variant adaptive_feather --exp_dir experiments/APTOS_adaptive_feather_seed42 --seed 42 --splits hold_out --max_images 20 --save_overlays --overlay_limit 20
```

This small run is strongly recommended before processing every image.

---

### 17. Generate Grad-CAM for all variants and seeds

Grad-CAM should be generated for the same experiment matrix used for classification.

Conceptually:

```text
APTOS
├── no_mask
│   ├── seed42
│   ├── seed123
│   └── seed2024
├── hard_mask
├── fixed_feather
└── adaptive_feather
```

and equivalently for EyePACS.

This allows the same retinal image to be compared across preprocessing conditions instead of interpreting isolated heatmaps.

---

### 18. Analyze Grad-CAM alignment across variants

Once the experiment directories contain their Grad-CAM summary CSV files, run:

```bash
python scripts/gradcamtools/analyze_gradcam_alignment.py --exp_root experiments --source_dataset_name APTOS --split hold_out
```

The script searches experiment directories matching:

```text
<dataset>_<variant>_seed<seed>
```

and combines the available masking variants.

The analysis produces outputs such as:

```text
_gradcam_alignment_analysis/
└── APTOS/
    └── hold_out/
        ├── gradcam_alignment_per_image_hold_out.csv
        ├── gradcam_alignment_summary_by_variant_hold_out.csv
        ├── gradcam_alignment_by_class_hold_out.csv
        ├── gradcam_alignment_by_brightness_hold_out.csv
        ├── gradcam_alignment_by_pattern_hold_out.csv
        ├── gradcam_alignment_by_correctness_hold_out.csv
        ├── gradcam_alignment_paired_filenames_hold_out.csv
        ├── gradcam_alignment_thresholds_hold_out.json
        └── gradcam_alignment_interpretation_hold_out.md
```

This stage is especially useful for identifying whether a masking strategy reduces background attention but introduces another shortcut, such as excessive concentration at the FOV boundary.

---

### 19. External Grad-CAM evaluation

`batch_explain_external_dataset.py` is intended to apply a trained source-domain model to an independently labeled external dataset such as Messidor-2.

Its main inputs are:

```text
--model_exp_dir
--external_dataset_root
--external_dataset_name
--external_labels_csv
--variant
```

Inspect the exact CLI with:

```bash
python scripts/gradcamtools/batch_explain_external_dataset.py --help
```

A conceptual example is:

```bash
python scripts/gradcamtools/batch_explain_external_dataset.py --model_exp_dir experiments/APTOS_adaptive_feather_seed42 --external_dataset_root data/messidor2 --external_dataset_name MESSIDOR2 --external_labels_csv data/messidor2/labels.csv --variant adaptive_feather --save_overlays
```

The external analysis writes Grad-CAM summaries separately from the source-domain experiment so that internal and cross-domain attention behavior can be compared.

> **Current compatibility note:** this script depends on additional helper imports used by the original experimental pipeline. Before relying on it in a clean installation, verify that all helper modules referenced by the script are present and that the `gradcamtools` package path matches the import statements.

---

### 20. Utility scripts

The scripts under `scripts/tools/` support dataset preparation and result post-processing.

They are not all required for the core training path.

#### Create a simple label file

`make_labels_from_csv.py` can convert a larger experiment CSV into:

```text
filename,true_label
```

For example:

```bash
python scripts/tools/make_labels_from_csv.py --csv data/aptos2019/labels.csv --out_csv work/aptos2019/holdout_labels.csv --usage hold_out --dataset APTOS
```

#### Generate a manifest

A manifest linking file paths to labels can be created with:

```bash
python scripts/tools/generate_manifest.py --root_dir data/aptos2019/adaptive_feather --labels_csv work/aptos2019/holdout_labels.csv --dataset_name APTOS --out_csv work/aptos2019/manifest.csv
```

#### Merge result files with labels

If a prediction CSV does not contain ground-truth labels:

```bash
python scripts/tools/merge_results_with_labels.py --results_csv path/to/results.csv --labels_csv path/to/labels.csv --out_csv path/to/results_with_labels.csv
```

#### Compute classification and calibration metrics

`eval_metrics.py` can compute QWK, calibration and referable-DR metrics from a compatible prediction CSV containing at least:

```text
true_label
pred_label
probs_json
```

For example:

```bash
python scripts/tools/eval_metrics.py --csv path/to/predictions.csv --output_json path/to/metrics.json --export_cm_csv path/to/confusion_matrix.csv --save_plots path/to/plots
```

Additional utility scripts are available for constructing evaluation directories and converting result files to manifests:

```text
check_double_clahe.py
from_results_to_manifest.py
from_results_to_manifest_overlay.py
make_eval_dir_from_csv.py
make_eval_dir_from_csv_bylabel.py
```

Use:

```bash
python scripts/tools/<script_name>.py --help
```

before running one of these specialized utilities.

---

### 21. Legacy and low-level Grad-CAM utilities

Some files under `scripts/gradcamtools/` originate from earlier versions of the experimental pipeline.

In particular:

```text
batch_explain_val.py
make_gradcam.py
quantify_explanations.py
triage_report.py
```

should be considered lower-level or specialized utilities rather than the recommended first entry point for reproducing the current masking experiment.

Some of them reference historical module names from the original development environment.

For the current ablation workflow, prefer:

```text
batch_explain_mask_ablation.py
        ↓
analyze_gradcam_alignment.py
```

unless a specialized analysis specifically requires one of the older utilities.

---

### 22. Move reproducible outputs into `results/`

Training is normally performed under a local working directory such as:

```text
experiments/
```

After an experiment has been verified, the manuscript-facing outputs can be organized under the repository's `results/` hierarchy.

For example:

```text
results/
└── aptos2019/
    └── adaptive_feather/
        └── seed42/
            ├── experiment_config.json
            ├── internal/
            │   ├── calibration/
            │   ├── gradcam/
            │   └── metrics/
            └── external/
                └── messidor2/
                    ├── gradcam/
                    └── metrics/
```

The repository should contain the compact artifacts needed to reproduce the reported tables and figures, rather than raw datasets or unnecessarily large intermediate files.

Large trained checkpoints should normally be archived separately when repository size becomes a concern.

---

### 23. Recommended reproducibility checklist

Before considering an experiment complete, verify that:

- the exact same label/split CSV is used across the four preprocessing variants;
- all variants contain the expected images;
- no filenames are duplicated ambiguously;
- the same architecture and image size are used across variants;
- the training policies remain fixed unless they are explicitly part of another ablation;
- the random seed is recorded;
- `experiment_config.json` exists;
- `compute_device.json` exists when device tracking is enabled;
- `split_counts.json` matches the intended experimental split;
- calibration is fitted only on `val_calib`;
- `hold_out` is not used for training or temperature fitting;
- Grad-CAM is generated from the corresponding trained checkpoint;
- internal and external results are kept separate;
- the exact outputs used by the manuscript are copied into `results/`.

---

### 24. Minimal reproduction path

For users who only want to understand the basic workflow, the minimum sequence is:

```text
1. Clone repository
2. Install requirements
3. Obtain one retinal dataset
4. Prepare labels.csv with fixed usage splits
5. Run calculate_p50.py
6. Run generate_ablation_variants.py
7. Arrange no_mask / hard_mask / fixed_feather / adaptive_feather
8. Run train_mask_ablation.py for one variant and one seed
9. Inspect metrics and calibration
10. Run batch_explain_mask_ablation.py
11. Inspect Grad-CAM summaries and overlays
```

For full experimental reproduction:

```text
APTOS + EyePACS
        ×
4 masking strategies
        ×
3 seeds
        +
internal evaluation
        +
temperature calibration
        +
Grad-CAM analysis
        +
independent Messidor-2 evaluation
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the experimental design and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for additional reproducibility information.

## Experimental design

The repository should preserve the study's one-factor-at-a-time comparison of retinal FOV preprocessing strategies while keeping the remaining training and evaluation pipeline fixed.

| Strategy | Short name | Intended role |
|---|---|---|
| No masking | `no_mask` | Baseline without explicit FOV masking |
| Hard masking | `hard_mask` | Binary suppression of pixels outside the retinal FOV |
| Fixed feathering | `fixed_feather` | Smooth transition at the FOV boundary with a fixed feathering rule |
| Adaptive feathering | `adaptive_feather` | Image-dependent smooth transition at the FOV boundary |

Reported analyses may include:

- Quadratic Weighted Kappa (QWK)
- Standard classification metrics
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Grad-CAM-based attention analysis
- Internal hold-out evaluation
- Cross-domain / external evaluation on Messidor-2

---

## Reproducibility principles

For every reported experiment, the public release should expose or document:

- dataset split or split-generation procedure;
- preprocessing strategy and parameters;
- random seed(s);
- model architecture and initialization;
- optimizer and learning-rate configuration;
- number of epochs and early-stopping criteria;
- image size and augmentation settings;
- checkpoint-selection criterion;
- evaluation scripts;
- software/package versions;
- hardware used for the final runs, when relevant;
- mapping between manuscript tables/figures and generated result files.

Do **not** commit private credentials, local absolute paths, raw datasets, or restricted dataset files.

---

## Results

Machine-readable outputs should be stored under `results/` using stable names. Prefer CSV/JSON for numerical summaries and PNG/PDF/SVG for figures.

Suggested hierarchy:

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

Large model checkpoints should normally be distributed through a release, archival repository, or model-hosting service rather than committed directly to Git.

---

## Citation

If you use this repository, please cite the associated manuscript and the archived software release once available.

GitHub can read [`CITATION.cff`](CITATION.cff) and expose a **Cite this repository** button automatically.

---

## Data availability

APTOS 2019, EyePACS, and Messidor-2 are third-party datasets. Their original licenses, access conditions, and terms of use remain applicable. This repository does not redistribute the raw images.

See [`docs/DATASETS.md`](docs/DATASETS.md).

---

## Code availability

The final manuscript-facing release should be tagged (for example, `v1.0.0`) and archived with a persistent identifier such as a DOI. The commit hash or release tag cited in the manuscript should correspond to the exact code used for the reported analyses.

---

## License

Unless replaced before publication, the repository scaffold is provided under the MIT License. Dataset licenses are separate and are not affected by this software license.

## Authors

- Luis Alejandro Cely Díez
- Gabriela Martínez Eslava
- Julio Barón Velandia
