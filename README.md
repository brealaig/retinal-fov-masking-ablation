# Retinal Field-of-View Masking Ablation

Supplementary code and reproducibility materials for the study of retinal field-of-view (FOV) masking strategies in multiclass diabetic retinopathy grading.

The repository is designed to accompany the manuscript comparing four preprocessing strategies:

1. **No masking**
2. **Hard masking**
3. **Fixed feathering**
4. **Adaptive feathering**

Experiments are organized around **APTOS 2019** and **EyePACS** as development datasets, with **Messidor-2** used for independent external evaluation. The repository also provides space for classification, calibration, and Grad-CAM analyses.

> **Repository status:** scaffold prepared for public release. Experimental scripts, exact package versions, trained-model metadata, and final result files should be added before archival/release.

---

## Repository structure

```text
retinal-fov-masking-ablation/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
├── data/
├── docs/
│   ├── DATASETS.md
│   ├── EXPERIMENTS.md
│   └── REPRODUCIBILITY.md
├── notebooks/
├── paper/
├── results/
├── scripts/
├── src/
│   └── retinal_fov_masking/
│       ├── preprocessing/
│       ├── training/
│       ├── evaluation/
│       ├── interpretability/
│       ├── calibration/
│       └── utils/
└── tests/
```

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-GITHUB-USER>/retinal-fov-masking-ablation.git
cd retinal-fov-masking-ablation
```

### 2. Create an isolated Python environment

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

Before the public release associated with the manuscript, replace the dependency ranges in `requirements.txt` with the **exact tested versions** used to generate the reported results.

### 3. Obtain the datasets

Raw retinal images are **not redistributed in this repository**. Follow the dataset-specific instructions in [`docs/DATASETS.md`](docs/DATASETS.md).

Expected local layout:

```text
data/
├── aptos2019/
├── eyepacs/
└── messidor2/
```

The `data/` directory is excluded from version control.

### 4. Add or run experiment scripts

Executable entry-point scripts should be placed in `scripts/`. Reusable implementation code should be placed under `src/retinal_fov_masking/`.

Recommended naming convention:

```text
scripts/
├── preprocess_dataset.py
├── train_model.py
├── evaluate_internal.py
├── evaluate_external.py
├── evaluate_calibration.py
└── generate_gradcam.py
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the experiment matrix and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the reproducibility checklist.

---

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
