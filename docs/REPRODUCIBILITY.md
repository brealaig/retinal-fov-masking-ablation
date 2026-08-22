# Reproducibility checklist

Before creating the manuscript-associated release, verify every item below.

## Code

- [ ] All manuscript experiments can be launched from scripts or clearly documented commands.
- [ ] Reusable logic is located under `src/`.
- [ ] No essential logic exists only in an unpublished notebook.
- [ ] Random seeds are explicitly set and documented.
- [ ] Local machine paths have been removed.
- [ ] No API keys, credentials, or personal tokens are present.

## Environment

- [ ] Python version is recorded.
- [ ] TensorFlow version is pinned.
- [ ] CUDA/cuDNN versions are documented when GPU execution depends on them.
- [ ] Exact dependency versions are exported.
- [ ] Hardware used for manuscript runs is documented.

## Data

- [ ] Dataset sources and access dates are recorded.
- [ ] Dataset licensing / terms are respected.
- [ ] Raw datasets are not committed.
- [ ] Split-generation logic or split manifests are available.
- [ ] Label mappings are documented.
- [ ] Exclusion criteria are documented.

## Training

- [ ] Architecture is documented.
- [ ] Initialization / transfer-learning weights are documented.
- [ ] Image size is documented.
- [ ] Augmentations are documented.
- [ ] Optimizer and learning rate are documented.
- [ ] Epoch count / early stopping are documented.
- [ ] Model-selection criterion is documented.

## Evaluation

- [ ] Internal evaluation can be reproduced.
- [ ] External Messidor-2 evaluation can be reproduced.
- [ ] QWK computation is reproducible.
- [ ] Classification metrics are reproducible.
- [ ] ECE and NLL computation is reproducible.
- [ ] Grad-CAM generation and summary metrics are reproducible.

## Manuscript mapping

- [ ] Every main table has a source result file or generation script.
- [ ] Every main figure has a generation script.
- [ ] Result filenames are stable.
- [ ] The manuscript cites a fixed repository release/tag.
- [ ] The final Git commit hash is recorded.
- [ ] The release is archived with a persistent DOI if possible.

## Release

Recommended workflow:

```bash
git status
git add .
git commit -m "Manuscript reproducibility release"
git tag -a v1.0.0 -m "Code associated with manuscript submission"
git push origin main
git push origin v1.0.0
```

After the repository is complete, archive the tagged release in a long-term research repository and update `CITATION.cff` with the DOI.
