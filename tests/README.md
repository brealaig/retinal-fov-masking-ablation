# Tests

Add lightweight tests for the parts of the pipeline most likely to affect reproducibility, especially:

- FOV mask generation;
- fixed/adaptive feathering;
- label mapping;
- QWK calculation;
- ECE calculation;
- image normalization;
- output dimensions and ranges.

A small synthetic image fixture is preferable to redistributing restricted retinal images.
