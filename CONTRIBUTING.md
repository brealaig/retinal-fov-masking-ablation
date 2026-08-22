# Contributing

This repository accompanies a scientific manuscript. Changes that affect reported results should be traceable and reproducible.

When adding or modifying an experiment:

1. update or add the corresponding configuration;
2. preserve the dataset split unless the experiment explicitly studies the split;
3. store result summaries in machine-readable form;
4. document any change that can alter reported metrics;
5. add a test when modifying preprocessing or metric computation.

For manuscript-associated releases, avoid rewriting Git history after the release tag has been cited.
