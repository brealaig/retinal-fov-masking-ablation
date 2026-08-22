# Manuscript mapping

Use this directory to map manuscript tables and figures to their generating scripts and result files.

Recommended file:

```text
paper/result_manifest.csv
```

Suggested columns:

```text
manuscript_item,description,script,inputs,output,commit
```

Example:

```text
Table_X,Internal QWK summary,scripts/evaluate_internal.py,...,...,...
```

This makes it much easier to verify the exact provenance of every reported result during peer review.
