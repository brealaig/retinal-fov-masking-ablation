import argparse
import pandas as pd
from pathlib import Path

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--results_csv", required=True, help="gradcam_results.csv from the run")
argument_parser.add_argument("--dataset_name", default="InternalVal")
argument_parser.add_argument("--out_csv", required=True)
args = argument_parser.parse_args()

results_dataframe = pd.read_csv(args.results_csv)

if "true_label" not in results_dataframe.columns and "y_true" in results_dataframe.columns:
    results_dataframe["true_label"] = results_dataframe["y_true"]

if "filepath" not in results_dataframe.columns and "image_path" in results_dataframe.columns:
    results_dataframe["filepath"] = results_dataframe["image_path"]

missing_columns = [
    column_name
    for column_name in ["filepath", "true_label"]
    if column_name not in results_dataframe.columns
]

if missing_columns:
    raise SystemExit(
        f"Missing columns in results: {missing_columns}. "
        "Make sure batch_explain_val exports true_label."
    )

manifest_dataframe = pd.DataFrame({
    "filepath": results_dataframe["filepath"],
    "class_5": results_dataframe["true_label"].astype(int),
    "dataset_name": args.dataset_name
}).dropna()

manifest_dataframe = (
    manifest_dataframe
    .drop_duplicates(subset=["filepath"])
    .reset_index(drop=True)
)

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
manifest_dataframe.to_csv(args.out_csv, index=False)

print(
    f"[OK] Internal manifest saved to {args.out_csv} | "
    f"n={len(manifest_dataframe)}"
)
