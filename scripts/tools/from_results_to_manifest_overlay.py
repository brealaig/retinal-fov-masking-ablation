import argparse
import pandas as pd
from pathlib import Path

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--results_csv", required=True)
argument_parser.add_argument("--dataset_name", default="InternalVal")
argument_parser.add_argument("--out_csv", required=True)
args = argument_parser.parse_args()

results_dataframe = pd.read_csv(args.results_csv)

selected_path_column = None
for column_name in ["out_overlay", "out_original", "out_heatmap"]:
    if column_name in results_dataframe.columns:
        selected_path_column = column_name
        break

if selected_path_column is None:
    raise SystemExit(
        "Could not find out_overlay/out_original/out_heatmap columns in results."
    )

if "y_true" not in results_dataframe.columns:
    raise SystemExit("Could not find y_true column in results.")

manifest_dataframe = pd.DataFrame({
    "filepath": results_dataframe[selected_path_column].astype(str),
    "class_5": results_dataframe["y_true"].astype(int),
    "dataset_name": args.dataset_name
}).dropna()

manifest_dataframe = (
    manifest_dataframe
    .drop_duplicates(subset=["filepath"])
    .reset_index(drop=True)
)

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
manifest_dataframe.to_csv(args.out_csv, index=False, encoding="utf-8")

print(
    f"[OK] Overlay-based manifest saved to {args.out_csv} | "
    f"n={len(manifest_dataframe)}"
)
