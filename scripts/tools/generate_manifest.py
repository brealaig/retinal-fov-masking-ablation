import argparse
import pandas as pd
from pathlib import Path

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--root_dir", required=True, help="Root directory containing images")
argument_parser.add_argument("--labels_csv", required=True, help="CSV with columns: filename,true_label (0..4)")
argument_parser.add_argument("--dataset_name", required=True)
argument_parser.add_argument("--out_csv", required=True)
args = argument_parser.parse_args()

root_directory = Path(args.root_dir)
labels_dataframe = pd.read_csv(args.labels_csv)
labels_dataframe["filename"] = labels_dataframe["filename"].astype(str)

file_paths_by_name = {}
duplicate_filenames = set()

for file_path in root_directory.rglob("*.*"):
    filename = file_path.name
    if filename in file_paths_by_name:
        duplicate_filenames.add(filename)
    else:
        file_paths_by_name[filename] = str(file_path)

if duplicate_filenames:
    raise SystemExit(
        f"[ERROR] Duplicate filenames found in root_dir "
        f"(e.g. {list(sorted(duplicate_filenames))[:10]}). "
        "Use relative paths as keys or rename files to make them unique."
    )

labels_dataframe["filepath"] = labels_dataframe["filename"].map(file_paths_by_name)
missing_file_count = labels_dataframe["filepath"].isna().sum()

if missing_file_count:
    print(
        f"[WARN] {missing_file_count} files from labels were not found in "
        f"{root_directory}"
    )

manifest_dataframe = pd.DataFrame({
    "filepath": labels_dataframe["filepath"],
    "class_5": labels_dataframe["true_label"].astype(int),
    "dataset_name": args.dataset_name
}).dropna()

manifest_dataframe.to_csv(args.out_csv, index=False)

print(
    f"[OK] Manifest saved to {args.out_csv} | "
    f"n={len(manifest_dataframe)}"
)
