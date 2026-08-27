import os
import shutil
import argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def find_file(images_root: Path, filename: str) -> Path | None:
    direct_path = images_root / filename
    if direct_path.exists():
        return direct_path

    for candidate_path in images_root.rglob(filename):
        if candidate_path.is_file():
            return candidate_path

    return None

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    shutil.copy2(src, dst)

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--csv", required=True)
    argument_parser.add_argument("--images_root", required=True)
    argument_parser.add_argument("--out_dir", required=True)
    argument_parser.add_argument("--usage", default="")
    argument_parser.add_argument("--split", default="")
    argument_parser.add_argument("--dataset", default="")
    argument_parser.add_argument(
        "--mode",
        default="hardlink",
        choices=["hardlink", "symlink", "copy"]
    )
    argument_parser.add_argument(
        "--label_col",
        default="level",
        help="Column containing labels 0..4 (default: level)"
    )
    argument_parser.add_argument("--filename_col", default="filename")
    argument_parser.add_argument("--limit", type=int, default=0)
    args = argument_parser.parse_args()

    dataframe = pd.read_csv(args.csv)

    for column_name in [
        args.filename_col,
        args.label_col,
        "usage",
        "split",
        "dataset"
    ]:
        if column_name in dataframe.columns:
            dataframe[column_name] = dataframe[column_name].astype(str)

    if args.usage:
        dataframe = dataframe[
            dataframe["usage"].str.lower() == args.usage.lower()
        ]

    if args.split:
        dataframe = dataframe[
            dataframe["split"].str.lower() == args.split.lower()
        ]

    if args.dataset:
        dataframe = dataframe[
            dataframe["dataset"].str.upper() == args.dataset.upper()
        ]

    if args.limit and args.limit > 0:
        dataframe = dataframe.head(args.limit)

    images_root = Path(args.images_root)
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    missing_files = []
    created_count = 0

    for _, row in dataframe.iterrows():
        filename = str(row[args.filename_col])
        label = int(float(row[args.label_col]))

        if label < 0 or label > 4:
            continue

        source_path = find_file(images_root, filename)

        if source_path is None or not source_path.exists():
            missing_files.append(filename)
            continue

        destination_path = output_directory / str(label) / filename
        link_or_copy(source_path, destination_path, args.mode)
        created_count += 1

    print(f"[OK] out_dir: {output_directory}")
    print(f"[OK] created: {created_count} | missing: {len(missing_files)}")

    if missing_files:
        print("[WARN] Missing examples:", missing_files[:10])

if __name__ == "__main__":
    main()
