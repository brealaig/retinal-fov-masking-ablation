import os
import shutil
import argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def find_file(images_root: Path, filename: str) -> Path | None:
    for file_path in images_root.rglob(filename):
        if file_path.is_file():
            return file_path
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
    argument_parser.add_argument(
        "--csv",
        required=True,
        help="Master CSV with filename, level, dataset, split, usage"
    )
    argument_parser.add_argument(
        "--images_root",
        required=True,
        help="Root directory containing the images, searched by filename"
    )
    argument_parser.add_argument(
        "--out_dir",
        required=True,
        help="Destination directory used as the evaluation view"
    )
    argument_parser.add_argument(
        "--usage",
        default="",
        help="Filter by usage (internal/external). Empty means no filter."
    )
    argument_parser.add_argument(
        "--split",
        default="",
        help="Filter by split (val/test/train). Empty means no filter."
    )
    argument_parser.add_argument(
        "--dataset",
        default="",
        help="Filter by exact dataset name (e.g. MESSIDOR2). Empty means no filter."
    )
    argument_parser.add_argument(
        "--mode",
        default="hardlink",
        choices=["hardlink", "symlink", "copy"]
    )
    argument_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means no limit"
    )
    args = argument_parser.parse_args()

    dataframe = pd.read_csv(args.csv)

    for column_name in ["usage", "split", "dataset", "filename"]:
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
    processed_count = 0

    for filename in dataframe["filename"].tolist():
        if not filename.lower().endswith(IMG_EXTS):
            pass

        source_path = images_root / filename

        if not source_path.exists():
            source_path = find_file(images_root, filename)

        if source_path is None or not source_path.exists():
            missing_files.append(filename)
            continue

        destination_path = output_directory / filename
        link_or_copy(source_path, destination_path, args.mode)
        processed_count += 1

    print(f"[OK] out_dir: {output_directory}")
    print(f"[OK] created: {processed_count} | missing: {len(missing_files)}")

    if missing_files:
        print("[WARN] Missing examples:", missing_files[:10])

if __name__ == "__main__":
    main()
