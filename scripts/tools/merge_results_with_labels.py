import argparse
import pandas as pd
import re
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CANDIDATE_COLS = [
    "filepath", "filename", "image", "img", "img_path", "image_path", "path",
    "file", "fname", "uid", "id", "id_code", "sample", "sample_path",
    "relative_path", "abs_path"
]

def looks_like_image_name(s: str) -> bool:
    stripped_value = s.strip()
    if not stripped_value:
        return False

    lowercase_value = stripped_value.lower()
    return (
        any(lowercase_value.endswith(extension) for extension in IMG_EXTS)
        and len(lowercase_value) > 4
    )

def extract_name_from_any_cell(row) -> str | None:
    for cell_value in row.values:
        if isinstance(cell_value, str):
            candidate_value = cell_value.strip()
            if not candidate_value:
                continue

            try:
                filename = Path(candidate_value).name
            except Exception:
                filename = candidate_value

            if looks_like_image_name(filename):
                return filename

    return None

def coerce_filename_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for column_name in CANDIDATE_COLS:
        if column_name in df.columns:
            filename_series = df[column_name].apply(
                lambda value: (
                    Path(value).name
                    if isinstance(value, str) and value.strip()
                    else None
                )
            )

            if (
                filename_series.notna().any()
                and filename_series.astype(str)
                .str.lower()
                .str.endswith(IMG_EXTS)
                .any()
            ):
                return (
                    filename_series,
                    f"[OK] filename derived from column '{column_name}'"
                )

    extracted_filenames = []

    for _, row in df.iterrows():
        filename = extract_name_from_any_cell(row)
        extracted_filenames.append(filename)

    filename_series = pd.Series(extracted_filenames)

    if filename_series.notna().any():
        return (
            filename_series,
            "[OK] filename derived by cell scanning heuristic"
        )

    return (
        pd.Series([None] * len(df)),
        "[FAIL] Could not derive filename from any column or cell"
    )

def normalize_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def pick_label_column(lbl: pd.DataFrame) -> str:
    for column_name in [
        "true_label", "class_5", "label", "icdr_label", "grade", "level"
    ]:
        if column_name in lbl.columns:
            return column_name

    raise SystemExit(
        "[ERROR] labels.csv must contain a class column "
        "(true_label / class_5 / label / icdr_label / grade / level)."
    )

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--results_csv", required=True)
    argument_parser.add_argument("--labels_csv", required=True)
    argument_parser.add_argument("--out_csv", required=True)
    args = argument_parser.parse_args()

    results_dataframe = pd.read_csv(args.results_csv)
    labels_dataframe = pd.read_csv(args.labels_csv)

    results_dataframe["__filename_raw"], derivation_message = (
        coerce_filename_series(results_dataframe)
    )
    print(derivation_message)

    label_key_column = (
        "filename"
        if "filename" in labels_dataframe.columns
        else labels_dataframe.columns[0]
    )
    label_column = pick_label_column(labels_dataframe)

    results_dataframe["__key"] = normalize_name(
        results_dataframe["__filename_raw"]
    )
    labels_dataframe["__key"] = normalize_name(
        labels_dataframe[label_key_column].astype(str)
    )

    merged_dataframe = results_dataframe.merge(
        labels_dataframe[["__key", label_column]],
        on="__key",
        how="left"
    )
    merged_dataframe = merged_dataframe.rename(
        columns={label_column: "true_label"}
    )

    total_rows = len(merged_dataframe)
    missing_filename_count = merged_dataframe["__key"].isna().sum()
    missing_label_count = merged_dataframe["true_label"].isna().sum()

    print(
        f"[STATS] total={total_rows} | "
        f"missing_derivable_filename={int(missing_filename_count)} | "
        f"missing_label_after_merge={int(missing_label_count)}"
    )

    merged_dataframe.drop(
        columns=["__key", "__filename_raw"],
        errors="ignore"
    ).to_csv(
        args.out_csv,
        index=False,
        encoding="utf-8"
    )

    print(f"[OK] Wrote {args.out_csv}")
