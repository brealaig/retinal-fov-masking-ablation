import argparse
from pathlib import Path
import pandas as pd

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--csv", required=True)
    argument_parser.add_argument("--out_csv", required=True)
    argument_parser.add_argument("--usage", default="")
    argument_parser.add_argument("--split", default="")
    argument_parser.add_argument("--dataset", default="")
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

    labels_dataframe = dataframe[["filename", "level"]].copy()
    labels_dataframe = labels_dataframe.rename(
        columns={"level": "true_label"}
    )
    labels_dataframe["true_label"] = labels_dataframe["true_label"].astype(int)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    labels_dataframe.to_csv(args.out_csv, index=False)

    print(
        f"[OK] Labels saved to {args.out_csv} | "
        f"n={len(labels_dataframe)}"
    )

if __name__ == "__main__":
    main()
