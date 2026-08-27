from __future__ import annotations

import os
import csv
import math
import json
import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2


def read_csv(path: str) -> List[Dict[str, str]]:
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        for row in reader:
            rows.append(row)

    return rows


def index_by_stem_or_idx(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    indexed_rows = {}

    for row in rows:
        stem = row.get(
            "stem",
            "",
        )

        if not stem:
            output_path = (
                row.get(
                    "out_original",
                    "",
                )
                or row.get(
                    "out_overlay",
                    "",
                )
            )

            if output_path:
                stem = os.path.splitext(
                    os.path.basename(
                        output_path
                    )
                )[0]
                stem = stem.replace(
                    "_original",
                    "",
                ).replace(
                    "_overlay",
                    "",
                )

        if not stem:
            stem = f"idx_{row.get('idx', '')}"

        indexed_rows[stem] = row

    return indexed_rows


def merge_rows(
    R: List[Dict[str, str]],
    M: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    result_index = index_by_stem_or_idx(
        R
    )
    metrics_index = index_by_stem_or_idx(
        M
    )

    merged_rows = []

    for stem_key, result_row in result_index.items():
        merged_row = dict(
            result_row
        )

        if stem_key in metrics_index:
            merged_row.update(
                metrics_index[
                    stem_key
                ]
            )
        else:
            merged_row.setdefault(
                "auc_insertion",
                "",
            )
            merged_row.setdefault(
                "auc_deletion",
                "",
            )
            merged_row.setdefault(
                "F_score",
                "",
            )
            merged_row.setdefault(
                "energy_inside",
                "",
            )
            merged_row.setdefault(
                "border_ratio",
                "",
            )

        def to_float(
            value,
            default=0.0,
        ):
            try:
                return float(
                    value
                )
            except Exception:
                return default

        def to_int(
            value,
            default=0,
        ):
            try:
                return int(
                    value
                )
            except Exception:
                return default

        merged_row["y_true"] = to_int(
            merged_row.get(
                "y_true",
                0,
            )
        )
        merged_row["y_pred"] = to_int(
            merged_row.get(
                "y_pred",
                0,
            )
        )
        merged_row["correct"] = to_int(
            merged_row.get(
                "correct",
                0,
            )
        )
        merged_row["prob_pred"] = to_float(
            merged_row.get(
                "prob_pred",
                merged_row.get(
                    "prob_used",
                    0.0,
                ),
            )
        )
        merged_row["auc_insertion"] = to_float(
            merged_row.get(
                "auc_insertion",
                0.0,
            )
        )
        merged_row["auc_deletion"] = to_float(
            merged_row.get(
                "auc_deletion",
                0.0,
            )
        )
        merged_row["F_score"] = to_float(
            merged_row.get(
                "F_score",
                0.0,
            )
        )
        merged_row["energy_inside"] = to_float(
            merged_row.get(
                "energy_inside",
                0.0,
            )
        )
        merged_row["border_ratio"] = to_float(
            merged_row.get(
                "border_ratio",
                0.0,
            )
        )

        merged_rows.append(
            merged_row
        )

    return merged_rows


def select_top(
    rows: List[Dict[str, str]],
    k: int,
    *,
    where=None,
    key=None,
    reverse=True,
) -> List[Dict[str, str]]:
    candidates = [
        row
        for row in rows
        if (
            where(row)
            if where
            else True
        )
    ]

    candidates.sort(
        key=(
            key
            if key
            else (
                lambda row: row["F_score"]
            )
        ),
        reverse=reverse,
    )

    return candidates[:k]


def load_rgb(
    path: str,
    size=(224, 224),
) -> Optional[np.ndarray]:
    image_bgr = cv2.imread(
        path,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        return None

    if size:
        image_bgr = cv2.resize(
            image_bgr,
            size,
            interpolation=cv2.INTER_AREA,
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def draw_caption(
    img_rgb: np.ndarray,
    caption: str,
) -> np.ndarray:
    image = img_rgb.copy()
    height, width = image.shape[:2]
    padding = 4
    box_height = 18

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (
            0,
            height - box_height - 2,
        ),
        (
            width,
            height,
        ),
        (
            255,
            255,
            255,
        ),
        thickness=-1,
    )

    image = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0.0,
    )

    cv2.putText(
        image,
        caption,
        (
            6,
            height - 6,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            0,
            0,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    return image


def make_mosaic(
    rows: List[Dict[str, str]],
    out_path: str,
    cols=5,
    cell=(224, 224),
    kind="overlay",
):
    if not rows:
        return

    images = []

    for row in rows:
        image_path = row.get(
            (
                "out_overlay"
                if kind == "overlay"
                else "out_original"
            ),
            "",
        )

        image_rgb = load_rgb(
            image_path,
            size=cell,
        )

        if image_rgb is None:
            continue

        caption = (
            f"y={row['y_true']}  "
            f"ŷ={row['y_pred']}  "
            f"F={row['F_score']:.2f}  "
            f"in={row['energy_inside']:.2f}  "
            f"b={row['border_ratio']:.2f}"
        )

        image_rgb = draw_caption(
            image_rgb,
            caption,
        )
        images.append(
            image_rgb
        )

    if not images:
        return

    cell_height, cell_width = (
        cell[1],
        cell[0],
    )
    row_count = int(
        math.ceil(
            len(images)
            / cols
        )
    )

    canvas = (
        np.ones(
            (
                row_count * cell_height,
                cols * cell_width,
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )

    for image_index, image in enumerate(
        images
    ):
        row_index = (
            image_index
            // cols
        )
        column_index = (
            image_index
            % cols
        )

        canvas[
            row_index * cell_height:
            (row_index + 1) * cell_height,
            column_index * cell_width:
            (column_index + 1) * cell_width,
            :,
        ] = image

    os.makedirs(
        os.path.dirname(
            out_path
        ),
        exist_ok=True,
    )

    cv2.imwrite(
        out_path,
        cv2.cvtColor(
            canvas,
            cv2.COLOR_RGB2BGR,
        ),
    )


def write_html(
    exp_dir: str,
    merged_rows: List[Dict[str, str]],
    agg_path: str,
    mosaics: Dict[str, str],
    out_html: str,
):
    os.makedirs(
        os.path.dirname(
            out_html
        ),
        exist_ok=True,
    )

    generated_at = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )
    experiment_title = os.path.basename(
        exp_dir.rstrip(
            "/\\"
        )
    )

    aggregate_table = ""

    if os.path.exists(
        agg_path
    ):
        aggregate_rows = read_csv(
            agg_path
        )
        sections = {
            "global": [],
            "class": [],
            "dataset": [],
        }

        for aggregate_row in aggregate_rows:
            sections[
                aggregate_row["group"]
            ].append(
                aggregate_row
            )

        def render_section(
            name,
            rows,
        ):
            if not rows:
                return ""

            table_header = (
                "<tr><th>key</th><th>n</th><th>acc</th>"
                "<th>AUC_ins</th><th>AUC_del</th><th>F</th>"
                "<th>inside</th><th>border</th></tr>"
            )
            table_rows = []

            for row in rows:
                table_rows.append(
                    f"<tr><td>{row['key']}</td>"
                    f"<td>{row['n']}</td>"
                    f"<td>{float(row['acc']):.3f}</td>"
                    f"<td>{float(row['auc_ins']):.3f}</td>"
                    f"<td>{float(row['auc_del']):.3f}</td>"
                    f"<td>{float(row['F']):.3f}</td>"
                    f"<td>{float(row['energy_inside']):.3f}</td>"
                    f"<td>{float(row['border_ratio']):.3f}</td></tr>"
                )

            return (
                f"<h3>{name}</h3>"
                f"<table class='t'>"
                f"{table_header}"
                f"{''.join(table_rows)}"
                f"</table>"
            )

        aggregate_table = (
            render_section(
                "Global",
                sections["global"],
            )
            + render_section(
                "By true class",
                sections["class"],
            )
            + render_section(
                "By dataset",
                sections["dataset"],
            )
        )

    top_fidelity = select_top(
        merged_rows,
        20,
        key=lambda row: row["F_score"],
        reverse=True,
    )
    lowest_fidelity = select_top(
        merged_rows,
        20,
        key=lambda row: row["F_score"],
        reverse=False,
    )

    def rows_to_html(
        rows,
    ):
        table_header = (
            "<tr><th>y</th><th>ŷ</th><th>prob</th>"
            "<th>F</th><th>inside</th><th>border</th>"
            "<th>overlay</th></tr>"
        )
        table_rows = []

        for row in rows:
            overlay_link = row.get(
                "out_overlay",
                "",
            )
            table_rows.append(
                f"<tr><td>{row['y_true']}</td>"
                f"<td>{row['y_pred']}</td>"
                f"<td>{row['prob_pred']:.3f}</td>"
                f"<td>{row['F_score']:.3f}</td>"
                f"<td>{row['energy_inside']:.3f}</td>"
                f"<td>{row['border_ratio']:.3f}</td>"
                f"<td><a href='{relpath(overlay_link, out_html)}'>view</a></td></tr>"
            )

        return (
            f"<table class='t'>"
            f"{table_header}"
            f"{''.join(table_rows)}"
            f"</table>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Grad-CAM triage — {experiment_title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1 {{ margin: 0 0 8px 0; }}
small {{ color:#666; }}
section {{ margin: 24px 0; }}
.t {{ border-collapse: collapse; font-size: 14px; }}
.t th, .t td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 13px; color:#333; margin-top: 6px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #eee; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
.badge {{ display:inline-block; background:#eef; padding:2px 6px; border-radius:6px; font-size:12px; margin-left:6px; }}
</style>
</head>
<body>
<h1>Grad-CAM — Explanation Triage <span class="badge">{experiment_title}</span></h1>
<small>Generated: {generated_at}</small>

<section>
  <h2>Aggregate Summary</h2>
  {aggregate_table if aggregate_table else "<p><i>metrics_agg.csv was not found</i></p>"}
</section>

<section>
  <h2>Mosaics</h2>
  <div class="grid">
    {figure_img("Clean correct predictions", mosaics.get("clean",""))}
    {figure_img("Errors with coherent attention", mosaics.get("err_coherent",""))}
    {figure_img("Errors with spurious attention", mosaics.get("err_spurious",""))}
    {figure_img("Borderline cases", mosaics.get("borderline",""))}
  </div>
</section>

<section>
  <h2>Top 20 by F</h2>
  {rows_to_html(top_fidelity)}
</section>

<section>
  <h2>Bottom 20 by F</h2>
  {rows_to_html(lowest_fidelity)}
</section>

<section>
  <h2>Notes</h2>
  <ul>
    <li><b>F</b>: fidelity score based on insertion/deletion relative to a random control. Higher is better.</li>
    <li><b>inside</b>: proportion of heatmap energy inside the retinal mask.</li>
    <li><b>border</b>: energy in the peripheral ring, used as a border-bias indicator. Lower is better.</li>
    <li>Click “view” to open the overlay for each case.</li>
  </ul>
</section>

</body>
</html>
"""

    with open(
        out_html,
        "w",
        encoding="utf-8",
    ) as html_file:
        html_file.write(
            html
        )

    print(
        "[DONE] Report written to",
        out_html,
    )


def relpath(
    path: str,
    target_html: str,
) -> str:
    try:
        base_directory = os.path.dirname(
            target_html
        )
        return os.path.relpath(
            path,
            start=base_directory,
        )
    except Exception:
        return path


def figure_img(
    caption: str,
    img_path: str,
) -> str:
    if (
        not img_path
        or not os.path.exists(
            img_path
        )
    ):
        return (
            f"<figure><figcaption>"
            f"{caption} (no data)"
            f"</figcaption></figure>"
        )

    relative_path = relpath(
        img_path,
        os.path.join(
            os.path.dirname(
                img_path
            ),
            "..",
            "..",
            "..",
            "eval_reports",
            "dummy.html",
        ),
    )

    return (
        f"<figure>"
        f"<img src='{relative_path}'/>"
        f"<figcaption>{caption}</figcaption>"
        f"</figure>"
    )


def build_triage(
    exp_dir: str,
    top_k: int = 20,
):
    gradcam_dir = os.path.join(
        exp_dir,
        "gradcam",
    )
    overlay_dir = os.path.join(
        gradcam_dir,
        "overlays",
    )
    quant_dir = os.path.join(
        gradcam_dir,
        "quant",
    )
    results_csv = os.path.join(
        gradcam_dir,
        "gradcam_results.csv",
    )
    per_image_csv = os.path.join(
        quant_dir,
        "metrics_per_image.csv",
    )
    aggregate_csv = os.path.join(
        quant_dir,
        "metrics_agg.csv",
    )
    output_html = os.path.join(
        exp_dir.replace(
            "experiments" + os.sep,
            "eval_reports" + os.sep,
        ),
        f"{os.path.basename(exp_dir)}_gradcam_summary.html",
    )

    os.makedirs(
        os.path.dirname(
            output_html
        ),
        exist_ok=True,
    )

    result_rows = read_csv(
        results_csv
    )
    metric_rows = read_csv(
        per_image_csv
    )
    merged_rows = merge_rows(
        result_rows,
        metric_rows,
    )

    high_fidelity_threshold = (
        np.percentile(
            [
                row["F_score"]
                for row in merged_rows
            ],
            70,
        )
        if merged_rows
        else 0.0
    )
    low_fidelity_threshold = (
        np.percentile(
            [
                row["F_score"]
                for row in merged_rows
            ],
            30,
        )
        if merged_rows
        else 0.0
    )
    inside_energy_threshold = 0.85
    border_ratio_threshold = 0.20

    clean_cases = select_top(
        merged_rows,
        top_k,
        where=lambda row: (
            row["correct"] == 1
            and row["F_score"] >= high_fidelity_threshold
            and row["energy_inside"] >= inside_energy_threshold
            and row["border_ratio"] < border_ratio_threshold
        ),
        key=lambda row: (
            row["F_score"],
            row["energy_inside"],
        ),
    )

    coherent_error_cases = select_top(
        merged_rows,
        top_k,
        where=lambda row: (
            row["correct"] == 0
            and row["F_score"] >= high_fidelity_threshold
            and row["energy_inside"] >= inside_energy_threshold
            and row["border_ratio"] < border_ratio_threshold
        ),
        key=lambda row: row["F_score"],
    )

    spurious_error_cases = select_top(
        merged_rows,
        top_k,
        where=lambda row: (
            row["correct"] == 0
            and (
                row["F_score"] <= low_fidelity_threshold
                or row["energy_inside"] < inside_energy_threshold
                or row["border_ratio"] >= border_ratio_threshold
            )
        ),
        key=lambda row: (
            row["energy_inside"],
            -row["border_ratio"],
        ),
        reverse=False,
    )

    borderline_cases = select_top(
        merged_rows,
        top_k,
        where=lambda row: (
            row["F_score"] > low_fidelity_threshold
            and row["F_score"] < high_fidelity_threshold
        ),
        key=lambda row: abs(
            row["F_score"] - 0.5
        ),
    )

    mosaic_dir = os.path.join(
        quant_dir,
        "",
    )
    clean_mosaic_path = os.path.join(
        mosaic_dir,
        "mosaic_clean.png",
    )
    coherent_error_mosaic_path = os.path.join(
        mosaic_dir,
        "mosaic_err_coherent.png",
    )
    spurious_error_mosaic_path = os.path.join(
        mosaic_dir,
        "mosaic_err_spurious.png",
    )
    borderline_mosaic_path = os.path.join(
        mosaic_dir,
        "mosaic_borderline.png",
    )

    make_mosaic(
        clean_cases,
        clean_mosaic_path,
        cols=5,
        cell=(224, 224),
        kind="overlay",
    )
    make_mosaic(
        coherent_error_cases,
        coherent_error_mosaic_path,
        cols=5,
        cell=(224, 224),
        kind="overlay",
    )
    make_mosaic(
        spurious_error_cases,
        spurious_error_mosaic_path,
        cols=5,
        cell=(224, 224),
        kind="overlay",
    )
    make_mosaic(
        borderline_cases,
        borderline_mosaic_path,
        cols=5,
        cell=(224, 224),
        kind="overlay",
    )

    mosaics = dict(
        clean=clean_mosaic_path,
        err_coherent=coherent_error_mosaic_path,
        err_spurious=spurious_error_mosaic_path,
        borderline=borderline_mosaic_path,
    )

    write_html(
        exp_dir,
        merged_rows,
        aggregate_csv,
        mosaics,
        output_html,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "Grad-CAM triage report (HTML + mosaics)"
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
        help="experiments/exp_YYYYMMDD_HHMM",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of examples per mosaic",
    )

    args = parser.parse_args()

    build_triage(
        args.exp_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
