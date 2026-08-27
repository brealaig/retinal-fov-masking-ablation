from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VALID_VARIANTS = ["no_mask", "hard_mask", "fixed_feather", "adaptive_feather"]
EXP_RE = re.compile(
    r"(?P<dataset>.+)_(?P<variant>no_mask|hard_mask|fixed_feather|adaptive_feather)_seed(?P<seed>\d+)$"
)
REQUIRED_COLUMNS = [
    "filename",
    "true_label",
    "pred_label",
    "confidence",
    "correct",
    "inside_energy",
    "outside_energy",
    "border_ratio",
    "peripheral_attention",
]

PATTERN_LABELS = {
    "retinal_aligned": "Predominantly retinal attention",
    "extraretinal_elevated": "Elevated extraretinal attention",
    "extraretinal_dominant": "Dominant extraretinal attention",
    "border_dominant": "Potential border dependence",
    "mixed_extraretinal_border": "Mixed pattern: background and border",
    "peripheral_dominant": "Elevated peripheral attention",
    "mixed_or_diffuse": "Mixed or diffuse attention",
    "insufficient_metrics": "Insufficient metrics",
}

def sanitize_dataset_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "DATASET").strip()).strip("._-")
    return value or "DATASET"


SHORTCUT_PATTERNS = {
    "extraretinal_elevated",
    "extraretinal_dominant",
    "border_dominant",
    "mixed_extraretinal_border",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def _safe_quantile(series: pd.Series, q: float, fallback: float) -> float:
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric_values) == 0:
        return float(fallback)
    return float(numeric_values.quantile(q))


def _clip01(value: Any) -> float:
    try:
        numeric_value = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(numeric_value):
        return float("nan")
    return float(np.clip(numeric_value, 0.0, 1.0))


def read_experiment_config(exp_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(exp_dir, "experiment_config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except Exception:
        return {}


def find_overlay_path(exp_dir: str, split: str, filename: str, true_label: int, pred_label: int, correct: bool) -> str:
    status = "correct" if correct else "incorrect"
    filename_stem = os.path.splitext(os.path.basename(str(filename)))[0]
    overlay_dir = Path(exp_dir) / "gradcam" / split / status
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = overlay_dir / f"{filename_stem}_t{int(true_label)}_p{int(pred_label)}{extension}"
        if candidate.is_file():
            return str(candidate)
    return ""


def load_gradcam_rows(exp_root: str, split: str, variants: Sequence[str], source_dataset_name: Optional[str] = None) -> pd.DataFrame:
    dataframes: List[pd.DataFrame] = []
    exp_root_path = Path(exp_root)
    if not exp_root_path.is_dir():
        raise FileNotFoundError(f"exp_root does not exist: {exp_root}")

    for experiment_path in sorted(exp_root_path.iterdir()):
        if not experiment_path.is_dir():
            continue
        experiment_match = EXP_RE.match(experiment_path.name)
        if not experiment_match:
            continue
        variant = experiment_match.group("variant")
        experiment_config = read_experiment_config(str(experiment_path))
        dataset_name = str(experiment_config.get("dataset_name", experiment_config.get("dataset", experiment_match.group("dataset"))))
        if source_dataset_name and sanitize_dataset_name(dataset_name) != sanitize_dataset_name(source_dataset_name):
            continue
        if variant not in variants:
            continue
        seed = int(experiment_match.group("seed"))
        csv_path = experiment_path / f"gradcam_summary_{split}.csv"
        if not csv_path.is_file():
            continue

        gradcam_df = pd.read_csv(csv_path)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in gradcam_df.columns]
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {missing_columns}")

        gradcam_df = gradcam_df.copy()
        gradcam_df["variant"] = variant
        gradcam_df["seed"] = seed
        gradcam_df["split"] = split
        gradcam_df["experiment_dir"] = str(experiment_path)
        gradcam_df["source_csv"] = str(csv_path)
        gradcam_df["dataset"] = dataset_name
        gradcam_df["architecture"] = str(experiment_config.get("architecture", "EfficientNetB0"))
        gradcam_df["correct"] = gradcam_df["correct"].map(_to_bool)
        for column in [
            "true_label",
            "pred_label",
            "confidence",
            "inside_energy",
            "outside_energy",
            "border_ratio",
            "peripheral_attention",
            "p50",
        ]:
            if column in gradcam_df.columns:
                gradcam_df[column] = pd.to_numeric(gradcam_df[column], errors="coerce")

        gradcam_df["overlay_path"] = [
            find_overlay_path(
                str(experiment_path), split, record.filename, int(record.true_label), int(record.pred_label), bool(record.correct)
            )
            for record in gradcam_df.itertuples(index=False)
        ]
        dataframes.append(gradcam_df)

    if not dataframes:
        expected_filename = f"gradcam_summary_{split}.csv"
        raise FileNotFoundError(
            f"No files {expected_filename} were found within valid experiments in: {exp_root}"
        )
    return pd.concat(dataframes, ignore_index=True)


def derive_thresholds(
    df: pd.DataFrame,
    outside_high: Optional[float] = None,
    outside_dominant: Optional[float] = None,
    border_high: Optional[float] = None,
    peripheral_high: Optional[float] = None,
) -> Dict[str, float]:
                                                                                    
                                                                         
    pooled_count = len(df)
    outside_q75 = _safe_quantile(df["outside_energy"], 0.75, 0.35)
    outside_q90 = _safe_quantile(df["outside_energy"], 0.90, 0.50)
    border_q75 = _safe_quantile(df["border_ratio"], 0.75, 0.40)
    peripheral_q75 = _safe_quantile(df["peripheral_attention"], 0.75, 0.40)

    thresholds = {
        "outside_high": float(outside_high if outside_high is not None else min(0.50, max(0.35, outside_q75))),
        "outside_dominant": float(outside_dominant if outside_dominant is not None else min(0.70, max(0.50, outside_q90))),
        "border_high": float(border_high if border_high is not None else min(0.65, max(0.40, border_q75))),
        "peripheral_high": float(peripheral_high if peripheral_high is not None else min(0.65, max(0.40, peripheral_q75))),
        "central_retinal_median": _safe_quantile(
            pd.to_numeric(df["inside_energy"], errors="coerce")
            * (1.0 - pd.to_numeric(df["border_ratio"], errors="coerce")),
            0.50,
            0.30,
        ),
        "pooled_n": int(pooled_count),
        "method": "pooled_hybrid_absolute_plus_quantile",
    }
                             
    thresholds["outside_dominant"] = max(thresholds["outside_dominant"], thresholds["outside_high"])
    return thresholds


def classify_pattern(row: pd.Series, thresholds: Dict[str, float]) -> str:
    outside = _clip01(row.get("outside_energy"))
    border = _clip01(row.get("border_ratio"))
    peripheral = _clip01(row.get("peripheral_attention"))
    central = _clip01(row.get("central_retinal_energy"))
    if any(np.isnan(x) for x in [outside, border, peripheral, central]):
        return "insufficient_metrics"
    if outside >= thresholds["outside_dominant"]:
        return "extraretinal_dominant"
    if outside >= thresholds["outside_high"] and border >= thresholds["border_high"]:
        return "mixed_extraretinal_border"
    if outside >= thresholds["outside_high"]:
        return "extraretinal_elevated"
    if border >= thresholds["border_high"]:
        return "border_dominant"
    if peripheral >= thresholds["peripheral_high"]:
        return "peripheral_dominant"
    if central >= thresholds["central_retinal_median"]:
        return "retinal_aligned"
    return "mixed_or_diffuse"


def _variant_reason(variant: str, pattern: str) -> str:
    if pattern in {"extraretinal_elevated", "extraretinal_dominant", "mixed_extraretinal_border"}:
        if variant == "no_mask":
            return (
                "The image retains background, illumination, framing, and possible acquisition marks. "
                "The model may be exploiting contextual correlations that are not retinal lesions."
            )
        if variant == "hard_mask":
            return (
                "Although the mask removes part of the background, the abrupt cutoff creates a high-contrast transition "
                "that may become a stable visual cue."
            )
        if variant == "fixed_feather":
            return (
                "The fixed transition reduces the abrupt cutoff, but it may not adapt to all brightness levels or fields of view; "
                "peripheral gradients correlated with the class may persist."
            )
        if variant == "adaptive_feather":
            return (
                "The adaptive transition reduces border artifacts, but persistent extraretinal activation suggests "
                "that framing, image quality, illumination, or limitations of Grad-CAM itself may also be involved."
            )
    if pattern == "border_dominant":
        if variant == "hard_mask":
            return "The hard border is a strong discontinuity and may act as a segmentation or acquisition shortcut."
        return "Activation is concentrated near the retina-background transition; it may reflect residual border effects, cropping, or peripheral illumination."
    if pattern == "peripheral_dominant":
        return "The periphery may contain real lesions, but also vignetting, blur, illumination changes, or camera artifacts."
    if pattern == "retinal_aligned":
        return "Most of the energy remains within the retina and away from the border, a pattern more consistent with anatomical evidence."
    if pattern == "mixed_or_diffuse":
        return "Attention does not show a dominant location; it may correspond to global evidence, absence of lesions, or low spatial specificity."
    return "There are not enough metrics to provide a stable interpretation."


def _interpretation(row: pd.Series) -> Tuple[str, str, str]:
    pattern = str(row["attention_pattern"])
    correct = bool(row["correct"])
    true_label = int(row["true_label"]) if pd.notna(row["true_label"]) else -1
    variant = str(row["variant"])
    suspected = pattern in SHORTCUT_PATTERNS

    if correct and suspected:
        status = "Correct prediction with a possible visual shortcut"
    elif (not correct) and suspected:
        status = "Incorrect prediction with possible visual shortcut influence"
    elif correct and pattern == "retinal_aligned":
        status = "Correct prediction with plausible anatomical alignment"
    elif not correct and pattern == "retinal_aligned":
        status = "Incorrect prediction despite retinal attention"
    elif correct:
        status = "Correct prediction with mixed or peripheral attention"
    else:
        status = "Incorrect prediction with mixed or peripheral attention"

    reason = _variant_reason(variant, pattern)
    if true_label == 0 and correct and pattern in {
        "mixed_or_diffuse",
        "peripheral_dominant",
        "extraretinal_elevated",
        "extraretinal_dominant",
    }:
        reason += (
            " In class 0, there is no mandatory positive lesion to localize; the model may use global evidence of absence. "
            "This explains part of the diffuse attention, but does not rule out learning from background or acquisition cues."
        )

    if suspected:
        indication = (
            "Do not discard automatically. Prioritize review of the original image, compare the same filename across variants, "
            "review the raw map, and quantify whether the pattern repeats by class, brightness, or camera. "
            "Report it as a 'suspected shortcut', not as demonstrated causality."
        )
    elif pattern == "retinal_aligned":
        indication = (
            "A favorable case for qualitative analysis. Even so, verify that the highlighted region corresponds to plausible lesions or structures; "
            "being inside the retina does not guarantee clinical specificity."
        )
    elif pattern == "peripheral_dominant":
        indication = (
            "Check whether real peripheral lesions are present. If not, inspect vignetting, blur, cropping, and the mask transition."
        )
    else:
        indication = (
            "Interpret with caution and complement the analysis with raw maps, another attribution technique, and review of repeated cases."
        )
    return status, reason, indication


def enrich_rows(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    enriched_df = df.copy()
    inside_energy = pd.to_numeric(enriched_df["inside_energy"], errors="coerce").clip(0, 1)
    outside_energy = pd.to_numeric(enriched_df["outside_energy"], errors="coerce").clip(0, 1)
    border_ratio = pd.to_numeric(enriched_df["border_ratio"], errors="coerce").clip(0, 1)
    peripheral_attention = pd.to_numeric(enriched_df["peripheral_attention"], errors="coerce").clip(0, 1)

    enriched_df["central_retinal_energy"] = (inside_energy * (1.0 - border_ratio)).clip(0, 1)
    enriched_df["outside_to_inside_ratio"] = outside_energy / (inside_energy + 1e-8)
    enriched_df["shortcut_risk_score"] = (0.55 * outside_energy + 0.30 * border_ratio + 0.15 * peripheral_attention).clip(0, 1)
    enriched_df["attention_pattern"] = enriched_df.apply(lambda row: classify_pattern(row, thresholds), axis=1)
    enriched_df["attention_pattern_label"] = enriched_df["attention_pattern"].map(PATTERN_LABELS).fillna(enriched_df["attention_pattern"])
    enriched_df["shortcut_suspected"] = enriched_df["attention_pattern"].isin(SHORTCUT_PATTERNS)

    interpretations = enriched_df.apply(_interpretation, axis=1, result_type="expand")
    interpretations.columns = ["interpretation_status", "possible_reason", "indication"]
    enriched_df = pd.concat([enriched_df, interpretations], axis=1)
    return enriched_df


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean())


def summarize_group(group: pd.DataFrame) -> Dict[str, Any]:
    correct_mask = group["correct"].astype(bool)
    shortcut_mask = group["shortcut_suspected"].astype(bool)
    aligned_mask = group["attention_pattern"].eq("retinal_aligned")
    correct_rows = group[correct_mask]
    return {
        "n": int(len(group)),
        "correct_rate": _rate(correct_mask),
        "inside_energy_mean": float(group["inside_energy"].mean()),
        "outside_energy_mean": float(group["outside_energy"].mean()),
        "outside_energy_median": float(group["outside_energy"].median()),
        "border_ratio_mean": float(group["border_ratio"].mean()),
        "peripheral_attention_mean": float(group["peripheral_attention"].mean()),
        "central_retinal_energy_mean": float(group["central_retinal_energy"].mean()),
        "shortcut_risk_score_mean": float(group["shortcut_risk_score"].mean()),
        "shortcut_suspected_rate": _rate(shortcut_mask),
        "retinal_aligned_rate": _rate(aligned_mask),
        "correct_shortcut_count": int((correct_mask & shortcut_mask).sum()),
        "correct_shortcut_rate_all": float((correct_mask & shortcut_mask).mean()) if len(group) else float("nan"),
        "correct_shortcut_rate_among_correct": float(correct_rows["shortcut_suspected"].mean()) if len(correct_rows) else float("nan"),
        "correct_aligned_rate_among_correct": float(correct_rows["attention_pattern"].eq("retinal_aligned").mean()) if len(correct_rows) else float("nan"),
    }


def grouped_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    grouper: Any = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    for keys, group in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        summary_row = {column: key for column, key in zip(group_cols, keys)}
        summary_row.update(summarize_group(group))
        summary_rows.append(summary_row)
    return pd.DataFrame(summary_rows)


def _pct(value: Any) -> str:
    try:
        if not np.isfinite(float(value)):
            return "N/A"
        return f"{100.0 * float(value):.1f}%"
    except Exception:
        return "N/A"


def build_markdown_report(
    df: pd.DataFrame,
    summary_variant: pd.DataFrame,
    thresholds: Dict[str, float],
    split: str,
) -> str:
    report_lines: List[str] = []
    report_lines.append(f"# Grad-CAM Alignment Report — {split}")
    report_lines.append("")
    report_lines.append(
        "This report classifies attention patterns to prioritize review. "
        "A correct prediction may appear suspicious if Grad-CAM energy falls outside the retina or on the border. "
        "This is compatible with shortcut learning, but does not prove it by itself."
    )
    report_lines.append("")
    report_lines.append("## Thresholds used")
    report_lines.append("")
    report_lines.append(
        f"- Elevated extraretinal attention: outside_energy ≥ {thresholds['outside_high']:.3f}.\n"
        f"- Dominant extraretinal attention: outside_energy ≥ {thresholds['outside_dominant']:.3f}.\n"
        f"- Elevated border attention: border_ratio ≥ {thresholds['border_high']:.3f}.\n"
        f"- Elevated peripheral attention: peripheral_attention ≥ {thresholds['peripheral_high']:.3f}."
    )
    report_lines.append("")
    report_lines.append("Automatic thresholds are computed jointly across all variants in the split to preserve comparability.")
    report_lines.append("")
    report_lines.append("## Summary by variant")
    report_lines.append("")
    if summary_variant.empty:
        report_lines.append("There are not enough data.")
    else:
        for row in summary_variant.sort_values("variant").itertuples(index=False):
            report_lines.append(
                f"- **{row.variant}**: outside={row.outside_energy_mean:.3f}, "
                f"border={row.border_ratio_mean:.3f}, central={row.central_retinal_energy_mean:.3f}, "
                f"suspected shortcut={_pct(row.shortcut_suspected_rate)}, "
                f"suspected shortcuts among correct predictions={_pct(row.correct_shortcut_rate_among_correct)}."
            )
    report_lines.append("")
    report_lines.append("## Comparative interpretation")
    report_lines.append("")

    if not summary_variant.empty:
        summary_indexed = summary_variant.set_index("variant")
        best_out = summary_variant.loc[summary_variant["outside_energy_mean"].idxmin(), "variant"]
        best_border = summary_variant.loc[summary_variant["border_ratio_mean"].idxmin(), "variant"]
        best_central = summary_variant.loc[summary_variant["central_retinal_energy_mean"].idxmax(), "variant"]
        report_lines.append(f"- Lowest mean extraretinal energy: **{best_out}**.")
        report_lines.append(f"- Lowest mean border dependence: **{best_border}**.")
        report_lines.append(f"- Highest central retinal energy: **{best_central}**.")

        if "no_mask" in summary_indexed.index and "hard_mask" in summary_indexed.index:
            no_mask_summary = summary_indexed.loc["no_mask"]
            hard_mask_summary = summary_indexed.loc["hard_mask"]
            if hard_mask_summary["outside_energy_mean"] < no_mask_summary["outside_energy_mean"] and hard_mask_summary["border_ratio_mean"] > no_mask_summary["border_ratio_mean"]:
                report_lines.append(
                    "- The hard mask reduces energy outside the retina compared with no_mask, but increases border concentration. "
                    "This pattern is compatible with replacing a background shortcut with a border shortcut."
                )
        if "hard_mask" in summary_indexed.index:
            hard_mask_summary = summary_indexed.loc["hard_mask"]
            for feather_variant in ["fixed_feather", "adaptive_feather"]:
                if feather_variant in summary_indexed.index:
                    feather_summary = summary_indexed.loc[feather_variant]
                    if (
                        feather_summary["outside_energy_mean"] < hard_mask_summary["outside_energy_mean"]
                        and feather_summary["border_ratio_mean"] < hard_mask_summary["border_ratio_mean"]
                    ):
                        report_lines.append(
                            f"- **{feather_variant}** simultaneously reduces energy outside the retina and border concentration compared with hard_mask, "
                            "which supports the hypothesis that a smooth transition reduces spurious peripheral cues."
                        )
        if "fixed_feather" in summary_indexed.index and "adaptive_feather" in summary_indexed.index:
            fixed_feather_summary = summary_indexed.loc["fixed_feather"]
            adaptive_feather_summary = summary_indexed.loc["adaptive_feather"]
            if (
                adaptive_feather_summary["outside_energy_mean"] < fixed_feather_summary["outside_energy_mean"]
                and adaptive_feather_summary["border_ratio_mean"] < fixed_feather_summary["border_ratio_mean"]
            ):
                report_lines.append(
                    "- Adaptive feathering improves both metrics compared with fixed feathering; this is compatible with better adaptation to illumination and the retinal field of view."
                )
            elif (
                adaptive_feather_summary["outside_energy_mean"] > fixed_feather_summary["outside_energy_mean"]
                and adaptive_feather_summary["border_ratio_mean"] > fixed_feather_summary["border_ratio_mean"]
            ):
                report_lines.append(
                    "- Adaptive feathering does not improve these metrics compared with fixed feathering in this split. Feather thresholds, brightness categories, and extreme cases should be reviewed."
                )

        if float(summary_variant["shortcut_suspected_rate"].min()) > 0.50:
            report_lines.append(
                "- More than half of the images are suspicious even in the best variant. This may indicate shared background/crop confounding, "
                "an imprecise approximate retinal mask, or the low spatial resolution inherent to Grad-CAM at top_conv."
            )

    correct_class_zero_rows = df[(df["true_label"] == 0) & (df["correct"])]
    if len(correct_class_zero_rows):
        class_zero_shortcut_rate = float(correct_class_zero_rows["shortcut_suspected"].mean())
        report_lines.append(
            f"- Among correctly predicted class 0 images, {_pct(class_zero_shortcut_rate)} show a suspicious pattern. Part of this may be related to global evidence of lesion absence; "
            "however, repeated activation over the background or border remains methodologically relevant."
        )

    report_lines.append("")
    report_lines.append("## Methodological guidance")
    report_lines.append("")
    report_lines.append(
        "1. Do not automatically remove images based on Grad-CAM. Use the flags to prioritize review.\n"
        "2. Compare the same filename across the four variants. A systematic shift from background to border or from border to retina is more informative than an isolated map.\n"
        "3. Separate the analysis by class, brightness, and correct/incorrect predictions. Class 0 requires special caution because there is no mandatory positive lesion.\n"
        "4. Report the proportion of correct predictions with a suspected shortcut; accuracy or QWK alone do not guarantee anatomically plausible reasoning.\n"
        "5. Confirm a sample of cases using raw maps, another attribution technique, and clinical review before claiming causality."
    )
    report_lines.append("")
    report_lines.append("## Limitations")
    report_lines.append("")
    report_lines.append(
        "Grad-CAM is post-hoc, depends on the selected layer, and has limited spatial resolution. "
        "The mask used to quantify inside/outside attention is approximate, especially for no_mask. "
        "Therefore, these indicators express plausibility and shortcut risk, not a definitive causal explanation."
    )
    return "\n".join(report_lines) + "\n"


def build_paired_filename_table(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [column for column in ["filename", "true_label", "brightness_category", "p50", "split", "seed"] if column in df.columns]
    value_cols = [
        "correct",
        "confidence",
        "inside_energy",
        "outside_energy",
        "border_ratio",
        "peripheral_attention",
        "central_retinal_energy",
        "shortcut_risk_score",
        "attention_pattern",
        "attention_pattern_label",
        "shortcut_suspected",
    ]
    value_cols = [column for column in value_cols if column in df.columns]
    if not id_cols or not value_cols:
        return pd.DataFrame()

    paired_table = df.pivot_table(
        index=id_cols,
        columns="variant",
        values=value_cols,
        aggfunc="first",
    )
    paired_table.columns = [f"{metric}_{variant}" for metric, variant in paired_table.columns]
    paired_table = paired_table.reset_index()

    available_variants = sorted(df["variant"].dropna().astype(str).unique().tolist())
    risk_cols = [f"shortcut_risk_score_{v}" for v in available_variants if f"shortcut_risk_score_{v}" in paired_table.columns]
    if risk_cols:
        risks = paired_table[risk_cols].apply(pd.to_numeric, errors="coerce")
        paired_table["paired_variants_available"] = risks.notna().sum(axis=1)
        paired_table["paired_complete"] = paired_table["paired_variants_available"] == len(available_variants)
        has_any = risks.notna().any(axis=1)
        paired_table["best_variant_by_risk"] = ""
        paired_table["worst_variant_by_risk"] = ""
        paired_table.loc[has_any, "best_variant_by_risk"] = risks.loc[has_any].idxmin(axis=1).str.replace("shortcut_risk_score_", "", regex=False)
        paired_table.loc[has_any, "worst_variant_by_risk"] = risks.loc[has_any].idxmax(axis=1).str.replace("shortcut_risk_score_", "", regex=False)
        paired_table["risk_range_between_variants"] = risks.max(axis=1) - risks.min(axis=1)
    if {"shortcut_risk_score_no_mask", "shortcut_risk_score_adaptive_feather"}.issubset(paired_table.columns):
        paired_table["risk_reduction_adaptive_vs_no_mask"] = (
            pd.to_numeric(paired_table["shortcut_risk_score_no_mask"], errors="coerce")
            - pd.to_numeric(paired_table["shortcut_risk_score_adaptive_feather"], errors="coerce")
        )
    if {"border_ratio_no_mask", "border_ratio_hard_mask"}.issubset(paired_table.columns):
        paired_table["border_increase_hard_vs_no_mask"] = (
            pd.to_numeric(paired_table["border_ratio_hard_mask"], errors="coerce")
            - pd.to_numeric(paired_table["border_ratio_no_mask"], errors="coerce")
        )
    return paired_table


def save_outputs(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    exp_root: str,
    split: str,
    out_dir: Optional[str] = None,
    source_dataset_name: Optional[str] = None,
) -> Dict[str, str]:
    if out_dir:
        output_dir = Path(out_dir)
    elif source_dataset_name:
        output_dir = Path(exp_root) / "_gradcam_alignment_analysis" / sanitize_dataset_name(source_dataset_name) / split
    else:
        output_dir = Path(exp_root) / "_gradcam_alignment_analysis" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image = output_dir / f"gradcam_alignment_per_image_{split}.csv"
    summary_variant = output_dir / f"gradcam_alignment_summary_by_variant_{split}.csv"
    summary_class = output_dir / f"gradcam_alignment_by_class_{split}.csv"
    summary_brightness = output_dir / f"gradcam_alignment_by_brightness_{split}.csv"
    summary_pattern = output_dir / f"gradcam_alignment_by_pattern_{split}.csv"
    summary_correctness = output_dir / f"gradcam_alignment_by_correctness_{split}.csv"
    paired_filenames = output_dir / f"gradcam_alignment_paired_filenames_{split}.csv"
    thresholds_path = output_dir / f"gradcam_alignment_thresholds_{split}.json"
    report_path = output_dir / f"gradcam_alignment_interpretation_{split}.md"

    sort_columns = ["shortcut_risk_score", "outside_energy", "border_ratio"]
    df.sort_values(sort_columns, ascending=False).to_csv(per_image, index=False)

    by_variant = grouped_summary(df, [column for column in ["dataset", "variant", "seed", "split"] if column in df.columns])
    by_class = grouped_summary(df, [column for column in ["dataset", "variant", "seed", "true_label"] if column in df.columns])
    by_brightness = grouped_summary(df, [column for column in ["dataset", "variant", "seed", "brightness_category"] if column in df.columns])
    by_pattern = (
        df.groupby([column for column in ["dataset", "variant", "seed", "attention_pattern", "attention_pattern_label"] if column in df.columns], dropna=False)
        .agg(n=("filename", "count"), correct_rate=("correct", "mean"), confidence_mean=("confidence", "mean"))
        .reset_index()
    )
    by_correctness = grouped_summary(df, [column for column in ["dataset", "variant", "seed", "correct"] if column in df.columns])
    paired = build_paired_filename_table(df)

    by_variant.to_csv(summary_variant, index=False)
    by_class.to_csv(summary_class, index=False)
    by_brightness.to_csv(summary_brightness, index=False)
    by_pattern.to_csv(summary_pattern, index=False)
    by_correctness.to_csv(summary_correctness, index=False)
    paired.to_csv(paired_filenames, index=False)
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump({**thresholds, "generated_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2, ensure_ascii=False)
    report = build_markdown_report(df, by_variant, thresholds, split)
    report_path.write_text(report, encoding="utf-8")

    return {
        "out_dir": str(output_dir),
        "per_image": str(per_image),
        "summary_by_variant": str(summary_variant),
        "summary_by_class": str(summary_class),
        "summary_by_brightness": str(summary_brightness),
        "summary_by_pattern": str(summary_pattern),
        "summary_by_correctness": str(summary_correctness),
        "paired_filenames": str(paired_filenames),
        "thresholds": str(thresholds_path),
        "report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyzes retinal alignment and possible shortcut learning from Grad-CAM CSV files."
    )
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--source_dataset_name", default=None, help="Filters experiments by source dataset.")
    parser.add_argument("--split", default="hold_out", choices=["val_eval", "int_test", "hold_out"])
    parser.add_argument("--variants", nargs="+", default=VALID_VARIANTS, choices=VALID_VARIANTS)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--outside_high", type=float, default=None)
    parser.add_argument("--outside_dominant", type=float, default=None)
    parser.add_argument("--border_high", type=float, default=None)
    parser.add_argument("--peripheral_high", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gradcam_rows = load_gradcam_rows(args.exp_root, args.split, args.variants, source_dataset_name=args.source_dataset_name)
    thresholds = derive_thresholds(
        gradcam_rows,
        outside_high=args.outside_high,
        outside_dominant=args.outside_dominant,
        border_high=args.border_high,
        peripheral_high=args.peripheral_high,
    )
    enriched_rows = enrich_rows(gradcam_rows, thresholds)
    output_paths = save_outputs(enriched_rows, thresholds, args.exp_root, args.split, args.out_dir, source_dataset_name=args.source_dataset_name)

    print("[GradCAM alignment] Analysis completed")
    print(f"[GradCAM alignment] n={len(enriched_rows)} | split={args.split} | variants={args.variants}")
    for key, value in output_paths.items():
        print(f"[GradCAM alignment] {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
