from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compute_device_utils import (
    add_device_cli_arguments,
    compact_device_label,
    configure_tensorflow_device,
    preconfigure_device_from_argv,
)

_PRE_DEVICE_REQUEST = preconfigure_device_from_argv(sys.argv[1:])

import numpy as np
import pandas as pd
import tensorflow as tf

from external_dataset_utils import (
    VALID_VARIANTS,
    read_external_labels,
    resolve_external_dataframe,
    sanitize_dataset_name,
)
from rd_on_the_fly_mask_ablation import MaskAblationConfig, build_eval_ds, set_global_seed
from train_mask_ablation import build_model_for_inference, try_load_weights
from gradcam_tools.batch_explain_mask_ablation import make_gradcam, overlay_and_save, quantify_cam


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generates Grad-CAM for any labeled external dataset."
    )
    parser.add_argument(
        "--model_exp_dir",
        required=True,
        help="Source experiment containing trained weights.",
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--external_dataset_root", required=True)
    parser.add_argument("--external_dataset_name", required=True)
    parser.add_argument("--external_labels_csv", required=True)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    parser.add_argument("--images_subdir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--layer_name", default="top_conv")
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 = all external images.",
    )
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--overlay_limit", type=int, default=100)
    parser.add_argument("--output_dir", default=None)
    add_device_cli_arguments(parser)
    return parser.parse_args(argv)


def run_external_gradcam(
    model: tf.keras.Model,
    cfg: MaskAblationConfig,
    external_df: pd.DataFrame,
    source_dataset: str,
    external_dataset: str,
    layer_name: str,
    max_images: int,
    save_overlays: bool,
    overlay_limit: int,
    out_dir: str,
    device_info: Dict[str, Any],
) -> pd.DataFrame:
    if max_images and max_images > 0:
        working_df = external_df.head(int(max_images)).copy().reset_index(drop=True)
    else:
        working_df = external_df.copy().reset_index(drop=True)

    evaluation_dataset = build_eval_ds(cfg, working_df)
    result_rows = []
    row_offset = 0
    saved_overlay_count = 0

    for image_batch, label_batch in evaluation_dataset:
        heatmaps, probabilities = make_gradcam(
            model,
            image_batch,
            class_index=None,
            layer_name=layer_name,
        )
        image_batch_np = image_batch.numpy()
        true_labels = label_batch.numpy().reshape(-1).astype(int)
        predicted_labels = np.argmax(probabilities, axis=1).astype(int)

        for batch_index in range(image_batch_np.shape[0]):
            metadata = working_df.iloc[row_offset + batch_index]
            cam_metrics = quantify_cam(
                heatmaps[batch_index],
                image_batch_np[batch_index],
            )
            result_row = {
                "source_dataset": source_dataset,
                "external_dataset": external_dataset,
                "filename": str(metadata["filename"]),
                "true_label": int(true_labels[batch_index]),
                "pred_label": int(predicted_labels[batch_index]),
                "confidence": float(np.max(probabilities[batch_index])),
                "correct": bool(
                    true_labels[batch_index] == predicted_labels[batch_index]
                ),
                "usage": "external_test",
                "brightness_category": str(
                    metadata.get("brightness_category", "unknown")
                ),
                "p50": metadata.get("p50", np.nan),
                "variant": cfg.variant,
                "seed": int(cfg.seed),
                "abs_path": str(metadata.get("abs_path", "")),
                "path_resolution_mode": str(
                    metadata.get("path_resolution_mode", "")
                ),
                "compute_device": str(
                    device_info.get("selected_device", "unknown")
                ),
                "compute_device_name": str(
                    device_info.get("selected_device_name", "unknown")
                ),
            }
            result_row.update(cam_metrics)
            result_rows.append(result_row)

            if save_overlays and saved_overlay_count < int(overlay_limit):
                prediction_status = (
                    "correct" if result_row["correct"] else "incorrect"
                )
                overlay_path = os.path.join(
                    out_dir,
                    "overlays",
                    prediction_status,
                    f"{os.path.splitext(result_row['filename'])[0]}_t{result_row['true_label']}_p{result_row['pred_label']}.png",
                )
                overlay_and_save(
                    image_batch_np[batch_index],
                    heatmaps[batch_index],
                    overlay_path,
                )
                saved_overlay_count += 1

        row_offset += image_batch_np.shape[0]

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(
        os.path.join(out_dir, "gradcam_summary_external.csv"),
        index=False,
    )

    metric_columns = [
        "inside_energy",
        "outside_energy",
        "border_ratio",
        "peripheral_attention",
        "central_retinal_energy",
        "outside_to_inside_ratio",
        "retina_area_ratio",
        "border_area_ratio",
    ]
    metric_columns = [
        column
        for column in metric_columns
        if column in results_df.columns
    ]

    if len(results_df):
        results_df.groupby("true_label")[metric_columns].mean().reset_index().to_csv(
            os.path.join(out_dir, "gradcam_by_class_external.csv"),
            index=False,
        )
        results_df.groupby("brightness_category")[metric_columns].mean().reset_index().to_csv(
            os.path.join(out_dir, "gradcam_by_brightness_external.csv"),
            index=False,
        )
        results_df.groupby("correct")[metric_columns].mean().reset_index().to_csv(
            os.path.join(out_dir, "gradcam_by_correctness_external.csv"),
            index=False,
        )

    return results_df


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    model_experiment_dir = os.path.abspath(
        os.path.expanduser(args.model_exp_dir)
    )
    source_config = read_json(
        os.path.join(model_experiment_dir, "experiment_config.json")
    )
    source_dataset = str(
        source_config.get(
            "dataset_name",
            source_config.get("dataset", "unknown"),
        )
    )

    seed = int(
        args.seed
        if args.seed is not None
        else source_config.get("seed", 42)
    )

    source_image_size = source_config.get("img_size", [224, 224])
    inferred_image_size = (
        int(source_image_size[0])
        if isinstance(source_image_size, (list, tuple)) and source_image_size
        else 224
    )
    image_size = int(args.img_size or inferred_image_size)
    set_global_seed(seed)

    device_info = configure_tensorflow_device(
        tf,
        mode=args.device,
        gpu_index=args.gpu_index,
        operation="gradcam_external",
    )

    external_dataset_name = str(args.external_dataset_name).strip()
    external_dataset_slug = sanitize_dataset_name(external_dataset_name)
    output_dir = args.output_dir or os.path.join(
        model_experiment_dir,
        "external_tests",
        external_dataset_slug,
        "gradcam",
    )
    os.makedirs(output_dir, exist_ok=True)

    external_labels = read_external_labels(
        args.external_labels_csv,
        external_dataset_name,
    )
    external_df, path_resolution_report = resolve_external_dataframe(
        dataset_root=args.external_dataset_root,
        variant=args.variant,
        labels_df=external_labels,
        images_subdir=args.images_subdir,
        duplicate_policy="error",
    )

    print(
        f"[EXTERNAL-GRADCAM] source={source_dataset} | "
        f"external={external_dataset_name} | "
        f"variant={args.variant} | n={len(external_df)}"
    )
    print(
        f"[EXTERNAL-GRADCAM] direct={path_resolution_report['resolved_direct']} | "
        f"recursive={path_resolution_report['resolved_recursive_exact']} | "
        f"basename={path_resolution_report['resolved_by_basename']}"
    )

    config = MaskAblationConfig(
        dataset_root=args.external_dataset_root,
        dataset_name=external_dataset_name,
        variant=args.variant,
        labels_csv=args.external_labels_csv,
        exp_dir=output_dir,
        seed=seed,
        img_size=(image_size, image_size),
        batch_size=int(args.batch_size),
        sampling_policy="natural",
        augmentation_policy="none",
        loss_policy="ce_label_smoothing",
        enable_clahe=False,
        require_usage_column=False,
    )

    model = build_model_for_inference(
        num_classes=5,
        img_size=config.img_size,
    )
    weights_path = args.weights or os.path.join(
        model_experiment_dir,
        "final_all_phases.weights.h5",
    )

    if not try_load_weights(model, weights_path):
        raise FileNotFoundError(
            f"Could not load weights: {weights_path}"
        )

    results_df = run_external_gradcam(
        model=model,
        cfg=config,
        external_df=external_df,
        source_dataset=source_dataset,
        external_dataset=external_dataset_name,
        layer_name=args.layer_name,
        max_images=int(args.max_images),
        save_overlays=bool(args.save_overlays),
        overlay_limit=int(args.overlay_limit),
        out_dir=output_dir,
        device_info=device_info,
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": source_dataset,
        "external_dataset": external_dataset_name,
        "variant": args.variant,
        "seed": seed,
        "n": int(len(results_df)),
        "weights": weights_path,
        "layer_name": args.layer_name,
        "device": device_info,
        "path_resolution": path_resolution_report,
        "methodological_note": (
            "External Grad-CAM was computed without using the external dataset "
            "for training, calibration, or checkpoint selection. The map is "
            "generated for the model's predicted class."
        ),
    }

    with open(
        os.path.join(output_dir, "external_gradcam_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        f"[DONE] External Grad-CAM {external_dataset_name} | "
        f"variant={args.variant} | n={len(results_df)} | "
        f"device={compact_device_label(device_info)}"
    )
    print(f"[DONE] Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
