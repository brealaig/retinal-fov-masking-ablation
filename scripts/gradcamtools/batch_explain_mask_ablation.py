from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compute_device_utils import (
    add_device_cli_arguments,
    configure_tensorflow_device,
    preconfigure_device_from_argv,
)

_PRE_DEVICE_REQUEST = preconfigure_device_from_argv(sys.argv[1:])

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from rd_on_the_fly_mask_ablation import (
    MaskAblationConfig,
    VALID_VARIANTS,
    build_eval_ds,
    setup_logging,
    split_dataframes,
)
from train_mask_ablation import build_model_for_inference, try_load_weights


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam_array = np.asarray(cam, dtype=np.float32)
    cam_array = np.maximum(cam_array, 0)
    minimum_value, maximum_value = float(cam_array.min()), float(cam_array.max())
    if maximum_value - minimum_value < 1e-8:
        return np.zeros_like(cam_array, dtype=np.float32)
    return (cam_array - minimum_value) / (
        maximum_value - minimum_value + 1e-8
    )


def make_gradcam(
    model: tf.keras.Model,
    image_batch: tf.Tensor,
    class_index: Optional[np.ndarray] = None,
    layer_name: str = "top_conv",
) -> Tuple[np.ndarray, np.ndarray]:
    convolutional_layer = model.get_layer(layer_name)
    gradient_model = tf.keras.Model(
        model.inputs,
        [convolutional_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        convolutional_output, probabilities = gradient_model(
            image_batch,
            training=False,
        )
        if class_index is None:
            target_class_indices = tf.argmax(probabilities, axis=-1)
        else:
            target_class_indices = tf.convert_to_tensor(
                class_index,
                dtype=tf.int64,
            )

        gather_indices = tf.stack(
            [
                tf.range(
                    tf.shape(probabilities)[0],
                    dtype=tf.int64,
                ),
                target_class_indices,
            ],
            axis=1,
        )
        target_scores = tf.gather_nd(
            probabilities,
            gather_indices,
        )

    gradients = tape.gradient(
        target_scores,
        convolutional_output,
    )
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )
    heatmaps = tf.reduce_sum(
        convolutional_output * channel_weights[:, None, None, :],
        axis=-1,
    )
    heatmaps = heatmaps.numpy()
    probabilities_np = probabilities.numpy()

    resized_heatmaps = []
    for heatmap in heatmaps:
        normalized_heatmap = normalize_cam(heatmap)
        resized_heatmap = cv2.resize(
            normalized_heatmap,
            (image_batch.shape[2], image_batch.shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_heatmaps.append(
            normalize_cam(resized_heatmap)
        )

    return (
        np.stack(resized_heatmaps, axis=0).astype(np.float32),
        probabilities_np.astype(np.float32),
    )


def retinal_mask_from_rgb255(img_rgb255: np.ndarray) -> np.ndarray:
    image = np.clip(img_rgb255, 0, 255).astype(np.uint8)
    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )
    threshold = max(
        5,
        int(np.percentile(grayscale_image, 1)),
    )
    retinal_mask = (
        grayscale_image > threshold
    ).astype(np.uint8)

    morphology_kernel = np.ones(
        (5, 5),
        np.uint8,
    )
    retinal_mask = cv2.morphologyEx(
        retinal_mask,
        cv2.MORPH_OPEN,
        morphology_kernel,
    )
    retinal_mask = cv2.morphologyEx(
        retinal_mask,
        cv2.MORPH_CLOSE,
        morphology_kernel,
    )

    label_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            retinal_mask,
            connectivity=8,
        )
    )

    if label_count > 1:
        largest_component_label = 1 + np.argmax(
            component_stats[1:, cv2.CC_STAT_AREA]
        )
        retinal_mask = (
            component_labels == largest_component_label
        ).astype(np.uint8)

    return retinal_mask.astype(bool)


def quantify_cam(
    cam: np.ndarray,
    img_rgb255: np.ndarray,
    ring_width_px: int = 16,
) -> Dict[str, float]:
    normalized_cam = normalize_cam(cam)
    retinal_mask = retinal_mask_from_rgb255(img_rgb255)

    total_energy = float(
        normalized_cam.sum() + 1e-8
    )
    inside_energy = (
        float(
            normalized_cam[retinal_mask].sum()
            / total_energy
        )
        if np.any(retinal_mask)
        else np.nan
    )
    outside_energy = (
        float(
            normalized_cam[~retinal_mask].sum()
            / total_energy
        )
        if np.any(retinal_mask)
        else np.nan
    )
    retina_area_ratio = (
        float(np.mean(retinal_mask))
        if retinal_mask.size
        else np.nan
    )

    peak_y, peak_x = np.unravel_index(
        int(np.argmax(normalized_cam)),
        normalized_cam.shape,
    )
    cam_peak_inside = (
        bool(retinal_mask[peak_y, peak_x])
        if np.any(retinal_mask)
        else False
    )

    if np.any(retinal_mask):
        distance_map = cv2.distanceTransform(
            retinal_mask.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        border_mask = retinal_mask & (
            distance_map <= float(ring_width_px)
        )
        border_ratio = (
            float(
                normalized_cam[border_mask].sum()
                / (
                    normalized_cam[retinal_mask].sum()
                    + 1e-8
                )
            )
            if np.any(border_mask)
            else np.nan
        )

        retinal_distances = distance_map[retinal_mask]
        peripheral_cutoff = (
            np.percentile(retinal_distances, 25)
            if len(retinal_distances)
            else 0
        )
        peripheral_mask = retinal_mask & (
            distance_map <= peripheral_cutoff
        )
        peripheral_concentration = (
            float(
                normalized_cam[peripheral_mask].sum()
                / (
                    normalized_cam[retinal_mask].sum()
                    + 1e-8
                )
            )
            if np.any(peripheral_mask)
            else np.nan
        )
        border_area_ratio = (
            float(
                np.sum(border_mask)
                / (
                    np.sum(retinal_mask)
                    + 1e-8
                )
            )
            if np.any(border_mask)
            else np.nan
        )
    else:
        border_ratio = np.nan
        peripheral_concentration = np.nan
        border_area_ratio = np.nan

    central_retinal_energy = (
        float(
            inside_energy
            * (1.0 - border_ratio)
        )
        if np.isfinite(inside_energy)
        and np.isfinite(border_ratio)
        else np.nan
    )
    outside_to_inside_ratio = (
        float(
            outside_energy
            / (
                inside_energy
                + 1e-8
            )
        )
        if np.isfinite(outside_energy)
        and np.isfinite(inside_energy)
        else np.nan
    )

    return {
        "inside_energy": inside_energy,
        "outside_energy": outside_energy,
        "border_ratio": border_ratio,
        "peripheral_attention": peripheral_concentration,
        "central_retinal_energy": central_retinal_energy,
        "outside_to_inside_ratio": outside_to_inside_ratio,
        "retina_area_ratio": retina_area_ratio,
        "border_area_ratio": border_area_ratio,
        "cam_peak_inside": cam_peak_inside,
    }


def overlay_and_save(
    img_rgb255: np.ndarray,
    cam: np.ndarray,
    out_path: str,
) -> None:
    image = np.clip(
        img_rgb255,
        0,
        255,
    ).astype(np.uint8)

    heatmap = np.uint8(
        255 * normalize_cam(cam)
    )
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay_image = np.clip(
        0.55 * image
        + 0.45 * heatmap,
        0,
        255,
    ).astype(np.uint8)

    os.makedirs(
        os.path.dirname(out_path),
        exist_ok=True,
    )
    cv2.imwrite(
        out_path,
        cv2.cvtColor(
            overlay_image,
            cv2.COLOR_RGB2BGR,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM by variant for mask ablation."
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument(
        "--variant",
        required=True,
        choices=VALID_VARIANTS,
    )
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "val_eval",
            "int_test",
            "hold_out",
        ],
        choices=[
            "val_eval",
            "int_test",
            "hold_out",
        ],
    )
    parser.add_argument(
        "--layer_name",
        default="top_conv",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 = all images.",
    )
    parser.add_argument(
        "--save_overlays",
        action="store_true",
    )
    parser.add_argument(
        "--overlay_limit",
        type=int,
        default=60,
    )
    add_device_cli_arguments(parser)
    return parser.parse_args()


def run_split(
    model: tf.keras.Model,
    cfg: MaskAblationConfig,
    df_split: pd.DataFrame,
    split_name: str,
    layer_name: str,
    max_images: int,
    save_overlays: bool,
    overlay_limit: int,
) -> pd.DataFrame:
    if max_images and max_images > 0:
        working_df = (
            df_split.head(max_images)
            .copy()
            .reset_index(drop=True)
        )
    else:
        working_df = (
            df_split.copy()
            .reset_index(drop=True)
        )

    evaluation_dataset = build_eval_ds(
        cfg,
        working_df,
    )
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
        true_labels = (
            label_batch.numpy()
            .reshape(-1)
            .astype(int)
        )
        predicted_labels = np.argmax(
            probabilities,
            axis=1,
        ).astype(int)

        for batch_index in range(
            image_batch_np.shape[0]
        ):
            metadata = working_df.iloc[
                row_offset + batch_index
            ]
            cam_metrics = quantify_cam(
                heatmaps[batch_index],
                image_batch_np[batch_index],
            )

            result_row = {
                "dataset": cfg.dataset_name,
                "filename": str(
                    metadata["filename"]
                ),
                "true_label": int(
                    true_labels[batch_index]
                ),
                "pred_label": int(
                    predicted_labels[batch_index]
                ),
                "confidence": float(
                    np.max(
                        probabilities[batch_index]
                    )
                ),
                "correct": bool(
                    true_labels[batch_index]
                    == predicted_labels[batch_index]
                ),
                "usage": split_name,
                "brightness_category": str(
                    metadata.get(
                        "brightness_category",
                        "unknown",
                    )
                ),
                "p50": metadata.get(
                    "p50",
                    np.nan,
                ),
                "variant": cfg.variant,
                "seed": int(cfg.seed),
            }

            result_row.update(cam_metrics)
            result_rows.append(result_row)

            if (
                save_overlays
                and saved_overlay_count < overlay_limit
            ):
                prediction_status = (
                    "correct"
                    if result_row["correct"]
                    else "incorrect"
                )
                overlay_path = os.path.join(
                    cfg.exp_dir,
                    "gradcam",
                    split_name,
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

    results_df = pd.DataFrame(
        result_rows
    )
    summary_path = os.path.join(
        cfg.exp_dir,
        f"gradcam_summary_{split_name}.csv",
    )
    results_df.to_csv(
        summary_path,
        index=False,
    )

    if len(results_df):
        summary_metrics = [
            "inside_energy",
            "outside_energy",
            "border_ratio",
            "peripheral_attention",
        ]

        results_df.groupby(
            "true_label"
        )[summary_metrics].mean().reset_index().to_csv(
            os.path.join(
                cfg.exp_dir,
                f"gradcam_by_class_{split_name}.csv",
            ),
            index=False,
        )

        results_df.groupby(
            "brightness_category"
        )[summary_metrics].mean().reset_index().to_csv(
            os.path.join(
                cfg.exp_dir,
                f"gradcam_by_brightness_{split_name}.csv",
            ),
            index=False,
        )

    print(
        f"[Grad-CAM] {split_name}: "
        f"saved {summary_path} | n={len(results_df)}"
    )
    return results_df


def main() -> int:
    args = parse_args()

    device_info = configure_tensorflow_device(
        tf,
        mode=args.device,
        gpu_index=args.gpu_index,
        operation="gradcam_internal",
    )

    config = MaskAblationConfig(
        dataset_root=args.dataset_root,
        dataset_name=(
            args.dataset_name
            or os.path.basename(
                os.path.abspath(
                    args.dataset_root
                )
            )
        ),
        variant=args.variant,
        labels_csv=args.labels_csv,
        exp_dir=args.exp_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        img_size=(
            args.img_size,
            args.img_size,
        ),
        sampling_policy="class_balanced",
        augmentation_policy="mild_photometric",
        loss_policy="ce_label_smoothing",
        enable_clahe=False,
    )

    logger = setup_logging(config)

    with open(
        os.path.join(
            config.exp_dir,
            "gradcam_compute_device.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            device_info,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    data_splits = split_dataframes(
        config,
        logger,
    )

    model = build_model_for_inference(
        num_classes=config.num_classes,
        img_size=config.img_size,
    )

    weights_path = (
        args.weights
        or os.path.join(
            config.exp_dir,
            "final_all_phases.weights.h5",
        )
    )

    if not try_load_weights(
        model,
        weights_path,
    ):
        raise FileNotFoundError(
            f"Could not load weights: {weights_path}"
        )

    for split_name in args.splits:
        run_split(
            model,
            config,
            data_splits[split_name],
            split_name,
            args.layer_name,
            max_images=args.max_images,
            save_overlays=args.save_overlays,
            overlay_limit=args.overlay_limit,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
