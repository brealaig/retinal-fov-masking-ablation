from __future__ import annotations

import os
import csv
import math
import json
from typing import Dict, Tuple, List

import numpy as np
import cv2
import tensorflow as tf


def build_model_and_load_weights(
    exp_dir: str,
    weights_name: str,
) -> tf.keras.Model:
    weights_path = os.path.join(
        exp_dir,
        weights_name,
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file does not exist: {weights_path}"
        )

    try:
        import train_efficientnet_b0_v6 as training_module
    except Exception as error:
        raise RuntimeError(
            "Could not import 'train_efficientnet_b0_v6.py'. "
            "Run this script from the project root."
        ) from error

    model_builder = None

    for builder_name in (
        "build_model_for_inference",
        "build_model",
        "get_model",
        "create_model",
        "make_model",
    ):
        if hasattr(
            training_module,
            builder_name,
        ):
            model_builder = getattr(
                training_module,
                builder_name,
            )
            break

    if model_builder is None:
        raise RuntimeError(
            "The training script does not expose a public function "
            "for building the model."
        )

    model = model_builder()
    model.load_weights(
        weights_path
    )
    return model


def softmax_probs(
    model: tf.keras.Model,
    img_rgb255: np.ndarray,
) -> np.ndarray:
    input_tensor = tf.convert_to_tensor(
        img_rgb255[None, ...],
        dtype=tf.float32,
    )
    predictions = model(
        input_tensor,
        training=False,
    )

    if isinstance(
        predictions,
        (list, tuple),
    ):
        model_output = predictions[-1]
    else:
        model_output = predictions

    model_output = tf.convert_to_tensor(
        model_output
    )

    if model_output.shape[-1] > 1:
        model_output = tf.nn.softmax(
            model_output,
            axis=-1,
        )

    return model_output.numpy()[0]


def read_csv_rows(
    csv_path: str,
) -> List[Dict[str, str]]:
    rows = []

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )
        for row in reader:
            rows.append(row)

    return rows


def estimate_retina_mask_from_image(
    img_rgb255: np.ndarray,
) -> np.ndarray:
    grayscale_image = cv2.cvtColor(
        img_rgb255,
        cv2.COLOR_RGB2GRAY,
    )
    retinal_mask = (
        grayscale_image > 5
    ).astype(np.uint8) * 255

    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    retinal_mask = cv2.morphologyEx(
        retinal_mask,
        cv2.MORPH_CLOSE,
        morphology_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        retinal_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return np.zeros_like(
            retinal_mask,
            dtype=np.uint8,
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )
    filled_mask = np.zeros_like(
        retinal_mask
    )
    cv2.drawContours(
        filled_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return (
        filled_mask > 0
    ).astype(np.uint8)


def border_ring_from_mask(
    mask: np.ndarray,
    ring_px: int = 14,
) -> np.ndarray:
    binary_mask = (
        mask > 0
    ).astype(np.uint8)

    if binary_mask.max() == 0:
        return np.zeros_like(
            binary_mask,
            dtype=np.uint8,
        )

    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    inner_mask = cv2.erode(
        binary_mask,
        erosion_kernel,
        iterations=max(
            1,
            ring_px // 2,
        ),
    )
    border_ring = (
        binary_mask
        - inner_mask
    )
    border_ring[
        border_ring < 0
    ] = 0

    return border_ring.astype(
        np.uint8
    )


def sort_indices_by_heatmap(
    hmap01: np.ndarray,
) -> np.ndarray:
    flattened_heatmap = hmap01.reshape(
        -1
    )
    return np.argsort(
        -flattened_heatmap
    )


def apply_mask_progressive(
    img: np.ndarray,
    order_idx: np.ndarray,
    k_frac: float,
    mode: str = "deletion",
) -> np.ndarray:
    height, width, channels = img.shape
    pixel_count = height * width
    selected_count = int(
        np.clip(
            k_frac,
            0.0,
            1.0,
        )
        * pixel_count
    )

    baseline_image = cv2.GaussianBlur(
        img,
        (11, 11),
        5,
    )

    output_image = None

    if mode == "deletion":
        output_image = img.copy()

        if selected_count > 0:
            selected_indices = order_idx[
                :selected_count
            ]
            row_indices = (
                selected_indices
                // width
            )
            column_indices = (
                selected_indices
                % width
            )
            output_image[
                row_indices,
                column_indices,
                :,
            ] = baseline_image[
                row_indices,
                column_indices,
                :,
            ]
    elif mode == "insertion":
        output_image = baseline_image.copy()

        if selected_count > 0:
            selected_indices = order_idx[
                :selected_count
            ]
            row_indices = (
                selected_indices
                // width
            )
            column_indices = (
                selected_indices
                % width
            )
            output_image[
                row_indices,
                column_indices,
                :,
            ] = img[
                row_indices,
                column_indices,
                :,
            ]
    else:
        raise ValueError(
            "mode must be 'deletion' or 'insertion'"
        )

    return output_image


def curve_auc(
    xs: np.ndarray,
    ys: np.ndarray,
) -> float:
    return float(
        np.trapz(
            ys,
            x=xs,
        )
    )


def quantify(
    exp_dir: str,
    weights_name: str,
    n_steps: int = 25,
    ring_px: int = 14,
):
    gradcam_dir = os.path.join(
        exp_dir,
        "gradcam",
    )
    input_csv_path = os.path.join(
        gradcam_dir,
        "gradcam_results.csv",
    )
    raw_dir = os.path.join(
        gradcam_dir,
        "raw",
    )
    overlay_dir = os.path.join(
        gradcam_dir,
        "overlays",
    )
    output_dir = os.path.join(
        gradcam_dir,
        "quant",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    result_rows = read_csv_rows(
        input_csv_path
    )

    if len(result_rows) == 0:
        raise RuntimeError(
            f"No rows found in {input_csv_path}"
        )

    model = build_model_and_load_weights(
        exp_dir,
        weights_name,
    )

    curve_points = np.linspace(
        0.0,
        1.0,
        num=n_steps,
        dtype=np.float32,
    )

    per_image_path = os.path.join(
        output_dir,
        "metrics_per_image.csv",
    )
    per_image_file = open(
        per_image_path,
        "w",
        newline="",
        encoding="utf-8",
    )
    per_image_fields = [
        "idx",
        "dataset",
        "y_true",
        "y_pred",
        "correct",
        "auc_insertion",
        "auc_deletion",
        "auc_insertion_random",
        "auc_deletion_random",
        "F_score",
        "energy_inside",
        "energy_outside",
        "border_ratio",
        "cls_used",
        "prob_used",
        "stem",
    ]
    per_image_writer = csv.DictWriter(
        per_image_file,
        fieldnames=per_image_fields,
    )
    per_image_writer.writeheader()

    aggregate_metrics = {}

    for result_row in result_rows:
        sample_index = int(
            result_row["idx"]
        )
        true_label = int(
            result_row["y_true"]
        )
        predicted_label = int(
            result_row["y_pred"]
        )
        correct = int(
            result_row["correct"]
        )
        dataset_name = result_row.get(
            "dataset",
            "",
        )
        class_used = int(
            result_row.get(
                "cls_used",
                predicted_label,
            )
        )
        probability_used = float(
            result_row.get(
                "prob_used",
                result_row.get(
                    "prob_pred",
                    0.0,
                ),
            )
        )

        npy_path = result_row.get(
            "out_npy",
            "",
        )

        if (
            not npy_path
            or not os.path.exists(
                npy_path
            )
        ):
            overlay_path = result_row.get(
                "out_overlay",
                "",
            )

            if overlay_path:
                image_stem = os.path.splitext(
                    os.path.basename(
                        overlay_path
                    )
                )[0].replace(
                    "_overlay",
                    "",
                )
                npy_path = os.path.join(
                    raw_dir,
                    f"{image_stem}_gradcam.npy",
                )

        if not os.path.exists(
            npy_path
        ):
            continue

        heatmap = np.load(
            npy_path
        )
        heatmap = np.clip(
            heatmap.astype(np.float32),
            0.0,
            1.0,
        )

        original_path = result_row.get(
            "out_original",
            "",
        )

        if not os.path.exists(
            original_path
        ):
            image_stem = os.path.splitext(
                os.path.basename(
                    npy_path
                )
            )[0].replace(
                "_gradcam",
                "",
            )
            original_path = os.path.join(
                overlay_dir,
                f"{image_stem}_original.png",
            )

        original_bgr = cv2.imread(
            original_path,
            cv2.IMREAD_COLOR,
        )

        if original_bgr is None:
            continue

        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB,
        )

        retinal_mask = estimate_retina_mask_from_image(
            original_rgb
        )
        total_energy = float(
            heatmap.sum()
            + 1e-8
        )
        inside_energy = float(
            (
                heatmap
                * retinal_mask
            ).sum()
        )
        outside_energy = (
            total_energy
            - inside_energy
        )

        border_ring = border_ring_from_mask(
            retinal_mask,
            ring_px=ring_px,
        )
        border_energy = float(
            (
                heatmap
                * (
                    border_ring > 0
                )
            ).sum()
        )
        border_ratio = (
            border_energy
            / (
                total_energy
                + 1e-8
            )
        )

        importance_order = sort_indices_by_heatmap(
            heatmap
        )
        random_order = np.random.permutation(
            importance_order
        )

        deletion_probabilities = []
        insertion_probabilities = []
        random_deletion_probabilities = []
        random_insertion_probabilities = []

        for fraction in curve_points:
            deletion_image = apply_mask_progressive(
                original_rgb,
                importance_order,
                fraction,
                mode="deletion",
            )
            insertion_image = apply_mask_progressive(
                original_rgb,
                importance_order,
                fraction,
                mode="insertion",
            )
            deletion_probability = softmax_probs(
                model,
                deletion_image,
            )[class_used]
            insertion_probability = softmax_probs(
                model,
                insertion_image,
            )[class_used]

            deletion_probabilities.append(
                deletion_probability
            )
            insertion_probabilities.append(
                insertion_probability
            )

            random_deletion_image = apply_mask_progressive(
                original_rgb,
                random_order,
                fraction,
                mode="deletion",
            )
            random_insertion_image = apply_mask_progressive(
                original_rgb,
                random_order,
                fraction,
                mode="insertion",
            )
            random_deletion_probability = softmax_probs(
                model,
                random_deletion_image,
            )[class_used]
            random_insertion_probability = softmax_probs(
                model,
                random_insertion_image,
            )[class_used]

            random_deletion_probabilities.append(
                random_deletion_probability
            )
            random_insertion_probabilities.append(
                random_insertion_probability
            )

        deletion_probabilities = np.array(
            deletion_probabilities
        )
        insertion_probabilities = np.array(
            insertion_probabilities
        )
        random_deletion_probabilities = np.array(
            random_deletion_probabilities
        )
        random_insertion_probabilities = np.array(
            random_insertion_probabilities
        )

        insertion_auc = float(
            np.trapz(
                insertion_probabilities,
                x=curve_points,
            )
        )
        deletion_auc = float(
            np.trapz(
                1.0
                - deletion_probabilities,
                x=curve_points,
            )
        )
        random_insertion_auc = float(
            np.trapz(
                random_insertion_probabilities,
                x=curve_points,
            )
        )
        random_deletion_auc = float(
            np.trapz(
                1.0
                - random_deletion_probabilities,
                x=curve_points,
            )
        )

        fidelity_score = (
            0.5
            * (
                insertion_auc
                - random_insertion_auc
            )
            + 0.5
            * (
                deletion_auc
                - random_deletion_auc
            )
        )

        image_stem = os.path.splitext(
            os.path.basename(
                original_path
            )
        )[0].replace(
            "_original",
            "",
        )

        per_image_writer.writerow(
            dict(
                idx=sample_index,
                dataset=dataset_name,
                y_true=true_label,
                y_pred=predicted_label,
                correct=correct,
                auc_insertion=insertion_auc,
                auc_deletion=deletion_auc,
                auc_insertion_random=random_insertion_auc,
                auc_deletion_random=random_deletion_auc,
                F_score=fidelity_score,
                energy_inside=(
                    inside_energy
                    / (
                        total_energy
                        + 1e-8
                    )
                ),
                energy_outside=(
                    outside_energy
                    / (
                        total_energy
                        + 1e-8
                    )
                ),
                border_ratio=border_ratio,
                cls_used=class_used,
                prob_used=probability_used,
                stem=image_stem,
            )
        )

        global_key = (
            "global",
            "all",
        )
        class_key = (
            "class",
            true_label,
        )
        dataset_key = (
            "dataset",
            dataset_name
            if dataset_name
            else "unknown",
        )

        for aggregate_key in (
            global_key,
            class_key,
            dataset_key,
        ):
            if aggregate_key not in aggregate_metrics:
                aggregate_metrics[
                    aggregate_key
                ] = dict(
                    n=0,
                    auc_ins=0.0,
                    auc_del=0.0,
                    auc_ins_r=0.0,
                    auc_del_r=0.0,
                    F=0.0,
                    en_in=0.0,
                    en_out=0.0,
                    border=0.0,
                    acc_corr=0,
                )

            aggregate_row = aggregate_metrics[
                aggregate_key
            ]
            aggregate_row["n"] += 1
            aggregate_row["auc_ins"] += insertion_auc
            aggregate_row["auc_del"] += deletion_auc
            aggregate_row["auc_ins_r"] += random_insertion_auc
            aggregate_row["auc_del_r"] += random_deletion_auc
            aggregate_row["F"] += fidelity_score
            aggregate_row["en_in"] += (
                inside_energy
                / (
                    total_energy
                    + 1e-8
                )
            )
            aggregate_row["en_out"] += (
                outside_energy
                / (
                    total_energy
                    + 1e-8
                )
            )
            aggregate_row["border"] += border_ratio
            aggregate_row["acc_corr"] += correct

    per_image_file.close()

    aggregate_path = os.path.join(
        output_dir,
        "metrics_agg.csv",
    )

    with open(
        aggregate_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as aggregate_file:
        aggregate_writer = csv.writer(
            aggregate_file
        )
        aggregate_writer.writerow(
            [
                "group",
                "key",
                "n",
                "acc",
                "auc_ins",
                "auc_del",
                "auc_ins_r",
                "auc_del_r",
                "F",
                "energy_inside",
                "energy_outside",
                "border_ratio",
            ]
        )

        for (
            group_name,
            group_key,
        ), aggregate_row in aggregate_metrics.items():
            sample_count = max(
                1,
                aggregate_row["n"],
            )
            accuracy = (
                aggregate_row["acc_corr"]
                / sample_count
            )

            aggregate_writer.writerow(
                [
                    group_name,
                    group_key,
                    aggregate_row["n"],
                    accuracy,
                    aggregate_row["auc_ins"]
                    / sample_count,
                    aggregate_row["auc_del"]
                    / sample_count,
                    aggregate_row["auc_ins_r"]
                    / sample_count,
                    aggregate_row["auc_del_r"]
                    / sample_count,
                    aggregate_row["F"]
                    / sample_count,
                    aggregate_row["en_in"]
                    / sample_count,
                    aggregate_row["en_out"]
                    / sample_count,
                    aggregate_row["border"]
                    / sample_count,
                ]
            )

    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(
            figsize=(6, 4),
            dpi=120,
        )

        global_metrics = aggregate_metrics.get(
            (
                "global",
                "all",
            ),
            None,
        )

        if (
            global_metrics
            and global_metrics["n"] > 0
        ):
            summary_text = (
                f"n={global_metrics['n']}  "
                f"acc={global_metrics['acc_corr']/global_metrics['n']:.3f}\n"
                f"AUC_ins={global_metrics['auc_ins']/global_metrics['n']:.3f}  "
                f"AUC_del={global_metrics['auc_del']/global_metrics['n']:.3f}\n"
                f"F={global_metrics['F']/global_metrics['n']:.3f}  "
                f"inside={global_metrics['en_in']/global_metrics['n']:.3f}  "
                f"border={global_metrics['border']/global_metrics['n']:.3f}"
            )

            plt.text(
                0.05,
                0.5,
                summary_text,
                fontsize=11,
            )
            plt.axis(
                "off"
            )
            figure.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    "summary_global.png",
                )
            )
            plt.close(
                figure
            )
    except Exception:
        pass

    print(
        "[DONE] metrics_per_image.csv and metrics_agg.csv written to",
        output_dir,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "Grad-CAM quantification (fidelity and bias)"
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--weights",
        default="best_phaseD.weights.h5",
        type=str,
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=25,
        help="Number of points in the Deletion/Insertion curve",
    )
    parser.add_argument(
        "--ring_px",
        type=int,
        default=14,
        help="Peripheral ring width in pixels",
    )

    args = parser.parse_args()

    quantify(
        args.exp_dir,
        args.weights,
        n_steps=args.n_steps,
        ring_px=args.ring_px,
    )


if __name__ == "__main__":
    main()
