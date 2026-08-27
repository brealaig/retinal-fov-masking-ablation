from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, Counter

import numpy as np
import tensorflow as tf

from gradcam_tools.make_gradcam import (
    gradcam_single,
    discover_io_layers,
    save_triplet,
)


def build_model_and_load_weights(exp_dir: str, weights_name: str) -> tf.keras.Model:
    weights_path = os.path.join(exp_dir, weights_name)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    try:
        import train_efficientnet_b0_v6 as training_module
    except Exception as error:
        raise RuntimeError(
            "Could not import 'train_efficientnet_b0_v8.py'. Run this script from the project root."
        ) from error

    model_builder = None
    for builder_name in ("build_model_for_inference", "build_model", "get_model"):
        if hasattr(training_module, builder_name):
            model_builder = getattr(training_module, builder_name)
            break

    if model_builder is None:
        for builder_name in ("create_model", "make_model"):
            if hasattr(training_module, builder_name):
                model_builder = getattr(training_module, builder_name)
                break

    if model_builder is None:
        raise RuntimeError(
            "The training script does not expose a public function for building the model."
        )

    model = model_builder()
    model.load_weights(weights_path)
    _ = discover_io_layers(model)
    return model


def compile_for_eval(model: tf.keras.Model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
        ],
    )


def try_build_val_dataset_from_script(batch_size: int = 1):
    import rd_on_the_fly_v6 as dataset_module

    if hasattr(dataset_module, "Config") and hasattr(dataset_module, "build_val_ds"):
        config = dataset_module.Config()

        if hasattr(config, "aug"):
            try:
                if hasattr(config.aug, "enable"):
                    config.aug.enable = False
                if hasattr(config.aug, "photometric"):
                    config.aug.photometric = False
                if hasattr(config.aug, "geometric"):
                    config.aug.geometric = False
                if hasattr(config.aug, "color"):
                    config.aug.color = False
                if hasattr(config.aug, "brightness"):
                    config.aug.brightness = 0.0
                if hasattr(config.aug, "contrast"):
                    config.aug.contrast = 0.0
                if hasattr(config.aug, "saturation"):
                    config.aug.saturation = 0.0
                if hasattr(config.aug, "hue"):
                    config.aug.hue = 0.0
            except Exception:
                pass

        if hasattr(config, "seed"):
            config.seed = 42

        if hasattr(dataset_module, "setup_logging"):
            try:
                dataset_module.setup_logging(config)
            except Exception:
                pass

        validation_dataset = dataset_module.build_val_ds(config)
        return validation_dataset.unbatch().batch(1)

    candidates = [
        ("get_datasets", dict()),
        ("build_datasets", dict()),
        ("make_datasets", dict()),
        ("get_val_dataset", dict()),
        ("build_val_dataset", dict()),
        ("make_val_dataset", dict()),
    ]

    validation_dataset = None

    for function_name, kwargs in candidates:
        if hasattr(dataset_module, function_name):
            dataset_output = (
                getattr(dataset_module, function_name)(**kwargs)
                if kwargs
                else getattr(dataset_module, function_name)()
            )

            if isinstance(dataset_output, tuple) and len(dataset_output) >= 2:
                validation_dataset = dataset_output[1]
            elif isinstance(dataset_output, dict):
                for split_name in ("val_eval", "val", "validation"):
                    if split_name in dataset_output:
                        validation_dataset = dataset_output[split_name]
                        break
            elif isinstance(dataset_output, tf.data.Dataset):
                validation_dataset = dataset_output

            if validation_dataset is not None:
                break

    if validation_dataset is None:
        raise RuntimeError(
            "Could not obtain 'val_eval' from rd_on_the_fly_v8.py. "
            "Make sure build_val_ds(cfg) is exposed or use --val_dir."
        )

    return validation_dataset.unbatch().batch(1)


def build_val_from_dir(val_dir: str, img_size=(224, 224)):
    autotune = tf.data.AUTOTUNE
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        color_mode="rgb",
        shuffle=False,
        batch_size=1,
    )

    def to_uint8(images, labels):
        images = tf.cast(tf.round(images), tf.uint8)
        metadata = {"path": tf.constant("")}
        return images, labels, metadata

    validation_dataset = validation_dataset.map(
        to_uint8,
        num_parallel_calls=autotune,
    )
    return validation_dataset


def parse_sample(sample):
    if isinstance(sample, (list, tuple)):
        if len(sample) == 2:
            images, labels = sample
            metadata = {}
        elif len(sample) == 3:
            images, labels, metadata = sample
        else:
            raise ValueError("Unexpected batch format: unsupported tuple length.")
    elif isinstance(sample, dict):
        images = sample.get("image", None)
        labels = sample.get("label", None)
        metadata = sample
        if images is None or labels is None:
            raise ValueError("Sample dictionary is missing 'image' or 'label'.")
    else:
        raise ValueError("Unsupported sample format.")

    image_array = (
        images.numpy()
        if isinstance(images, tf.Tensor)
        else np.asarray(images)
    )

    if image_array.ndim == 4:
        image_array = image_array[0]

    if image_array.dtype != np.uint8:
        image_array = np.clip(
            np.rint(image_array),
            0,
            255,
        ).astype(np.uint8)

    true_label = (
        int(labels.numpy().reshape(-1)[0])
        if isinstance(labels, tf.Tensor)
        else int(np.array(labels).reshape(-1)[0])
    )

    sample_path = ""

    if isinstance(metadata, dict):
        for path_key in ("path", "filepath", "file", "filename", "id"):
            if path_key in metadata:
                path_value = metadata[path_key]
                if isinstance(path_value, tf.Tensor):
                    try:
                        path_value = path_value.numpy().decode("utf-8")
                    except Exception:
                        path_value = str(path_value.numpy())
                sample_path = str(path_value)
                break

    return image_array, true_label, sample_path


def open_csv_writer(csv_path: str):
    fieldnames = [
        "idx",
        "filepath",
        "dataset",
        "y_true",
        "y_pred",
        "prob_pred",
        "correct",
        "method",
        "cls_used",
        "prob_used",
        "out_original",
        "out_heatmap",
        "out_overlay",
        "out_npy",
        "probs_json",
    ]

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )
    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    return csv_file, writer, fieldnames


def _to_xy(*sample):
    if len(sample) == 1:
        sample = sample[0]

    if isinstance(sample, (list, tuple)):
        images, labels = sample[:2]
    elif isinstance(sample, dict):
        images = sample.get("image", None)
        labels = sample.get("label", None)
    else:
        raise ValueError("Unsupported format in _to_xy.")

    return tf.cast(images, tf.float32), labels


def collect_balanced_subset(
    val_ds,
    per_class: int,
    num_classes: int = 5,
    max_scan: int = 20000,
):
    class_limits = (
        [per_class] * num_classes
        if per_class > 0
        else [20] * num_classes
    )
    class_counts = [0] * num_classes
    image_batches = []
    labels = []
    scanned_samples = 0

    for item in val_ds.as_numpy_iterator():
        scanned_samples += 1

        if isinstance(item, (list, tuple)):
            images, item_labels = item[:2]
        else:
            images = item["image"]
            item_labels = item["label"]

        label_value = int(
            np.array(item_labels).reshape(-1)[0]
        )

        if (
            0 <= label_value < num_classes
            and class_counts[label_value] < class_limits[label_value]
        ):
            image_batches.append(images)
            labels.append(label_value)
            class_counts[label_value] += 1

            if all(
                class_counts[class_index] >= class_limits[class_index]
                for class_index in range(num_classes)
            ):
                break

        if scanned_samples >= max_scan:
            break

    if not image_batches:
        return None, None

    image_array = np.concatenate(
        image_batches,
        axis=0,
    ).astype(np.float32)
    label_array = np.array(
        labels,
        dtype=np.int64,
    )
    return image_array, label_array


def simple_confusion_matrix(
    y_true,
    y_pred,
    num_classes: int = 5,
):
    confusion_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[
            int(true_label),
            int(predicted_label),
        ] += 1

    return confusion_matrix


def _load_train_prior(exp_dir: str, C: int) -> np.ndarray:
    prior_path = os.path.join(
        exp_dir,
        "data_prior_train.json",
    )

    if os.path.exists(prior_path):
        try:
            with open(
                prior_path,
                "r",
                encoding="utf-8",
            ) as prior_file:
                class_prior = np.array(
                    json.load(prior_file),
                    dtype=np.float32,
                )

            if class_prior.size == C and class_prior.sum() > 0:
                class_prior = class_prior / float(
                    class_prior.sum()
                )
                return class_prior
        except Exception:
            pass

    return np.ones(
        (C,),
        dtype=np.float32,
    ) / float(C)


def apply_temperature_and_prior(
    probs: np.ndarray,
    T: float,
    prior_mode: str,
    prior_custom: str,
    exp_dir: str,
) -> np.ndarray:
    probabilities = np.clip(
        probs.astype(np.float32),
        1e-8,
        1.0,
    )

    if T is None or T <= 0:
        T = 1.0

    temperature_scaled = probabilities ** (
        1.0 / float(T)
    )
    temperature_scaled = temperature_scaled / np.clip(
        temperature_scaled.sum(
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )

    class_count = temperature_scaled.shape[1]

    if prior_mode == "uniform":
        class_prior = np.ones(
            (class_count,),
            dtype=np.float32,
        ) / float(class_count)
    elif prior_mode == "custom":
        if not prior_custom:
            raise ValueError(
                "--prior_custom must be provided when prior_mode=custom"
            )

        class_prior = np.array(
            json.loads(prior_custom),
            dtype=np.float32,
        )

        if class_prior.size != class_count:
            raise ValueError(
                f"prior_custom contains {class_prior.size} classes "
                f"but the model contains {class_count}."
            )

        class_prior = class_prior / np.clip(
            class_prior.sum(),
            1e-8,
            None,
        )
    else:
        class_prior = _load_train_prior(
            exp_dir,
            class_count,
        )

    adjusted_probabilities = (
        temperature_scaled
        * class_prior.reshape((1, -1))
    )
    adjusted_probabilities = adjusted_probabilities / np.clip(
        adjusted_probabilities.sum(
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )

    return adjusted_probabilities


def main():
    import argparse

    parser = argparse.ArgumentParser(
        "Batch Grad-CAM runner for val_eval / external dataset"
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Filename inside exp_dir, e.g. best_phaseD.weights.h5",
    )
    parser.add_argument(
        "--methods",
        default="gradcam",
        type=str,
    )
    parser.add_argument(
        "--per_class",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
    )
    parser.add_argument(
        "--debug_eval_steps",
        type=int,
        default=400,
        help="Evaluation steps for the runner validation dataset before explanation.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Inference calibration temperature (T>=1 softens probabilities).",
    )
    parser.add_argument(
        "--prior_mode",
        type=str,
        default="train",
        choices=["train", "uniform", "custom"],
        help=(
            "Prior reweighting: 'train' reads data_prior_train.json; "
            "'uniform' uses a flat prior; 'custom' uses --prior_custom."
        ),
    )
    parser.add_argument(
        "--prior_custom",
        type=str,
        default="",
        help=(
            'JSON list containing a custom prior, e.g. '
            '"[0.6,0.15,0.15,0.07,0.03]". '
            "Only used when --prior_mode=custom."
        ),
    )

    args = parser.parse_args()

    experiment_dir = args.exp_dir
    overlay_output_dir = os.path.join(
        experiment_dir,
        "gradcam",
        "overlays",
    )
    raw_output_dir = os.path.join(
        experiment_dir,
        "gradcam",
        "raw",
    )
    os.makedirs(
        overlay_output_dir,
        exist_ok=True,
    )
    os.makedirs(
        raw_output_dir,
        exist_ok=True,
    )

    model = build_model_and_load_weights(
        experiment_dir,
        args.weights,
    )

    if args.val_dir:
        validation_dataset = build_val_from_dir(
            args.val_dir,
            img_size=(224, 224),
        )
    else:
        validation_dataset = try_build_val_dataset_from_script()

    try:
        compile_for_eval(model)
        shuffled_validation_dataset = validation_dataset.shuffle(
            10000,
            seed=42,
            reshuffle_each_iteration=False,
        )
        evaluation_steps = (
            args.debug_eval_steps
            if args.debug_eval_steps > 0
            else None
        )
        shuffled_evaluation = model.evaluate(
            shuffled_validation_dataset.map(
                lambda *sample: _to_xy(*sample)
            ),
            steps=evaluation_steps,
            verbose=1,
        )
        print(
            "[DEBUG] runner evaluation (shuffled):",
            shuffled_evaluation,
        )
    except Exception as error:
        print(
            "[DEBUG] runner evaluation (shuffled): skipped due to error:",
            error,
        )

    try:
        balanced_images, balanced_labels = collect_balanced_subset(
            (
                shuffled_validation_dataset
                if "shuffled_validation_dataset" in locals()
                else validation_dataset.shuffle(
                    10000,
                    seed=123,
                    reshuffle_each_iteration=False,
                )
            ),
            per_class=max(
                1,
                min(args.per_class, 200),
            ),
            num_classes=5,
            max_scan=20000,
        )

        if balanced_images is not None:
            balanced_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    balanced_images,
                    balanced_labels,
                )
            ).batch(32)

            balanced_evaluation = model.evaluate(
                balanced_dataset,
                verbose=1,
            )
            print(
                "[DEBUG] runner evaluation (balanced subset):",
                balanced_evaluation,
            )

            balanced_probabilities = model.predict(
                balanced_dataset,
                verbose=0,
            )
            balanced_predictions = balanced_probabilities.argmax(
                axis=1
            )
            true_histogram = dict(
                Counter(
                    balanced_labels.tolist()
                )
            )
            prediction_histogram = dict(
                Counter(
                    balanced_predictions.tolist()
                )
            )
            print(
                "[DEBUG] true histogram (balanced):",
                true_histogram,
            )
            print(
                "[DEBUG] prediction histogram (balanced):",
                prediction_histogram,
            )

            confusion_matrix = simple_confusion_matrix(
                balanced_labels,
                balanced_predictions,
                num_classes=5,
            )
            print(
                "[DEBUG] confusion matrix (rows=true, cols=pred):\n",
                confusion_matrix,
            )

            calibrated_balanced_probabilities = apply_temperature_and_prior(
                balanced_probabilities,
                args.temp,
                args.prior_mode,
                args.prior_custom,
                experiment_dir,
            )
            calibrated_balanced_predictions = (
                calibrated_balanced_probabilities.argmax(
                    axis=1
                )
            )
            calibrated_prediction_histogram = dict(
                Counter(
                    calibrated_balanced_predictions.tolist()
                )
            )
            print(
                "[DEBUG] prediction histogram (balanced, calibrated):",
                calibrated_prediction_histogram,
            )

            calibrated_confusion_matrix = simple_confusion_matrix(
                balanced_labels,
                calibrated_balanced_predictions,
                num_classes=5,
            )
            print(
                "[DEBUG] calibrated confusion matrix (rows=true, cols=pred):\n",
                calibrated_confusion_matrix,
            )
        else:
            print(
                "[DEBUG] balanced subset: could not be built "
                "(insufficient data in scan)."
            )
    except Exception as error:
        print(
            "[DEBUG] runner evaluation (balanced): skipped due to error:",
            error,
        )

    class_limits = defaultdict(
        lambda: (
            math.inf
            if args.per_class < 0
            else args.per_class
        )
    )
    class_counts = defaultdict(int)

    csv_path = os.path.join(
        experiment_dir,
        "gradcam",
        "gradcam_results.csv",
    )
    csv_file, writer, fieldnames = open_csv_writer(
        csv_path
    )

    try:
        probabilities_layer = model.get_layer(
            "probs"
        ).output
        num_classes = int(
            probabilities_layer.shape[-1]
        )
    except Exception:
        num_classes = 5

    sample_index = 0

    for sample in validation_dataset:
        try:
            image_uint8, true_label, sample_path = parse_sample(
                sample
            )
        except Exception as error:
            print(
                "[WARN] Could not parse sample; skipping it. Error:",
                error,
            )
            continue

        if class_counts[true_label] >= class_limits[true_label]:
            continue

        model_input = image_uint8.astype(
            np.float32
        )[None, ...]

        probabilities = model.predict(
            model_input,
            verbose=0,
        )
        adjusted_probabilities = apply_temperature_and_prior(
            probabilities,
            args.temp,
            args.prior_mode,
            args.prior_custom,
            experiment_dir,
        )
        predicted_label = int(
            np.argmax(
                adjusted_probabilities[0]
            )
        )
        predicted_probability = float(
            np.max(
                adjusted_probabilities[0]
            )
        )

        gradcam_result = gradcam_single(
            model,
            image_uint8,
            class_index=predicted_label,
            target_conv_name="top_conv",
            alpha_overlay=args.alpha,
            use_guided_relu=False,
        )

        correct = int(
            predicted_label == true_label
        )

        output_stem = (
            os.path.splitext(
                os.path.basename(
                    sample_path
                )
            )[0]
            if sample_path
            else f"val_{sample_index:06d}"
        )

        output_paths = save_triplet(
            overlay_output_dir,
            output_stem,
            image_uint8,
            gradcam_result["heatmap"],
            gradcam_result["overlay"],
        )

        npy_path = ""

        if args.save_npy:
            npy_path = os.path.join(
                raw_output_dir,
                f"{output_stem}_gradcam.npy",
            )
            np.save(
                npy_path,
                gradcam_result["heatmap"],
            )

        dataset_name = ""

        if sample_path:
            lowercase_path = sample_path.lower()

            if "eyepacs" in lowercase_path:
                dataset_name = "eyepacs"
            elif "idrid" in lowercase_path:
                dataset_name = "idrid"
            elif "aptos" in lowercase_path:
                dataset_name = "aptos"
            elif "messidor" in lowercase_path:
                dataset_name = "messidor"

        result_row = dict(
            idx=sample_index,
            filepath=sample_path,
            dataset=dataset_name,
            y_true=true_label,
            y_pred=predicted_label,
            prob_pred=predicted_probability,
            correct=correct,
            method="gradcam",
            cls_used=predicted_label,
            prob_used=predicted_probability,
            out_original=output_paths["original"],
            out_heatmap=output_paths["heatmap"],
            out_overlay=output_paths["overlay"],
            out_npy=npy_path,
            probs_json=json.dumps(
                [
                    float(value)
                    for value in adjusted_probabilities[0].tolist()
                ]
            ),
        )

        writer.writerow(result_row)
        csv_file.flush()

        class_counts[true_label] += 1
        sample_index += 1

        if all(
            class_counts[class_index] >= class_limits[class_index]
            for class_index in range(num_classes)
        ):
            break

    csv_file.close()

    print(
        f"[DONE] Explanations saved: {sample_index} rows in {csv_path}"
    )
    print(
        f"Overlays saved in: {overlay_output_dir}"
    )

    if args.save_npy:
        print(
            f".npy maps saved in: {raw_output_dir}"
        )


if __name__ == "__main__":
    main()
