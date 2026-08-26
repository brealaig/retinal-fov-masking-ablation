from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from compute_device_utils import (
    add_device_cli_arguments,
    configure_tensorflow_device,
    preconfigure_device_from_argv,
)

_PRE_DEVICE_REQUEST = preconfigure_device_from_argv(sys.argv[1:])

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

from tensorflow.keras import callbacks, layers, metrics, models, regularizers
from tensorflow.keras.applications import efficientnet as eff
from tensorflow.keras.optimizers import Adam

from rd_on_the_fly_mask_ablation import (
    MaskAblationConfig,
    VALID_AUGMENTATION,
    VALID_LOSS,
    VALID_SAMPLING,
    VALID_USAGE,
    VALID_VARIANTS,
    build_eval_ds,
    build_train_ds,
    class_counts,
    estimate_steps_per_epoch,
    set_global_seed,
    setup_logging,
    split_dataframes,
    write_experiment_config,
)

# -------------------------
# REPRODUCIBLE ENVIRONMENT
# -------------------------

def configure_environment(seed: int, device: str = "auto", gpu_index: int = 0,
                          operation: str = "training") -> Dict[str, object]:
    set_global_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return configure_tensorflow_device(
        tf,
        mode=device,
        gpu_index=gpu_index,
        operation=operation,
    )


# -------------------------
# MODEL
# -------------------------

def build_model_for_inference(num_classes: int = 5, img_size: Tuple[int, int] = (224, 224)) -> tf.keras.Model:
    weight_decay = 1e-5
    image_input = layers.Input(shape=(img_size[0], img_size[1], 3), dtype=tf.float32, name="input_rgb255")
    features = layers.Lambda(eff.preprocess_input, name="preproc")(image_input)
    backbone = eff.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=features)
    backbone.trainable = False
    features = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    features = layers.Dense(256, activation=tf.keras.activations.gelu,
                     kernel_regularizer=regularizers.l2(weight_decay), name="head_dense")(features)
    features = layers.BatchNormalization(name="head_bn")(features)
    features = layers.Dropout(0.30, name="head_dropout")(features)
    class_probabilities = layers.Dense(num_classes, activation="softmax", name="probs")(features)
    return models.Model(inputs=image_input, outputs=class_probabilities, name="EffNetB0_RD_mask_ablation")


def set_backbone_trainable_by_name(model: tf.keras.Model, blocks: Sequence[int] = (5, 6, 7)) -> None:
    import re
    block_name_pattern = re.compile(rf"block({'|'.join(str(b) for b in blocks)})[a-z]_.*")
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers:
        layer_name = layer.name
        if block_name_pattern.match(layer_name) or layer_name == "top_conv":
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
    for head_name in ("gap", "head_dense", "head_bn", "head_dropout", "probs"):
        try:
            model.get_layer(head_name).trainable = True
        except Exception:
            pass
    trainable_parameter_count = int(np.sum([np.prod(weight_tensor.shape) for weight_tensor in model.trainable_weights]))
    total_parameter_count = int(np.sum([np.prod(weight_tensor.shape) for weight_tensor in model.weights]))
    print(f"[MODEL] Trainable params: {trainable_parameter_count:,} / {total_parameter_count:,}")


# -------------------------
# LOSSES
# -------------------------

class SmoothSparseCE(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, epsilon: float = 0.02, name: str = "smooth_sparse_ce"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        target_class_probability = 1.0 - self.epsilon
        other_class_probability = self.epsilon / tf.cast(self.num_classes - 1, tf.float32)
        one_hot_targets = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)
        smoothed_targets = one_hot_targets * target_class_probability + (1.0 - one_hot_targets) * other_class_probability
        return self.ce(smoothed_targets, y_pred)


class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, gamma: float = 1.3, alpha: Optional[Sequence[float]] = None,
                 name: str = "sparse_focal"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else list(float(alpha_value) for alpha_value in alpha)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        numerical_epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, numerical_epsilon, 1.0 - numerical_epsilon)
        one_hot_targets = tf.one_hot(y_true, depth=self.num_classes, dtype=tf.float32)
        true_class_probability = tf.reduce_sum(one_hot_targets * y_pred, axis=-1)
        if self.alpha is not None:
            class_alpha = tf.gather(tf.constant(self.alpha, dtype=tf.float32), y_true)
        else:
            class_alpha = 1.0
        return tf.reduce_mean(-class_alpha * tf.pow(1.0 - true_class_probability, self.gamma) * tf.math.log(true_class_probability + numerical_epsilon))


class CEPlusFocal(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, alpha: Optional[Sequence[float]] = None, gamma: float = 1.3,
                 ce_epsilon: float = 0.02, lam_focal: float = 0.25, name: str = "ce_plus_focal"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.ce = SmoothSparseCE(num_classes=num_classes, epsilon=ce_epsilon)
        self.focal = SparseFocalLoss(num_classes=num_classes, gamma=gamma, alpha=alpha)
        self.lam = float(lam_focal)

    def call(self, y_true, y_pred):
        return (1.0 - self.lam) * self.ce(y_true, y_pred) + self.lam * self.focal(y_true, y_pred)


def cb_focal_alpha_from_counts(counts: Sequence[int], beta: float = 0.9999) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64) + 1e-8
    effective_sample_number = 1.0 - np.power(beta, counts)
    class_balance_weights = (1.0 - beta) / effective_sample_number
    class_balance_weights = class_balance_weights / (np.mean(class_balance_weights) + 1e-8)
    return class_balance_weights.astype(np.float32)


def build_loss(cfg: MaskAblationConfig, train_counts: Dict[int, int]) -> tf.keras.losses.Loss:
    if cfg.loss_policy == "ce":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    if cfg.loss_policy == "ce_label_smoothing":
        return SmoothSparseCE(num_classes=cfg.num_classes, epsilon=0.02)
    if cfg.loss_policy == "ce_plus_focal":
        class_count_values = [train_counts.get(class_index, 1) for class_index in range(cfg.num_classes)]
        focal_alpha_weights = np.clip(cb_focal_alpha_from_counts(class_count_values), 0.5, 1.5)
        return CEPlusFocal(num_classes=cfg.num_classes, alpha=focal_alpha_weights.tolist(), gamma=1.3, ce_epsilon=0.02, lam_focal=0.25)
    raise ValueError(f"Unsupported loss_policy: {cfg.loss_policy}")


# -------------------------
# CALLBACKS
# -------------------------

class ValKappaCallback(callbacks.Callback):
    def __init__(self, ds, key: str = "val_eval_qwk", num_classes: int = 5, max_batches: Optional[int] = None):
        super().__init__()
        self.ds = ds
        self.key = key
        self.num_classes = int(num_classes)
        self.max_batches = max_batches

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        true_label_batches, predicted_label_batches = [], []
        processed_batch_count = 0
        for image_batch, label_batch in self.ds:
            class_probabilities = self.model.predict(image_batch, verbose=0)
            true_label_batches.append(label_batch.numpy().reshape(-1))
            predicted_label_batches.append(np.argmax(class_probabilities, axis=-1))
            processed_batch_count += 1
            if self.max_batches is not None and processed_batch_count >= self.max_batches:
                break
        if not true_label_batches:
            print("[WARN] ValKappaCallback: Empty dataset.")
            return
        from sklearn.metrics import cohen_kappa_score
        true_labels = np.concatenate(true_label_batches)
        predicted_labels = np.concatenate(predicted_label_batches)
        quadratic_kappa = cohen_kappa_score(true_labels, predicted_labels, weights="quadratic")
        logs[self.key] = float(quadratic_kappa)
        print(f"\n[VAL] {self.key}: {quadratic_kappa:.4f}")


def every_n_ckpt(path_tmpl: str, n: int = 2):
    class _EveryN(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if n > 0 and (epoch + 1) % n == 0:
                checkpoint_path = path_tmpl.format(epoch=epoch + 1)
                try:
                    self.model.save_weights(checkpoint_path)
                    print(f"[CKPT] Periodic saving: {checkpoint_path}")
                except Exception as exc:
                    print(f"[CKPT] Periodic saving failed ({checkpoint_path}): {exc}")
    return _EveryN()


def try_load_weights(model: tf.keras.Model, path: str) -> bool:
    if path and os.path.isfile(path):
        try:
            model.load_weights(path)
            print(f"[WEIGHTS] Weights loaded: {path}")
            return True
        except Exception as exc:
            print(f"[WEIGHTS] Weights couldn't be loaded {path}: {exc}")
    else:
        print(f"[WEIGHTS] Checkpoint doesn't exist: {path}")
    return False


# -------------------------
# CALIBRATION AND METRICS
# -------------------------

def predict_proba(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    probability_batches, label_batches = [], []
    for image_batch, label_batch in ds:
        batch_probabilities = model.predict(image_batch, verbose=0)
        batch_probabilities = np.asarray(batch_probabilities, dtype=np.float32)
        probability_batches.append(batch_probabilities)
        label_batches.append(label_batch.numpy().reshape(-1).astype(np.int32))
    if not probability_batches:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(probability_batches, axis=0), np.concatenate(label_batches, axis=0)


def predict_tta(model: tf.keras.Model, ds: tf.data.Dataset, enabled: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return predict_proba(model, ds)
    probability_batches, label_batches = [], []
    for image_batch, label_batch in ds:
        original_probabilities = model.predict(image_batch, verbose=0)
        horizontal_flip_probabilities = model.predict(tf.image.flip_left_right(image_batch), verbose=0)
        rotation_probabilities = model.predict(tf.image.rot90(image_batch, k=2), verbose=0)
        probability_batches.append(((original_probabilities + horizontal_flip_probabilities + rotation_probabilities) / 3.0).astype(np.float32))
        label_batches.append(label_batch.numpy().reshape(-1).astype(np.int32))
    if not probability_batches:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(probability_batches, axis=0), np.concatenate(label_batches, axis=0)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(max(1e-6, temperature))
    numerical_epsilon = 1e-8
    log_probabilities = np.log(np.clip(probs, numerical_epsilon, 1.0))
    temperature_scaled_logits = log_probabilities / temperature
    exponentiated_logits = np.exp(temperature_scaled_logits - np.max(temperature_scaled_logits, axis=-1, keepdims=True))
    return exponentiated_logits / np.sum(exponentiated_logits, axis=-1, keepdims=True)


def nll_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    numerical_epsilon = 1e-8
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true.astype(int)], numerical_epsilon, 1.0))))


def temperature_search_nll(probs: np.ndarray, y_true: np.ndarray,
                           grid: np.ndarray = np.arange(0.70, 2.51, 0.05)) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 1.0, float("nan")
    best_temperature, best_negative_log_likelihood = 1.0, float("inf")
    for candidate_temperature in grid:
        calibrated_probabilities = apply_temperature(probs, float(candidate_temperature))
        candidate_nll = nll_score(y_true, calibrated_probabilities)
        if candidate_nll < best_negative_log_likelihood:
            best_temperature, best_negative_log_likelihood = float(candidate_temperature), float(candidate_nll)
    return best_temperature, best_negative_log_likelihood


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidence_scores = np.max(probs, axis=1)
    predicted_labels = np.argmax(probs, axis=1)
    correct_predictions = (predicted_labels == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    expected_calibration_error = 0.0
    for bin_index in range(n_bins):
        lower_bound, upper_bound = bin_edges[bin_index], bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            bin_mask = (confidence_scores >= lower_bound) & (confidence_scores <= upper_bound)
        else:
            bin_mask = (confidence_scores >= lower_bound) & (confidence_scores < upper_bound)
        if np.any(bin_mask):
            expected_calibration_error += float(np.mean(bin_mask)) * abs(float(np.mean(confidence_scores[bin_mask])) - float(np.mean(correct_predictions[bin_mask])))
    return float(expected_calibration_error)


def brier_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int = 5) -> float:
    one_hot_targets = np.eye(num_classes, dtype=np.float32)[y_true.astype(int)]
    return float(np.mean(np.sum((probs - one_hot_targets) ** 2, axis=1)))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, num_classes: int = 5) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
    if len(y_true) == 0:
        return {metric_name: float("nan") for metric_name in [
            "accuracy", "macro_F1", "weighted_F1", "balanced_accuracy", "QWK",
            "top2_accuracy", "ECE", "NLL", "Brier"
        ]}
    predicted_labels = np.argmax(probs, axis=1)
    top2_class_indices = np.argsort(probs, axis=1)[:, -2:]
    return {
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "macro_F1": float(f1_score(y_true, predicted_labels, average="macro", labels=list(range(num_classes)), zero_division=0)),
        "weighted_F1": float(f1_score(y_true, predicted_labels, average="weighted", labels=list(range(num_classes)), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_labels)),
        "QWK": float(cohen_kappa_score(y_true, predicted_labels, weights="quadratic")),
        "top2_accuracy": float(np.mean([int(y_true[sample_index] in top2_class_indices[sample_index]) for sample_index in range(len(y_true))])),
        "ECE": ece_score(y_true, probs),
        "NLL": nll_score(y_true, probs),
        "Brier": brier_score(y_true, probs, num_classes=num_classes),
    }


def confusion_and_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import confusion_matrix
    confusion_matrix_values = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix_values, index=[f"true_{class_index_label}" for class_index_label in range(num_classes)], columns=[f"pred_{class_index_label}" for class_index_label in range(num_classes)])
    per_class_rows = []
    total_samples = int(confusion_matrix_values.sum())
    for class_index in range(num_classes):
        true_positives = int(confusion_matrix_values[class_index, class_index])
        false_negatives = int(confusion_matrix_values[class_index, :].sum() - true_positives)
        false_positives = int(confusion_matrix_values[:, class_index].sum() - true_positives)
        true_negatives = int(total_samples - true_positives - false_negatives - false_positives)
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else np.nan
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else np.nan
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else np.nan
        per_class_rows.append({
            "class": class_index,
            "support": int(confusion_matrix_values[class_index, :].sum()),
            "TP": true_positives,
            "FP": false_positives,
            "FN": false_negatives,
            "TN": true_negatives,
            "sensitivity": float(sensitivity) if not np.isnan(sensitivity) else np.nan,
            "precision": float(precision) if not np.isnan(precision) else np.nan,
            "specificity": float(specificity) if not np.isnan(specificity) else np.nan,
        })
    return confusion_matrix_dataframe, pd.DataFrame(per_class_rows)


def save_prediction_outputs(exp_dir: str, split_name: str, df_split: pd.DataFrame, y_true: np.ndarray,
                            probs: np.ndarray, cfg: MaskAblationConfig, temperature: float,
                            calibration_applied: bool = True) -> Dict[str, float]:
    os.makedirs(exp_dir, exist_ok=True)
    predicted_labels = np.argmax(probs, axis=1)
    computed_metrics = compute_metrics(y_true, probs, num_classes=cfg.num_classes)
    metrics_record = dict(computed_metrics)
    metrics_record.update({
        "dataset": cfg.dataset_name,
        "variant": cfg.variant,
        "seed": int(cfg.seed),
        "split": split_name,
        "n": int(len(y_true)),
        "temperature": float(temperature),
        "calibration_applied": bool(calibration_applied),
    })
    pd.DataFrame([metrics_record]).to_csv(os.path.join(exp_dir, f"metrics_{split_name}.csv"), index=False)

    confusion_matrix_dataframe, per_class_metrics_dataframe = confusion_and_per_class(y_true, predicted_labels, num_classes=cfg.num_classes)
    confusion_matrix_dataframe.to_csv(os.path.join(exp_dir, f"confusion_matrix_{split_name}.csv"))
    per_class_metrics_dataframe.insert(0, "split", split_name)
    per_class_metrics_dataframe.insert(0, "seed", int(cfg.seed))
    per_class_metrics_dataframe.insert(0, "variant", cfg.variant)
    per_class_metrics_dataframe.insert(0, "dataset", cfg.dataset_name)
    per_class_metrics_dataframe.to_csv(os.path.join(exp_dir, f"per_class_metrics_{split_name}.csv"), index=False)

    predictions_dataframe = pd.DataFrame({
        "filename": df_split["filename"].astype(str).values,
        "true_label": y_true.astype(int),
        "pred_label": predicted_labels.astype(int),
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "prob_2": probs[:, 2],
        "prob_3": probs[:, 3],
        "prob_4": probs[:, 4],
        "confidence": np.max(probs, axis=1),
        "usage": df_split["usage"].astype(str).values,
        "brightness_category": df_split["brightness_category"].astype(str).values,
        "p50": df_split["p50"].values,
        "dataset": cfg.dataset_name,
        "variant": cfg.variant,
        "seed": int(cfg.seed),
    })
    predictions_dataframe.to_csv(os.path.join(exp_dir, f"predictions_{split_name}.csv"), index=False)
    return metrics_record


def save_brightness_metrics(exp_dir: str, split_name: str, df_split: pd.DataFrame, y_true: np.ndarray,
                            probs: np.ndarray, cfg: MaskAblationConfig) -> None:
    evaluation_dataframe = df_split[["filename", "usage", "brightness_category", "p50", "label"]].copy().reset_index(drop=True)
    evaluation_dataframe["true_label"] = y_true.astype(int)
    evaluation_dataframe["pred_label"] = np.argmax(probs, axis=1).astype(int)
    for class_index in range(cfg.num_classes):
        evaluation_dataframe[f"prob_{class_index}"] = probs[:, class_index]

    brightness_metric_rows = []
    for brightness_category, brightness_subset in evaluation_dataframe.groupby("brightness_category", dropna=False):
        subset_indices = brightness_subset.index.values
        subset_metrics = compute_metrics(y_true[subset_indices], probs[subset_indices], num_classes=cfg.num_classes)
        subset_metrics.update({"dataset": cfg.dataset_name, "variant": cfg.variant, "seed": int(cfg.seed), "split": split_name,
                   "brightness_category": brightness_category, "n": int(len(brightness_subset))})
        brightness_metric_rows.append(subset_metrics)
    pd.DataFrame(brightness_metric_rows).to_csv(os.path.join(exp_dir, f"metrics_by_brightness_{split_name}.csv"), index=False)

    class_brightness_metric_rows = []
    for (class_label, brightness_category), brightness_subset in evaluation_dataframe.groupby(["true_label", "brightness_category"], dropna=False):
        subset_indices = brightness_subset.index.values
        subset_metrics = compute_metrics(y_true[subset_indices], probs[subset_indices], num_classes=cfg.num_classes)
        class_brightness_metric_rows.append({
            "dataset": cfg.dataset_name,
            "variant": cfg.variant,
            "seed": int(cfg.seed),
            "split": split_name,
            "class": int(class_label),
            "brightness_category": brightness_category,
            "n": int(len(brightness_subset)),
            "accuracy": subset_metrics["accuracy"],
            "macro_F1": subset_metrics["macro_F1"],
            "QWK": subset_metrics["QWK"],
            "ECE": subset_metrics["ECE"],
            "NLL": subset_metrics["NLL"],
            "Brier": subset_metrics["Brier"],
        })
    pd.DataFrame(class_brightness_metric_rows).to_csv(os.path.join(exp_dir, f"metrics_by_class_and_brightness_{split_name}.csv"), index=False)


def evaluate_splits(model: tf.keras.Model, splits: Dict[str, pd.DataFrame], cfg: MaskAblationConfig,
                    temperature: float, split_names: Sequence[str]) -> List[Dict[str, float]]:
    evaluation_rows = []
    for split_name in split_names:
        if split_name not in splits:
            raise ValueError(f"Invalid split: {split_name}")
        evaluation_dataset = build_eval_ds(cfg, splits[split_name])
        raw_probabilities, y_true = predict_tta(model, evaluation_dataset, enabled=cfg.use_tta_main)
        calibrated_probabilities = apply_temperature(raw_probabilities, temperature) if temperature and temperature != 1.0 else raw_probabilities
        metrics_record = save_prediction_outputs(cfg.exp_dir, split_name, splits[split_name], y_true, calibrated_probabilities, cfg, temperature,
                                      calibration_applied=bool(temperature and temperature != 1.0))
        save_brightness_metrics(cfg.exp_dir, split_name, splits[split_name], y_true, calibrated_probabilities, cfg)
        print(f"[EVAL] {split_name}: QWK={metrics_record['QWK']:.4f} | acc={metrics_record['accuracy']:.4f} | macroF1={metrics_record['macro_F1']:.4f} | ECE={metrics_record['ECE']:.4f}")
        evaluation_rows.append(metrics_record)
    return evaluation_rows


# -------------------------
# TRAINING
# -------------------------

def compile_model(model: tf.keras.Model, optimizer, loss_obj) -> None:
    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=[
            metrics.SparseCategoricalAccuracy(name="acc"),
            metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
        ],
    )


def maybe_class_weight(cfg: MaskAblationConfig, class_weight: Dict[int, float]) -> Optional[Dict[int, float]]:
    if not cfg.use_class_weight:
        return None
    if cfg.sampling_policy != "natural":
        print("[WARN] class_weight was requested with balanced sampling. It is disabled to avoid double compensation.")
        return None
    if cfg.loss_policy == "ce_plus_focal":
        print("[WARN] class_weight was requested with focal loss. It is disabled by default to avoid double compensation.")
        return None
    return class_weight


def phase_callbacks(exp_dir: str, val_eval_ds: tf.data.Dataset, num_classes: int, phase: str, patience: int,
                    save_every: int = 0) -> List[callbacks.Callback]:
    checkpoint_path = os.path.join(exp_dir, f"best_phase{phase}.weights.h5")
    callback_list: List[callbacks.Callback] = [
        ValKappaCallback(val_eval_ds, key="val_eval_qwk", num_classes=num_classes),
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_eval_qwk", mode="max", save_best_only=True, save_weights_only=True),
        callbacks.EarlyStopping(monitor="val_eval_qwk", mode="max", patience=patience, restore_best_weights=True),
        callbacks.CSVLogger(os.path.join(exp_dir, f"history_phase{phase}.csv"), append=False),
        callbacks.TensorBoard(log_dir=os.path.join(exp_dir, f"tb_logs_phase{phase}"), write_graph=False),
        callbacks.BackupAndRestore(os.path.join(exp_dir, f"bak_phase{phase}")),
    ]
    if save_every and save_every > 0:
        callback_list.append(every_n_ckpt(os.path.join(exp_dir, f"phase{phase}_epoch{{epoch:03d}}.h5"), n=save_every))
    return callback_list


def train_model(cfg: MaskAblationConfig, splits: Dict[str, pd.DataFrame], args) -> tf.keras.Model:
    training_dataset, class_weight_mapping, training_dataframe = build_train_ds(cfg, splits["train"])
    validation_dataset = build_eval_ds(cfg, splits["val_eval"])
    steps_per_epoch = estimate_steps_per_epoch(training_dataframe, cfg, max_steps=args.max_steps_per_epoch)
    print(f"[DATA] train={len(training_dataframe)} | val_eval={len(splits['val_eval'])} | val_calib={len(splits['val_calib'])} | steps/epoch={steps_per_epoch}")

    sanity_image_batch, sanity_label_batch = next(iter(training_dataset))
    assert sanity_image_batch.dtype == tf.float32 and sanity_image_batch.shape[-3:] == (cfg.img_size[0], cfg.img_size[1], 3)
    print(f"[SANITY] batch0={sanity_image_batch.shape} | range~[{float(tf.reduce_min(sanity_image_batch).numpy()):.2f}, {float(tf.reduce_max(sanity_image_batch).numpy()):.2f}]")

    model = build_model_for_inference(num_classes=cfg.num_classes, img_size=cfg.img_size)
    training_loss = build_loss(cfg, class_counts(training_dataframe, cfg.num_classes))
    class_weight_argument = maybe_class_weight(cfg, class_weight_mapping)

    # PHASE A (Classification head with a frozen backbone) 

    if args.epochs_a > 0:
        compile_model(model, Adam(learning_rate=args.lr_a, clipnorm=1.0), training_loss)
        model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs_a,
            steps_per_epoch=steps_per_epoch,
            callbacks=phase_callbacks(cfg.exp_dir, validation_dataset, cfg.num_classes, "A", patience=args.patience_a, save_every=args.save_every_a),
            verbose=1,
            class_weight=class_weight_argument,
        )
        model.save_weights(os.path.join(cfg.exp_dir, "final_phaseA_last.weights.h5"))

    # PHASE B (Selective fine-tuning)

    phase_a_checkpoint = os.path.join(cfg.exp_dir, "best_phaseA.weights.h5")
    try_load_weights(model, phase_a_checkpoint)
    if args.epochs_b > 0:
        set_backbone_trainable_by_name(model, blocks=(4, 5, 6, 7))
        phase_b_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr_b,
            decay_steps=max(1, steps_per_epoch * args.epochs_b),
            alpha=args.min_lr_b / args.lr_b,
        )
        compile_model(model, Adam(learning_rate=phase_b_learning_rate, clipnorm=1.0), training_loss)
        model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs_b,
            steps_per_epoch=steps_per_epoch,
            callbacks=phase_callbacks(cfg.exp_dir, validation_dataset, cfg.num_classes, "B", patience=args.patience_b, save_every=args.save_every_b),
            verbose=1,
        )
        model.save_weights(os.path.join(cfg.exp_dir, "final_phaseB_last.weights.h5"))

    # PHASE C (Deeper fine-tuning)

    phase_b_checkpoint = os.path.join(cfg.exp_dir, "best_phaseB.weights.h5")
    try_load_weights(model, phase_b_checkpoint)
    if args.epochs_c > 0:
        set_backbone_trainable_by_name(model, blocks=(2, 3, 4, 5, 6, 7))
        phase_c_learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=args.lr_c,
            first_decay_steps=max(1, steps_per_epoch * max(1, args.epochs_c // 2)),
            t_mul=1.0,
            m_mul=0.8,
            alpha=args.min_lr_c / args.lr_c,
        )
        compile_model(model, Adam(learning_rate=phase_c_learning_rate, clipnorm=1.0), training_loss)
        model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs_c,
            steps_per_epoch=steps_per_epoch,
            callbacks=phase_callbacks(cfg.exp_dir, validation_dataset, cfg.num_classes, "C", patience=args.patience_c, save_every=args.save_every_c),
            verbose=1,
        )
        model.save_weights(os.path.join(cfg.exp_dir, "final_phaseC_last.weights.h5"))

    # PHASE D (Lightweight tail)

    phase_c_checkpoint = os.path.join(cfg.exp_dir, "best_phaseC.weights.h5")
    try_load_weights(model, phase_c_checkpoint)
    if args.epochs_d > 0:
        phase_d_learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr_d,
            decay_steps=max(1, steps_per_epoch * args.epochs_d),
            alpha=args.min_lr_d / args.lr_d,
        )
        compile_model(model, Adam(learning_rate=phase_d_learning_rate, clipnorm=1.0), training_loss)
        model.fit(
            training_dataset,
            validation_data=validation_dataset,
            epochs=args.epochs_d,
            steps_per_epoch=steps_per_epoch,
            callbacks=phase_callbacks(cfg.exp_dir, validation_dataset, cfg.num_classes, "D", patience=args.patience_d, save_every=args.save_every_d),
            verbose=1,
        )
        model.save_weights(os.path.join(cfg.exp_dir, "final_phaseD_last.weights.h5"))

    # LOADING THE BEST AVIABLE CHECKPOINT FOR FINAL EVALUATION

    checkpoint_candidates = [
        os.path.join(cfg.exp_dir, "best_phaseD.weights.h5"),
        os.path.join(cfg.exp_dir, "best_phaseC.weights.h5"),
        os.path.join(cfg.exp_dir, "best_phaseB.weights.h5"),
        os.path.join(cfg.exp_dir, "best_phaseA.weights.h5"),
        os.path.join(cfg.exp_dir, "final_phaseD_last.weights.h5"),
        os.path.join(cfg.exp_dir, "final_phaseC_last.weights.h5"),
        os.path.join(cfg.exp_dir, "final_phaseB_last.weights.h5"),
        os.path.join(cfg.exp_dir, "final_phaseA_last.weights.h5"),
    ]
    for checkpoint_path in checkpoint_candidates:
        if try_load_weights(model, checkpoint_path):
            break
    model.save_weights(os.path.join(cfg.exp_dir, "final_all_phases.weights.h5"))
    return model


def calibrate_temperature(model: tf.keras.Model, cfg: MaskAblationConfig, splits: Dict[str, pd.DataFrame]) -> float:
    calibration_dataset = build_eval_ds(cfg, splits["val_calib"])
    raw_calibration_probabilities, calibration_labels = predict_proba(model, calibration_dataset)
    if len(calibration_labels) == 0:
        calibration_summary = {"temperature": 1.0, "calibration_nll": None, "note": "val_calib is empty"}
        with open(os.path.join(cfg.exp_dir, "calibration_summary.json"), "w", encoding="utf-8") as file_handle:
            json.dump(calibration_summary, file_handle, indent=2, ensure_ascii=False)
        return 1.0

    uncalibrated_metrics = compute_metrics(calibration_labels, raw_calibration_probabilities, num_classes=cfg.num_classes)
    temperature, best_calibration_nll = temperature_search_nll(raw_calibration_probabilities, calibration_labels)
    calibrated_probabilities = apply_temperature(raw_calibration_probabilities, temperature)
    calibrated_metrics = compute_metrics(calibration_labels, calibrated_probabilities, num_classes=cfg.num_classes)
    calibration_summary = {
        "dataset": cfg.dataset_name,
        "variant": cfg.variant,
        "seed": int(cfg.seed),
        "temperature": float(temperature),
        "calibration_nll": float(best_calibration_nll),
        "raw_val_calib": uncalibrated_metrics,
        "temperature_scaled_val_calib": calibrated_metrics,
        "note": "Temperature fitted using val_calib only. val_eval/int_test/hold_out are not used to fit T.",
    }
    with open(os.path.join(cfg.exp_dir, "calibration_summary.json"), "w", encoding="utf-8") as file_handle:
        json.dump(calibration_summary, file_handle, indent=2, ensure_ascii=False)
    print(f"[CALIB] T={temperature:.2f} | NLL_calib={best_calibration_nll:.4f}")
    return float(temperature)


def load_temperature(exp_dir: str) -> float:
    calibration_summary_path = os.path.join(exp_dir, "calibration_summary.json")
    if not os.path.isfile(calibration_summary_path):
        return 1.0
    with open(calibration_summary_path, "r", encoding="utf-8") as file_handle:
        calibration_data = json.load(file_handle)
    return float(calibration_data.get("temperature", 1.0))


def sanitize_dataset_name(dataset_name: str) -> str:
    sanitized_name = str(dataset_name or "DATASET").strip()
    sanitized_name = re.sub(r"[^A-Za-z0-9._-]+", "_", sanitized_name).strip("._-")
    return sanitized_name or "DATASET"


def make_exp_dir(exp_root: str, dataset_name: str, variant: str, seed: int) -> str:
    dataset_slug = sanitize_dataset_name(dataset_name)
    experiment_directory = os.path.join(os.path.abspath(os.path.expanduser(exp_root)), f"{dataset_slug}_{variant}_seed{int(seed)}")
    os.makedirs(experiment_directory, exist_ok=True)
    return experiment_directory


def export_tflite(model: tf.keras.Model, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        serialized_tflite_model = converter.convert()
        with open(os.path.join(out_dir, "model_fp16.tflite"), "wb") as file_handle:
            file_handle.write(serialized_tflite_model)
        print("[TFLite] FP16 export completed.")
    except Exception as exc:
        print("[TFLite] FP16 export failed:", exc)


# -------------------------
# CLI
# -------------------------

def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train/evaluate EfficientNetB0 for retinal mask ablation study.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_root", default="experiments")
    parser.add_argument("--dataset_name", default=None, help="Logical dataset name; defaults to the dataset_root name.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampling_policy", default="class_balanced", choices=VALID_SAMPLING)
    parser.add_argument("--augmentation_policy", default="mild_photometric", choices=VALID_AUGMENTATION)
    parser.add_argument("--loss_policy", default="ce_label_smoothing", choices=VALID_LOSS)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--enable_mixup", action="store_true")
    parser.add_argument("--enable_cutmix", action="store_true")
    parser.add_argument("--use_tta_main", action="store_true")
    parser.add_argument("--enable_clahe", action="store_true", help="Supplementary analysis only. Default: disabled.")
    parser.add_argument("--validate_readable_images", action="store_true")

    parser.add_argument("--epochs_a", type=int, default=8)
    parser.add_argument("--epochs_b", type=int, default=20)
    parser.add_argument("--epochs_c", type=int, default=60)
    parser.add_argument("--epochs_d", type=int, default=30)
    parser.add_argument("--patience_a", type=int, default=5)
    parser.add_argument("--patience_b", type=int, default=6)
    parser.add_argument("--patience_c", type=int, default=8)
    parser.add_argument("--patience_d", type=int, default=8)
    parser.add_argument("--lr_a", type=float, default=5e-4)
    parser.add_argument("--lr_b", type=float, default=1.5e-5)
    parser.add_argument("--min_lr_b", type=float, default=3e-6)
    parser.add_argument("--lr_c", type=float, default=1.2e-5)
    parser.add_argument("--min_lr_c", type=float, default=3e-6)
    parser.add_argument("--lr_d", type=float, default=6e-6)
    parser.add_argument("--min_lr_d", type=float, default=2.4e-6)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1200)
    parser.add_argument("--save_every_a", type=int, default=2)
    parser.add_argument("--save_every_b", type=int, default=2)
    parser.add_argument("--save_every_c", type=int, default=2)
    parser.add_argument("--save_every_d", type=int, default=3)
    parser.add_argument("--no_tflite_export", action="store_true")

    parser.add_argument("--evaluate_only", action="store_true", help="Doesn't train; loads --weights and evaluates splits.")
    parser.add_argument("--weights", default=None, help="Path to .h5 weights for evaluate_only or re-evaluation.")
    parser.add_argument("--eval_splits", nargs="+", default=["val_eval", "int_test", "hold_out"],
                   choices=["val_eval", "int_test", "hold_out", "val_calib"])
    add_device_cli_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    operation = "internal_evaluation" if args.evaluate_only else "training"
    compute_device_info = configure_environment(
        args.seed,
        device=args.device,
        gpu_index=args.gpu_index,
        operation=operation,
    )
    dataset_name = args.dataset_name or os.path.basename(os.path.abspath(os.path.expanduser(args.dataset_root))) or "DATASET"
    experiment_directory = make_exp_dir(args.exp_root, dataset_name, args.variant, args.seed)

    selected_loss_policy = "ce_plus_focal" if args.use_focal else args.loss_policy
    experiment_config = MaskAblationConfig(
        dataset_root=args.dataset_root,
        dataset_name=dataset_name,
        variant=args.variant,
        labels_csv=args.labels_csv,
        exp_dir=experiment_directory,
        seed=args.seed,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        sampling_policy=args.sampling_policy,
        augmentation_policy=args.augmentation_policy,
        loss_policy=selected_loss_policy,
        use_class_weight=bool(args.use_class_weight),
        use_focal=bool(selected_loss_policy == "ce_plus_focal"),
        enable_mixup=bool(args.enable_mixup),
        enable_cutmix=bool(args.enable_cutmix),
        use_tta_main=bool(args.use_tta_main),
        enable_clahe=bool(args.enable_clahe),
        validate_readable_images=bool(args.validate_readable_images),
    )
    logger = setup_logging(experiment_config)
    write_experiment_config(experiment_config)
    with open(os.path.join(experiment_config.exp_dir, "compute_device.json"), "w", encoding="utf-8") as file_handle:
        json.dump(compute_device_info, file_handle, indent=2, ensure_ascii=False, default=str)

    dataset_splits = split_dataframes(experiment_config, logger)
    with open(os.path.join(experiment_config.exp_dir, "split_counts.json"), "w", encoding="utf-8") as file_handle:
        json.dump({split_key: {"n": len(split_dataframe), "class_counts": class_counts(split_dataframe, experiment_config.num_classes)} for split_key, split_dataframe in dataset_splits.items()},
                  file_handle, indent=2, ensure_ascii=False)

    model = build_model_for_inference(num_classes=experiment_config.num_classes, img_size=experiment_config.img_size)

    if args.evaluate_only:
        weights = args.weights or os.path.join(experiment_config.exp_dir, "final_all_phases.weights.h5")
        if not try_load_weights(model, weights):
            raise FileNotFoundError(f"Failed to load weights for evaluate_only: {weights}")
        temperature = load_temperature(experiment_config.exp_dir)
    else:
        model = train_model(experiment_config, dataset_splits, args)
        temperature = calibrate_temperature(model, experiment_config, dataset_splits)
        if not args.no_tflite_export:
            export_tflite(model, os.path.join(experiment_config.exp_dir, "tflite_export"))

    evaluation_rows = evaluate_splits(model, dataset_splits, experiment_config, temperature=temperature, split_names=args.eval_splits)
    pd.DataFrame(evaluation_rows).to_csv(os.path.join(experiment_config.exp_dir, "metrics_all_requested_splits.csv"), index=False)
    print(f"[DONE] Artifacts saved to: {experiment_config.exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
