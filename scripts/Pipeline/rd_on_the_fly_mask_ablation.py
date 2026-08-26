from __future__ import annotations

import os
import sys
import json
import math
import random
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

VALID_VARIANTS = ("no_mask", "hard_mask", "fixed_feather", "adaptive_feather")
VALID_USAGE = ("train", "val_calib", "val_eval", "int_test", "hold_out")

USAGE_FOLDER_ALIASES = {
    "train": ["train"],
    "val_calib": ["val_calib", "val"],
    "val_eval": ["val_eval", "val"],
    "int_test": ["int_test"],
    "hold_out": ["hold_out", "hold_hout"],
}
VALID_SAMPLING = ("natural", "class_balanced", "sqrt_balanced")
VALID_AUGMENTATION = ("none", "mild_photometric")
VALID_LOSS = ("ce", "ce_label_smoothing", "ce_plus_focal")


@dataclass
class MaskAblationConfig:

    # INPUTS

    dataset_root: str = r"C:\Articulo\Datasets\DATASET"
    dataset_name: str = "DATASET"
    variant: str = "adaptive_feather"
    labels_csv: Optional[str] = None
    exp_dir: Optional[str] = None

    # EXPERIMENT IDENTITY

    seed: int = 42
    architecture: str = "EfficientNetB0"
    img_size: Tuple[int, int] = (224, 224)
    num_classes: int = 5

    # SPLITS BY USAGE

    usage_train: str = "train"
    usage_val_calib: str = "val_calib"
    usage_val_eval: str = "val_eval"
    usage_int_test: str = "int_test"
    usage_hold_out: str = "hold_out"
    require_usage_column: bool = True

    # SEPARATE POLICIES

    sampling_policy: str = "class_balanced"
    augmentation_policy: str = "mild_photometric"
    loss_policy: str = "ce_label_smoothing"
    use_class_weight: bool = False
    use_focal: bool = False
    enable_mixup: bool = False
    enable_cutmix: bool = False
    use_tta_main: bool = False

    # GEOMETRIC AUGMENTATION

    flip_horizontal: bool = False
    rot_180: bool = False
    random_rotation: bool = False
    random_zoom: bool = False
    random_crop: bool = False

    # CLAHE

    enable_clahe: bool = False
    clahe_clip_limit: float = 1.2
    clahe_tile_grid: Tuple[int, int] = (16, 16)

    # MILD PHOTOMETRIC AUGMENTATION

    brightness_delta: float = 0.03
    contrast_lower: float = 0.92
    contrast_upper: float = 1.08
    saturation_lower: float = 0.94
    saturation_upper: float = 1.06
    hue_delta: float = 0.01
    gaussian_noise_std: float = 0.003
    enable_gaussian_noise: bool = True

    # TENSORFLOW DATA

    batch_size: int = 32
    shuffle_buffer: int = 12000
    deterministic: bool = False
    cache_to_disk: bool = False
    cache_dir: Optional[str] = None

    # PATH RESOLUTION

    recursive_index_if_missing: bool = True
    duplicate_policy: str = "error" 
    validate_readable_images: bool = False

    # QA

    make_previews: bool = False

    def __post_init__(self):
        self.dataset_root = os.path.abspath(os.path.expanduser(str(self.dataset_root)))
        self.dataset_name = str(self.dataset_name or os.path.basename(self.dataset_root) or "DATASET").strip()
        if self.labels_csv is None:
            self.labels_csv = os.path.join(self.dataset_root, "labels.csv")
        else:
            self.labels_csv = os.path.abspath(os.path.expanduser(str(self.labels_csv)))

        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_VARIANTS}; received: {self.variant}")
        if self.sampling_policy not in VALID_SAMPLING:
            raise ValueError(f"sampling_policy must be one of {VALID_SAMPLING}; received: {self.sampling_policy}")
        if self.augmentation_policy not in VALID_AUGMENTATION:
            raise ValueError(f"augmentation_policy must be one of {VALID_AUGMENTATION}; received: {self.augmentation_policy}")
        if self.loss_policy not in VALID_LOSS:
            raise ValueError(f"loss_policy must be one of {VALID_LOSS}; received: {self.loss_policy}")
        if self.duplicate_policy not in ("error", "first"):
            raise ValueError("duplicate_policy must be 'error' or 'first'")

        self.img_size = tuple(int(x) for x in self.img_size)
        if self.exp_dir is not None:
            self.exp_dir = os.path.abspath(os.path.expanduser(str(self.exp_dir)))
            os.makedirs(self.exp_dir, exist_ok=True)
        if self.cache_dir is None:
            cache_base_dir = self.exp_dir if self.exp_dir else self.dataset_root
            self.cache_dir = os.path.join(cache_base_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def variant_dir(self) -> str:
        return os.path.join(self.dataset_root, self.variant)

    def to_experiment_dict(self) -> Dict[str, object]:
        experiment_dict = asdict(self)
        experiment_dict.update({
            "dataset_root": self.dataset_root,
            "dataset_name": self.dataset_name,
            "variant": self.variant,
            "labels_csv": self.labels_csv,
            "seed": int(self.seed),
            "img_size": list(self.img_size),
            "architecture": self.architecture,
            "enable_clahe": bool(self.enable_clahe),
            "augmentation_policy": self.augmentation_policy,
            "sampling_policy": self.sampling_policy,
            "loss_policy": self.loss_policy,
            "use_class_weight": bool(self.use_class_weight),
            "use_focal": bool(self.use_focal),
            "use_mixup": bool(self.enable_mixup),
            "use_cutmix": bool(self.enable_cutmix),
            "use_tta_main": bool(self.use_tta_main),
            "usage_train": self.usage_train,
            "usage_val_calib": self.usage_val_calib,
            "usage_val_eval": self.usage_val_eval,
            "usage_int_test": self.usage_int_test,
            "usage_hold_out": self.usage_hold_out,
            "fecha_hora_ejecucion": datetime.now().isoformat(timespec="seconds"),
        })
        return experiment_dict


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logging(cfg: MaskAblationConfig) -> logging.Logger:
    logger = logging.getLogger("RD-MaskAblation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    if cfg.exp_dir:
        os.makedirs(cfg.exp_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(cfg.exp_dir, "run.log"), mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    logger.info("Logger initialized for mask ablation.")
    if not cfg.enable_clahe:
        logger.info("CLAHE disabled for mask ablation study")
    else:
        logger.warning("CLAHE ENABLED: use only as a supplementary analysis, not as the main experiment.")
    logger.info("Config:\n" + json.dumps(cfg.to_experiment_dict(), indent=2, ensure_ascii=False, default=str))
    return logger


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df



def _build_recursive_filename_index(variant_dir: str, duplicate_policy: str = "error"):

    filename_index: Dict[str, str] = {}
    filename_duplicates: Dict[str, List[str]] = {}
    basename_index: Dict[str, List[str]] = {}

    for current_root, _subdirectories, filenames in os.walk(variant_dir):
        for current_filename in filenames:
            filename_key = os.path.basename(current_filename)
            file_path = os.path.join(current_root, current_filename)
            if filename_key in filename_index:
                filename_duplicates.setdefault(filename_key, [filename_index[filename_key]]).append(file_path)
            else:
                filename_index[filename_key] = file_path
            basename = os.path.splitext(filename_key)[0]
            basename_index.setdefault(basename, []).append(file_path)

    basename_duplicates = {duplicate_name: duplicate_paths for duplicate_name, duplicate_paths in basename_index.items() if len(duplicate_paths) > 1}

    if filename_duplicates:
        duplicate_message_lines = ["Duplicate filenames found within the variant. This creates experimental ambiguity:"]
        for duplicate_name, paths in list(filename_duplicates.items())[:20]:
            duplicate_message_lines.append(f"  - {duplicate_name}: " + " | ".join(paths[:5]))
        if len(filename_duplicates) > 20:
            duplicate_message_lines.append(f"  ... {len(filename_duplicates)-20} additional duplicates")
        duplicate_message = "\n".join(duplicate_message_lines)
        if duplicate_policy == "error":
            raise ValueError(duplicate_message)
        print("[WARN] " + duplicate_message)

    return filename_index, filename_duplicates, basename_index, basename_duplicates


def _usage_folder_aliases(usage: str) -> List[str]:
    usage = str(usage)
    return list(USAGE_FOLDER_ALIASES.get(usage, [usage]))


def _candidate_paths(cfg: MaskAblationConfig, filename: str, label: int, usage: str) -> List[str]:

    variant_directory = cfg.variant_dir
    usage_aliases = _usage_folder_aliases(str(usage))
    candidate_paths = [
        os.path.join(variant_directory, filename),
        os.path.join(variant_directory, str(label), filename),
    ]
    for usage_alias in usage_aliases:
        candidate_paths.append(os.path.join(variant_directory, usage_alias, str(label), filename))

    for usage_alias in usage_aliases:
        candidate_paths.append(os.path.join(variant_directory, usage_alias, filename))
    return candidate_paths


def _resolve_paths(df: pd.DataFrame, cfg: MaskAblationConfig, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.isdir(cfg.variant_dir):
        raise FileNotFoundError(f"Variant directory does not exist: {cfg.variant_dir}")

    df = df.copy()
    resolved_paths: List[Optional[str]] = []
    resolution_modes: List[str] = []
    missing_indices: List[int] = []
    resolved_exact = 0
    resolved_by_basename = 0
    extension_mismatch_count = 0
    ambiguous_basename: Dict[str, List[str]] = {}

    for row_index, row in df.iterrows():
        filename = str(row["filename"])
        label = int(row["label"])
        usage = str(row.get("usage", ""))
        resolved_path = None
        for candidate_path in _candidate_paths(cfg, filename, label, usage):
            if os.path.isfile(candidate_path):
                resolved_path = candidate_path
                break
        if resolved_path is None:
            missing_indices.append(row_index)
            resolved_paths.append(None)
            resolution_modes.append("missing")
        else:
            resolved_paths.append(resolved_path)
            resolution_modes.append("exact")
            resolved_exact += 1

    if missing_indices and cfg.recursive_index_if_missing:
        logger.info(f"Direct paths were not found for {len(missing_indices)} images. Building recursive filename and basename indexes...")
        recursive_filename_index, _filename_duplicates, basename_index, _ambiguous_basenames_index = _build_recursive_filename_index(
            cfg.variant_dir,
            duplicate_policy=cfg.duplicate_policy,
        )
        for row_index in missing_indices:
            row_position = df.index.get_loc(row_index)
            if resolved_paths[row_position] is not None:
                continue
            requested_filename = str(df.loc[row_index, "filename"])
            if requested_filename in recursive_filename_index:
                resolved_paths[row_position] = recursive_filename_index[requested_filename]
                resolution_modes[row_position] = "recursive_exact"
                resolved_exact += 1
                continue

            requested_basename = os.path.splitext(os.path.basename(requested_filename))[0]
            basename_matches = basename_index.get(requested_basename, [])
            if len(basename_matches) == 1:
                resolved_paths[row_position] = basename_matches[0]
                resolution_modes[row_position] = "basename"
                resolved_by_basename += 1
                if os.path.splitext(os.path.basename(basename_matches[0]))[1].lower() != os.path.splitext(requested_filename)[1].lower():
                    extension_mismatch_count += 1
            elif len(basename_matches) > 1:
                ambiguous_basename[requested_filename] = basename_matches[:10]

    df["abs_path"] = resolved_paths
    df["path_resolution_mode"] = resolution_modes
    missing_rows = df[df["abs_path"].isna()].copy()
    if len(missing_rows) > 0:
        missing_sample = missing_rows[["filename", "label", "usage"]].head(20).to_string(index=False)
        ambiguity_details = ""
        if ambiguous_basename:
            ambiguity_message_lines = ["\nAmbiguous basenames detected:"]
            for ambiguous_filename, ambiguous_paths in list(ambiguous_basename.items())[:10]:
                ambiguity_message_lines.append(f"  - {ambiguous_filename}: " + " | ".join(ambiguous_paths[:5]))
            ambiguity_details = "\n" + "\n".join(ambiguity_message_lines)
        raise FileNotFoundError(
            f"Could not find {len(missing_rows)} images in variant '{cfg.variant}'. "
            f"Examples:\n{missing_sample}{ambiguity_details}"
        )

    df["abs_path"] = df["abs_path"].astype(str)
    logger.info(
        f"Paths successfully resolved for variant '{cfg.variant}': {len(df)} images | "
        f"exact={resolved_exact}, basename={resolved_by_basename}, extension_mismatch={extension_mismatch_count}"
    )
    if resolved_by_basename > 0:
        logger.warning(
            f"Resolved {resolved_by_basename} images by basename while ignoring the extension. "
            f"It is recommended to verify consistency between labels.csv and the physical files."
        )
    return df

def _validate_readable(df: pd.DataFrame, cfg: MaskAblationConfig, logger: logging.Logger) -> pd.DataFrame:
    if not cfg.validate_readable_images:
        return df
    readable_rows = []
    unreadable_paths = []
    for row in df.itertuples(index=False):
        image_path = getattr(row, "abs_path")
        decoded_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded_image is None:
            unreadable_paths.append(image_path)
        else:
            readable_rows.append(row._asdict())
    if unreadable_paths:
        logger.warning(
            f"Unreadable images: {len(unreadable_paths)}. "
            f"First examples: {unreadable_paths[:10]}"
        )
    return pd.DataFrame(readable_rows) if unreadable_paths else df


def load_labels_dataframe(cfg: MaskAblationConfig, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("RD-MaskAblation")
    if not os.path.isfile(cfg.labels_csv):
        raise FileNotFoundError(f"labels_csv was not found: {cfg.labels_csv}")

    df = pd.read_csv(cfg.labels_csv)
    df = _clean_columns(df)

    if "label" not in df.columns:
        if "level" in df.columns:
            df["label"] = df["level"]
            logger.info("Column 'label' was not found: label = level was created internally.")
        else:
            raise ValueError("labels.csv must contain either 'label' or 'level'.")

    if "dataset" not in df.columns:
        df["dataset"] = cfg.dataset_name
        logger.info(f"Column 'dataset' was not found: dataset = '{cfg.dataset_name}' was created internally.")

    required = {"filename", "label", "dataset"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"labels.csv does not contain the required columns: {missing_columns}")

    if cfg.require_usage_column and "usage" not in df.columns:
        raise ValueError("This experimental design requires a 'usage' column. No random split will be performed.")

    if "usage" not in df.columns:
        raise ValueError("The 'usage' column does not exist. This pipeline does not create a random split by default.")

    df["filename"] = df["filename"].astype(str)
    df["label"] = df["label"].astype(int)
    df["dataset"] = df["dataset"].astype(str)
    df["usage"] = df["usage"].astype(str)

    invalid_labels = sorted(set(df["label"].unique()) - set(range(cfg.num_classes)))
    if invalid_labels:
        raise ValueError(f"Labels outside the valid range 0..{cfg.num_classes-1}: {invalid_labels}")

    expected_usage_values = {cfg.usage_train, cfg.usage_val_calib, cfg.usage_val_eval, cfg.usage_int_test, cfg.usage_hold_out}
    invalid_usage_values = sorted(set(df["usage"].unique()) - expected_usage_values)
    if invalid_usage_values:
        raise ValueError(f"Unexpected usage values: {invalid_usage_values}. Expected: {sorted(expected_usage_values)}")

    if "brightness_category" not in df.columns:
        df["brightness_category"] = "unknown"
    df["brightness_category"] = df["brightness_category"].fillna("unknown").astype(str)

    if "p50" not in df.columns:
        df["p50"] = np.nan
    df["p50"] = pd.to_numeric(df["p50"], errors="coerce")

    df = _resolve_paths(df, cfg, logger)
    df = _validate_readable(df, cfg, logger)

    logger.info("Count by usage:\n" + df["usage"].value_counts().to_string())
    logger.info("Count by class:\n" + df["label"].value_counts().sort_index().to_string())
    return df.reset_index(drop=True)


def get_split_dataframe(df: pd.DataFrame, usage_value: str) -> pd.DataFrame:
    split_dataframe = df[df["usage"] == usage_value].copy().reset_index(drop=True)
    if len(split_dataframe) == 0:
        raise ValueError(f"Empty split for usage='{usage_value}'.")
    return split_dataframe


def split_dataframes(cfg: MaskAblationConfig, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    labels_dataframe = load_labels_dataframe(cfg, logger)
    return {
        "train": get_split_dataframe(labels_dataframe, cfg.usage_train),
        "val_calib": get_split_dataframe(labels_dataframe, cfg.usage_val_calib),
        "val_eval": get_split_dataframe(labels_dataframe, cfg.usage_val_eval),
        "int_test": get_split_dataframe(labels_dataframe, cfg.usage_int_test),
        "hold_out": get_split_dataframe(labels_dataframe, cfg.usage_hold_out),
    }


def class_counts(df: pd.DataFrame, num_classes: int = 5) -> Dict[int, int]:
    return df.groupby("label")["filename"].count().reindex(range(num_classes), fill_value=0).astype(int).to_dict()


def _decode_and_to_float(path: tf.Tensor, cfg: MaskAblationConfig) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, cfg.img_size, method=tf.image.ResizeMethod.BILINEAR)
    image.set_shape([cfg.img_size[0], cfg.img_size[1], 3])
    return image


def _apply_clahe_tf(image: tf.Tensor, cfg: MaskAblationConfig) -> tf.Tensor:
    if not cfg.enable_clahe:
        return image

    def _clahe_fn(img_f32: np.ndarray) -> np.ndarray:
        image_uint8 = np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)
        lab_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        lightness_channel, channel_a, channel_b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=tuple(cfg.clahe_tile_grid))
        equalized_lightness = clahe.apply(lightness_channel)
        equalized_lab_image = cv2.merge([equalized_lightness, channel_a, channel_b])
        equalized_rgb_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2RGB)
        return equalized_rgb_image.astype(np.float32) / 255.0

    clahe_image = tf.numpy_function(_clahe_fn, [image], tf.float32)
    clahe_image.set_shape([cfg.img_size[0], cfg.img_size[1], 3])
    return clahe_image


def _mild_photometric_augment(image: tf.Tensor, cfg: MaskAblationConfig) -> tf.Tensor:
    image = tf.image.random_brightness(image, max_delta=float(cfg.brightness_delta))
    image = tf.image.random_contrast(image, lower=float(cfg.contrast_lower), upper=float(cfg.contrast_upper))
    image = tf.image.random_saturation(image, lower=float(cfg.saturation_lower), upper=float(cfg.saturation_upper))
    if cfg.hue_delta and cfg.hue_delta > 0:
        image = tf.image.random_hue(image, max_delta=float(cfg.hue_delta))
    if cfg.enable_gaussian_noise and cfg.gaussian_noise_std > 0:
        gaussian_noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=float(cfg.gaussian_noise_std), dtype=tf.float32)
        image = image + gaussian_noise
    return tf.clip_by_value(image, 0.0, 1.0)


def _preprocess(image: tf.Tensor, label: tf.Tensor, training: bool, cfg: MaskAblationConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    if training and cfg.augmentation_policy == "mild_photometric":
        image = _mild_photometric_augment(image, cfg)
    # Geometric augmentations are not applied by default. Config fields are kept only for traceability.
    image = _apply_clahe_tf(image, cfg)
    image = tf.clip_by_value(image, 0.0, 1.0) * 255.0
    label = tf.cast(label, tf.int32)
    return image, label


def _dataset_from_df(df: pd.DataFrame, cfg: MaskAblationConfig, training: bool, repeat: bool = False) -> tf.data.Dataset:
    image_paths = df["abs_path"].values.astype(str)
    class_labels = df["label"].values.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_labels))
    if training:
        dataset = dataset.shuffle(buffer_size=min(max(len(image_paths), 1), int(cfg.shuffle_buffer)),
                        seed=cfg.seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()

    def _load(path, lbl):
        image = _decode_and_to_float(path, cfg)
        return _preprocess(image, lbl, training=training, cfg=cfg)

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=cfg.deterministic if training else True)
    return dataset


def _per_class_repeated_dataset(df_train: pd.DataFrame, class_id: int, cfg: MaskAblationConfig) -> tf.data.Dataset:
    class_dataframe = df_train[df_train["label"] == class_id].copy()
    if len(class_dataframe) == 0:
        raise ValueError(f"No class {class_id} images are available in train; balanced sampling cannot be used.")
    return _dataset_from_df(class_dataframe, cfg, training=True, repeat=True)


def _sampling_weights(df_train: pd.DataFrame, cfg: MaskAblationConfig) -> List[float]:
    class_sample_counts = np.array([max(1, int((df_train["label"] == c).sum())) for c in range(cfg.num_classes)], dtype=np.float64)
    if cfg.sampling_policy == "class_balanced":
        sampling_weights = np.ones_like(class_sample_counts, dtype=np.float64)
    elif cfg.sampling_policy == "sqrt_balanced":
        sampling_weights = 1.0 / np.sqrt(class_sample_counts)
    else:
        sampling_weights = class_sample_counts / class_sample_counts.sum()
    sampling_weights = sampling_weights / sampling_weights.sum()
    return sampling_weights.astype(float).tolist()


def build_train_ds(cfg: MaskAblationConfig, df_train: Optional[pd.DataFrame] = None) -> Tuple[tf.data.Dataset, Dict[int, float], pd.DataFrame]:
    set_global_seed(cfg.seed)
    logger = logging.getLogger("RD-MaskAblation")
    if df_train is None:
        df_train = split_dataframes(cfg, logger)["train"]

    training_class_counts = class_counts(df_train, cfg.num_classes)
    total_training_samples = float(len(df_train))
    class_weights = {int(class_id): float(total_training_samples / (cfg.num_classes * max(1, training_class_counts[class_id]))) for class_id in range(cfg.num_classes)}
    mean_class_weight = float(np.mean(list(class_weights.values())))
    class_weights = {class_id: float(class_weight_value / mean_class_weight) for class_id, class_weight_value in class_weights.items()}
    logger.info(f"Training count by class: {training_class_counts}")
    logger.info(f"class_weight calculated for reference only: {class_weights}")

    if cfg.sampling_policy == "natural":
        training_dataset = _dataset_from_df(df_train, cfg, training=True, repeat=True)
    else:
        per_class_datasets = [_per_class_repeated_dataset(df_train, class_id, cfg) for class_id in range(cfg.num_classes)]
        sampling_weights = _sampling_weights(df_train, cfg)
        logger.info(f"Sampling policy='{cfg.sampling_policy}' | weights={sampling_weights}")
        training_dataset = tf.data.Dataset.sample_from_datasets(per_class_datasets, weights=sampling_weights, seed=cfg.seed)
        training_dataset = training_dataset.shuffle(cfg.shuffle_buffer, seed=cfg.seed, reshuffle_each_iteration=True)

    training_dataset = training_dataset.batch(cfg.batch_size, drop_remainder=False)
    
    if cfg.cache_to_disk:
        training_dataset = training_dataset.cache(os.path.join(cfg.cache_dir, f"train_{cfg.variant}_{cfg.seed}.cache"))
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
    return training_dataset, class_weights, df_train


def build_eval_ds(cfg: MaskAblationConfig, df_eval: pd.DataFrame) -> tf.data.Dataset:
    set_global_seed(cfg.seed)
    evaluation_dataset = _dataset_from_df(df_eval, cfg, training=False, repeat=False)
    evaluation_dataset = evaluation_dataset.batch(cfg.batch_size, drop_remainder=False)
    evaluation_dataset = evaluation_dataset.prefetch(tf.data.AUTOTUNE)
    return evaluation_dataset


def build_split_ds(cfg: MaskAblationConfig, split_name: str, splits: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    if splits is None:
        splits = split_dataframes(cfg)
    if split_name not in splits:
        raise ValueError(f"Invalid split_name: {split_name}. Options: {sorted(splits)}")
    split_dataframe = splits[split_name]
    if split_name == "train":
        split_dataset, _unused_class_weights, split_dataframe = build_train_ds(cfg, split_dataframe)
    else:
        split_dataset = build_eval_ds(cfg, split_dataframe)
    return split_dataset, split_dataframe


def estimate_steps_per_epoch(df_train: pd.DataFrame, cfg: MaskAblationConfig, max_steps: int = 1200) -> int:
    num_training_samples = max(1, len(df_train))
    return int(min(math.ceil(num_training_samples / float(cfg.batch_size)), max_steps))


def write_experiment_config(cfg: MaskAblationConfig, path: Optional[str] = None) -> str:
    if path is None:
        if not cfg.exp_dir:
            raise ValueError("cfg.exp_dir is required when path is not provided.")
        path = os.path.join(cfg.exp_dir, "experiment_config.json")
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(cfg.to_experiment_dict(), config_file, indent=2, ensure_ascii=False, default=str)
    return path
