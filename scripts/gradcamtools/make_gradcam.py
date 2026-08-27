from __future__ import annotations

import os
import json
import math
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import tensorflow as tf

K = tf.keras.backend


def discover_io_layers(model: tf.keras.Model) -> Dict[str, str]:
    layer_names = {layer.name for layer in model.layers}

    input_name = "input_rgb255"
    preproc_name = "preproc"
    target_conv_name = "top_conv"
    probs_name = "probs"

    missing_layers = []
    for expected_name in [
        input_name,
        preproc_name,
        target_conv_name,
        probs_name,
    ]:
        if expected_name not in layer_names:
            missing_layers.append(expected_name)

    if missing_layers:
        raise ValueError(
            "Expected layers were not found in the model: %s.\n"
            "Make sure the layer names match or modify make_gradcam.py "
            "to use the actual names." % ", ".join(missing_layers)
        )

    return dict(
        input_name=input_name,
        preproc_name=preproc_name,
        target_conv_name=target_conv_name,
        probs_name=probs_name,
    )


def _compute_gradcam(
    model: tf.keras.Model,
    img_rgb255: np.ndarray,
    class_index: Optional[int] = None,
    target_conv_name: str = "top_conv",
    use_guided_relu: bool = False,
) -> Tuple[np.ndarray, int, float]:
    image = img_rgb255.astype(np.float32)

    if image.max() <= 1.0:
        image *= 255.0

    input_tensor = tf.convert_to_tensor(
        image[None, ...],
        dtype=tf.float32,
    )

    layer_info = discover_io_layers(model)
    probabilities_output = model.get_layer(
        layer_info["probs_name"]
    ).output

    target_layer_name = (
        target_conv_name
        or layer_info["target_conv_name"]
    )
    target_layer_output = model.get_layer(
        target_layer_name
    ).output

    gradient_model = tf.keras.Model(
        inputs=model.get_layer(
            layer_info["input_name"]
        ).input,
        outputs=[
            target_layer_output,
            probabilities_output,
        ],
    )

    with tf.GradientTape() as tape:
        convolutional_features, predictions = gradient_model(
            input_tensor,
            training=False,
        )

        if class_index is None:
            class_index = int(
                tf.argmax(
                    predictions[0]
                ).numpy()
            )

        class_score = predictions[
            :,
            class_index,
        ]

    gradients = tape.gradient(
        class_score,
        convolutional_features,
    )

    if use_guided_relu:
        convolutional_features = tf.nn.relu(
            convolutional_features
        )
        gradients = tf.nn.relu(
            gradients
        )

    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    activation_map = tf.reduce_sum(
        channel_weights[:, None, None, :]
        * convolutional_features,
        axis=-1,
    )

    activation_map = (
        activation_map[0]
        .numpy()
    )
    activation_map = np.maximum(
        activation_map,
        0,
    )

    maximum_activation = activation_map.max()

    if maximum_activation > 1e-8:
        activation_map = activation_map / (
            maximum_activation + 1e-8
        )

    heatmap = cv2.resize(
        activation_map,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    class_probability = float(
        predictions[
            0,
            class_index,
        ].numpy()
    )

    return (
        heatmap,
        int(class_index),
        class_probability,
    )


def colorize_heatmap(
    heatmap01: np.ndarray,
) -> np.ndarray:
    normalized_heatmap = np.clip(
        heatmap01,
        0.0,
        1.0,
    )
    normalized_heatmap = (
        normalized_heatmap
        * 255.0
    ).astype(np.uint8)

    colored_heatmap = cv2.applyColorMap(
        normalized_heatmap,
        cv2.COLORMAP_JET,
    )

    return colored_heatmap


def overlay_heatmap(
    img_rgb255: np.ndarray,
    heatmap01: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    if img_rgb255.dtype != np.uint8:
        base_image = np.clip(
            img_rgb255,
            0,
            255,
        ).astype(np.uint8)
    else:
        base_image = img_rgb255

    if base_image.max() <= 1:
        base_image = (
            base_image.astype(np.float32)
            * 255.0
        ).astype(np.uint8)

    colored_heatmap_bgr = colorize_heatmap(
        heatmap01
    )
    base_image_bgr = cv2.cvtColor(
        base_image,
        cv2.COLOR_RGB2BGR,
    )

    overlay_bgr = cv2.addWeighted(
        base_image_bgr,
        1.0 - alpha,
        colored_heatmap_bgr,
        alpha,
        0.0,
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return overlay_rgb


def save_triplet(
    out_dir: str,
    stem: str,
    img_rgb255: np.ndarray,
    heatmap01: np.ndarray,
    overlay_rgb: np.ndarray,
) -> Dict[str, str]:
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    original_path = os.path.join(
        out_dir,
        f"{stem}_original.png",
    )
    heatmap_path = os.path.join(
        out_dir,
        f"{stem}_heatmap.png",
    )
    overlay_path = os.path.join(
        out_dir,
        f"{stem}_overlay.png",
    )

    cv2.imwrite(
        original_path,
        cv2.cvtColor(
            np.clip(
                img_rgb255,
                0,
                255,
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        ),
    )
    cv2.imwrite(
        heatmap_path,
        colorize_heatmap(
            heatmap01
        ),
    )
    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            overlay_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    return dict(
        original=original_path,
        heatmap=heatmap_path,
        overlay=overlay_path,
    )


def gradcam_single(
    model: tf.keras.Model,
    img_rgb255: np.ndarray,
    class_index: Optional[int] = None,
    target_conv_name: str = "top_conv",
    alpha_overlay: float = 0.35,
    use_guided_relu: bool = False,
) -> Dict[str, object]:
    heatmap, selected_class, class_probability = _compute_gradcam(
        model=model,
        img_rgb255=img_rgb255,
        class_index=class_index,
        target_conv_name=target_conv_name,
        use_guided_relu=use_guided_relu,
    )

    overlay_rgb = overlay_heatmap(
        img_rgb255,
        heatmap,
        alpha=alpha_overlay,
    )

    return dict(
        heatmap=heatmap,
        overlay=overlay_rgb,
        cls=selected_class,
        prob=class_probability,
    )


def _load_image_224_rgb(
    path: str,
) -> np.ndarray:
    image_bgr = cv2.imread(
        path,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )
    image_rgb = cv2.resize(
        image_rgb,
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )

    return image_rgb.astype(
        np.uint8
    )


def _try_build_model_from_train_script(
    weights_path: str,
) -> tf.keras.Model:
    try:
        import train_efficientnet_b0_v6 as training_module
    except Exception as error:
        raise RuntimeError(
            "Could not import 'train_efficientnet_b0_v8'. "
            "To use this smoke test, make sure you run it from the project root "
            "and that train_efficientnet_b0_v8.py exists. "
            "In any case, this file is intended as a toolbox; "
            "the batch runner will build the model and call these functions."
        ) from error

    model_builder = None

    for builder_name in (
        "build_model_for_inference",
        "build_model",
        "get_model",
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
            "The 'train_efficientnet_b0_v8.py' script does not expose any public "
            "function for building the model "
            "(build_model_for_inference/build_model/get_model). "
            "Add one or build the model manually and pass it to these utilities."
        )

    model = model_builder()
    model.load_weights(
        weights_path
    )

    return model


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grad-CAM smoke test for a single image."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "Path to .h5 weights "
            "(e.g. experiments/.../best_phaseD.weights.h5)"
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help=(
            "Image to explain "
            "(resized to 224x224)"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Directory where original/heatmap/overlay "
            "will be saved"
        ),
    )
    parser.add_argument(
        "--class_index",
        type=int,
        default=None,
        help=(
            "Target class. By default, the predicted "
            "class (argmax) is used."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay alpha (0..1)",
    )
    parser.add_argument(
        "--use_guided_relu",
        action="store_true",
        help=(
            "Applies guided ReLU to convolutional "
            "activations and gradients."
        ),
    )
    parser.add_argument(
        "--target_conv",
        type=str,
        default="top_conv",
        help=(
            "Name of the target convolutional layer "
            "(default: top_conv)."
        ),
    )

    args = parser.parse_args()

    model = _try_build_model_from_train_script(
        args.weights
    )

    image_rgb = _load_image_224_rgb(
        args.image
    )

    result = gradcam_single(
        model,
        img_rgb255=image_rgb,
        class_index=args.class_index,
        target_conv_name=args.target_conv,
        alpha_overlay=args.alpha,
        use_guided_relu=args.use_guided_relu,
    )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    image_stem = os.path.splitext(
        os.path.basename(
            args.image
        )
    )[0]

    output_paths = save_triplet(
        args.out_dir,
        image_stem,
        image_rgb,
        result["heatmap"],
        result["overlay"],
    )

    metadata = dict(
        image=args.image,
        class_used=int(
            result["cls"]
        ),
        prob=float(
            result["prob"]
        ),
        **output_paths,
    )

    with open(
        os.path.join(
            args.out_dir,
            f"{image_stem}_meta.json",
        ),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    _cli()
