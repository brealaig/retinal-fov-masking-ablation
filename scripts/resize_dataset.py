"""
Resize retinal images to one or more square resolutions using letterboxing.

The original aspect ratio is preserved. Any remaining area is padded with
black pixels, matching the resizing procedure used in the ablation study.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


PAD_COLOR = (0, 0, 0)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Resize retinal images to one or more square resolutions "
            "using aspect-ratio-preserving letterboxing."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Root directory containing the input images."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root directory where resized datasets will be saved."
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help=(
            "One or more square output resolutions in pixels. "
            "Example: --sizes 224 512"
        )
    )

    return parser.parse_args()


def letterbox_resize(img, target_size, pad_color=PAD_COLOR):
    """Resize an image to a square canvas while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full(
        (target_size, target_size, 3),
        pad_color,
        dtype=np.uint8
    )

    x0 = (target_size - new_w) // 2
    y0 = (target_size - new_h) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def collect_images(root):
    """Collect supported image files recursively."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def validate_args(images_dir, sizes):
    if not images_dir.is_dir():
        raise NotADirectoryError(
            f"Input image directory does not exist: {images_dir}"
        )

    invalid_sizes = [size for size in sizes if size <= 0]
    if invalid_sizes:
        raise ValueError(
            f"All output sizes must be positive integers. "
            f"Invalid values: {invalid_sizes}"
        )


def main():
    args = parse_args()

    source_dir = args.images
    output_root = args.output
    sizes = sorted(set(args.sizes))

    validate_args(source_dir, sizes)

    files = collect_images(source_dir)
    if not files:
        raise RuntimeError(
            f"No supported images were found in: {source_dir}"
        )

    print(f"Images found: {len(files)}")
    print(f"Output sizes: {', '.join(str(size) for size in sizes)}")

    output_dirs = {
        size: output_root / f"r{size}"
        for size in sizes
    }

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = 0

    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"Warning: could not read image: {path}")
            errors += 1
            continue

        relative_path = path.relative_to(source_dir)

        try:
            for size in sizes:
                resized = letterbox_resize(
                    img,
                    target_size=size,
                    pad_color=PAD_COLOR
                )

                output_path = output_dirs[size] / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                success = cv2.imwrite(
                    str(output_path),
                    resized
                )

                if not success:
                    raise IOError(
                        f"Could not save image: {output_path}"
                    )

            processed += 1

        except Exception as exc:
            errors += 1
            print(f"Error processing {relative_path}: {exc}")

    print("\nResize completed.")
    print(f"Images processed successfully: {processed}")
    print(f"Images with errors:            {errors}")
    print(f"Output root:                   {output_root}")


if __name__ == "__main__":
    cv2.setUseOptimized(True)
    main()