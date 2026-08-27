import cv2, numpy as np, sys, glob
from pathlib import Path


def is_suspicious_double_clahe(img_bgr, grid=16, cv_hi=0.85):
    lab_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lightness_channel = lab_image[:, :, 0].astype(np.float32)
    height, width = lightness_channel.shape
    patch_height, patch_width = height // grid, width // grid

    if patch_height < 1 or patch_width < 1:
        return False, float("nan")

    variation_coefficients = []
    for row_index in range(grid):
        for column_index in range(grid):
            patch = lightness_channel[
                row_index * patch_height:(row_index + 1) * patch_height,
                column_index * patch_width:(column_index + 1) * patch_width,
            ]
            if patch.size == 0:
                continue
            patch_mean, patch_std = patch.mean(), patch.std()
            if patch_mean > 1e-3:
                variation_coefficients.append(patch_std / (patch_mean + 1e-6))

    if len(variation_coefficients) == 0:
        return False, float("nan")

    variation_coefficients = np.array(variation_coefficients)
    cv90 = float(np.percentile(variation_coefficients, 90))
    return (cv90 > cv_hi), cv90


if __name__ == "__main__":
    input_folder = Path(sys.argv[1])
    image_paths = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths += glob.glob(str(input_folder / extension))

    suspicious_count = 0
    for image_path in image_paths:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        is_suspicious, cv90 = is_suspicious_double_clahe(image_bgr)
        if is_suspicious:
            suspicious_count += 1
            print(f"[WARN] {image_path} possible double CLAHE | cv90={cv90:.3f}")

    print(f"Possible cases: {suspicious_count}/{len(image_paths)}")
