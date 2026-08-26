import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# CSV column containing the image name/path.
# Leave as None to auto-detect among:
# filename, image, img, path, file_path, rel_path, id_code, name
IMAGE_COL = None

# If the CSV contains id_code values without an extension, the script searches by stem:
# example: "000c1434d8d7" -> searches for 000c1434d8d7.jpg/png/jpeg...
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# If brightness_category or p50 columns already exist, overwrite them.
OVERWRITE_EXISTING = True

# P50 measurement
DISK_SHRINK_PCT = 0.08
BLUR_SIGMA_HOUGH = 5.0

# P50 ranges in the L/LAB channel inside the retinal disk
THRESH = [70, 85, 105, 135, 165]
LABELS = [
    "ExtremelyDark",
    "VeryDark",
    "Dark",
    "Normal",
    "Bright",
    "VeryBright"
]

# Bridge regions around the Normal category
BRIDGE_TO_DARK = 4.0
BRIDGE_TO_BRIGHT = 6.0


# ==================== UTILITIES ====================

def imread_bgr(p):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def normalize_key(x):
    return str(x).replace("\\", "/").strip().lower()


def collect_image_index(root):
    """
    Build a flexible index for locating images referenced by the CSV.

    Supports lookup by:
    - absolute path
    - relative path
    - filename with extension
    - stem without extension
    """
    root = Path(root)
    index = {}

    def add_key(key, path):
        key = normalize_key(key)
        if not key:
            return
        index.setdefault(key, []).append(path)

    files = []
    for ext in EXTS:
        files.extend(root.rglob(f"*{ext}"))

    files = sorted(files)

    for p in files:
        p = Path(p)
        try:
            rel = p.relative_to(root)
            add_key(rel.as_posix(), p)
        except Exception:
            pass

        add_key(p.name, p)
        add_key(p.stem, p)
        add_key(str(p), p)
        try:
            add_key(str(p.resolve()), p)
        except Exception:
            pass

    return index, files


def resolve_image_path(value, img_root, index):
    """
    Resolve an image path from the value stored in the CSV.
    Supports filenames, relative paths, absolute paths, and stems.
    """
    if pd.isna(value):
        return None, "empty_image_value"

    raw = str(value).strip()
    if raw == "":
        return None, "empty_image_value"

    root = Path(img_root)

    # 1) Direct absolute path or relative path from the current working directory
    p = Path(raw)
    if p.exists():
        return p, ""

    # 2) Path relative to IMG_ROOT
    p2 = root / raw
    if p2.exists():
        return p2, ""

    # 3) If no extension is provided, try the supported extensions
    raw_path = Path(raw)
    if raw_path.suffix == "":
        for ext in EXTS:
            candidate = root / f"{raw}{ext}"
            if candidate.exists():
                return candidate, ""

    # 4) Flexible lookup using the pre-built index
    keys = [
        raw,
        Path(raw).name,
        Path(raw).stem,
        normalize_key(raw),
    ]

    for k in keys:
        k = normalize_key(k)
        matches = index.get(k, [])
        if len(matches) == 1:
            return matches[0], ""
        elif len(matches) > 1:
            return matches[0], f"ambiguous_match_{len(matches)}"

    return None, "image_not_found"


def autodetect_image_col(df):
    candidates = [
        "filename",
        "file_name",
        "image",
        "image_name",
        "img",
        "path",
        "filepath",
        "file_path",
        "image_path",
        "rel_path",
        "id_code",
        "name",
    ]

    lower_to_original = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]

    raise ValueError(
        "Could not auto-detect the image column. "
        "Set IMAGE_COL manually to the name of the image column in the CSV."
    )


# ==================== ORIGINAL PROCESSING LOGIC ====================

def clamp_circle(cx, cy, r, h, w):
    r_min = int(0.20 * min(h, w))
    r_max = int(0.56 * min(h, w))
    r = int(np.clip(r, r_min, r_max))
    cx = int(np.clip(cx, r, w - r - 1))
    cy = int(np.clip(cy, r, h - r - 1))
    return cx, cy, r


def circle_by_nonblack(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    thr = max(5, int(np.percentile(v, 8)))
    m = (v > thr).astype(np.uint8) * 255

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    (x, y), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return int(x), int(y), int(r)


def circle_by_hough(bgr, blur_sigma=5.0):
    h, w = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=120,
        param2=28,
        minRadius=int(0.2 * min(h, w)),
        maxRadius=int(0.56 * min(h, w)),
    )

    if circles is None:
        return None

    c = np.squeeze(circles).astype(np.int32)

    if c.ndim == 1:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
    else:
        idx = np.argmax(c[:, 2])
        cx, cy, r = int(c[idx, 0]), int(c[idx, 1]), int(c[idx, 2])

    return cx, cy, r


def estimate_retina_circle(bgr):
    h, w = bgr.shape[:2]

    res = circle_by_nonblack(bgr)

    if res is None:
        res = circle_by_hough(bgr, BLUR_SIGMA_HOUGH)

    if res is None:
        cx, cy, r = w // 2, h // 2, int(0.42 * min(h, w))
    else:
        cx, cy, r = res

    return clamp_circle(cx, cy, r, h, w)


def p50_L_in_disk(bgr, cx, cy, r, shrink_pct=DISK_SHRINK_PCT):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    h, w = L.shape
    r_stats = max(1, int(round(r * (1.0 - shrink_pct))))

    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r_stats * r_stats)

    vals = L[mask]

    if vals.size:
        return float(np.percentile(vals, 50))

    return float(np.percentile(L, 50))


def bucket_idx(value, thresholds):
    idx = 0
    for t in thresholds:
        if value < t:
            return idx
        idx += 1
    return idx


def decide_with_bridges(p50):
    idx = bucket_idx(p50, THRESH)
    base = LABELS[idx]

    if base == "Normal":
        if p50 < 105.0 + BRIDGE_TO_DARK:
            return "Dark", base, "bridge_to_dark"

        if p50 > 135.0 - BRIDGE_TO_BRIGHT:
            return "Bright", base, "bridge_to_bright"

    return base, base, ""


# ==================== COMMAND-LINE ARGUMENTS ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute retinal P50 brightness values and assign "
            "brightness categories to a CSV file."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Root directory containing the retinal images."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV containing the image identifiers."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the output CSV will be saved."
    )

    return parser.parse_args()

# ==================== MAIN ====================

def main():
    args = parse_args()

    csv_in = args.csv
    csv_out = args.output
    img_root = args.images

    if not csv_in.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_in}")

    if not img_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_root}")

    df = pd.read_csv(csv_in)

    image_col = IMAGE_COL if IMAGE_COL is not None else autodetect_image_col(df)

    print(f"Input CSV: {csv_in}")
    print(f"Image directory: {img_root}")
    print(f"Image column used: {image_col}")

    if "brightness_category" in df.columns and not OVERWRITE_EXISTING:
        raise ValueError(
            "The brightness_category column already exists. "
            "Set OVERWRITE_EXISTING=True to overwrite it."
        )

    if "p50" in df.columns and not OVERWRITE_EXISTING:
        raise ValueError(
            "The p50 column already exists. "
            "Set OVERWRITE_EXISTING=True to overwrite it."
        )

    # Create/reset output columns
    df["brightness_category"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string"
    )

    df["p50"] = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    print("Indexing images...")
    image_index, all_files = collect_image_index(img_root)

    if not all_files:
        raise RuntimeError(
    f"No images were found in the image directory: {img_root}"
    )

    print(f"Images found in directory: {len(all_files)}")

    n_ok = 0
    n_error = 0
    n_missing = 0
    n_ambiguous = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating brightness_category and p50"):
        value = row[image_col]

        fp, resolve_note = resolve_image_path(value, img_root, image_index)

        if resolve_note.startswith("ambiguous_match"):
            n_ambiguous += 1

        if fp is None:
            n_missing += 1
            n_error += 1
            continue

        try:
            bgr = imread_bgr(fp)

            if bgr is None:
                n_error += 1
                continue

            cx, cy, r = estimate_retina_circle(bgr)
            p50 = p50_L_in_disk(bgr, cx, cy, r)

            final_label, base_label, bridge_flag = decide_with_bridges(p50)

            df.at[idx, "brightness_category"] = final_label
            df.at[idx, "p50"] = round(p50, 1)

            n_ok += 1

        except Exception as exc:
            n_error += 1
            tqdm.write(f"Error processing {fp}: {exc}")
            continue

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, encoding="utf-8")

    print("\nProcessing completed.")
    print(f"CSV rows:                   {len(df)}")
    print(f"Successfully processed:     {n_ok}")
    print(f"Images not found:           {n_missing}")
    print(f"Read/calculation errors:    {n_error}")
    print(f"Ambiguous matches:          {n_ambiguous}") 
    print(f"CSV saved to:               {csv_out}")


if __name__ == "__main__":
    main()