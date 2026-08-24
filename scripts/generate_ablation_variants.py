"""
Generate hard-mask, fixed-feather, and adaptive-feather retinal image
variants used in the FOV-masking ablation study.

Brightness categories are read from a CSV file and mapped to
category-specific FOV thresholds and adaptive feather widths. The same
estimated FOV mask is shared across all three processed variants.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# General preprocessing configuration
PAD_PCT = 0.02
PAD_COLOR = (0, 0, 0)

# PNG compression affects file size and speed, not image quality.
PNG_COMPRESSION = 3

# Optional quality-assurance outputs
SAVE_OVERLAY = True
SAVE_MASK = True

# Contrast enhancement is applied only to the image used for FOV estimation.
GLOBAL_CONTRAST_PARAMS = dict(plow=1.0, phigh=98.0, sat_gain=1.02)

# Brightness categories are read from the CSV rather than recomputed here.
USE_BRIGHTNESS_CSV = True
CSV_FILENAME_COL = "filename"
CSV_BRIGHTNESS_COL = "brightness_category"

# If True, images missing from the CSV are skipped and reported.
# If False, the script falls back to an image-derived P50 estimate.
REQUIRE_CSV_MATCH = False

# Policy for CSV categories not defined in BRIGHTNESS_FEATHER_PROFILES.
# "fallback_compute": estimate P50 from the image.
# "default_profile": use DEFAULT_BRIGHTNESS_PROFILE.
UNKNOWN_CATEGORY_POLICY = "fallback_compute"

# Fallback P50-to-feather mapping used only when no valid profile is available.
P50_BIN_BASE = (132.0, 153.0)
P50_ANCHORS = np.array([132.0, 141.0, 153.0], np.float32)
FEATHER_224_AN = np.array([18.0, 19.0, 20.0], np.float32)
FEATHER_CLAMP = (18.0, 20.0)

DEFAULT_BRIGHTNESS_PROFILE = dict(
    p50_proxy=141.0,
    p50_bin_base=(132.0, 153.0),
    p50_anchors=np.array([132.0, 141.0, 153.0], np.float32),
    feather_224_an=np.array([20.0, 21.0, 22.0], np.float32),
    feather_clamp=(20.0, 22.0),
)

# Category-specific parameters used by the experimental pipeline.
# p50_proxy controls the adaptive feather width, while
# NONPURE_BLACK_THR controls FOV segmentation.
BRIGHTNESS_FEATHER_PROFILES = {
    "extremely_dark": dict(
        p50_proxy=35.0,
        p50_bin_base=(0.0, 70.0),
        p50_anchors=np.array([0.0, 35.0, 70.0], np.float32),
        feather_224_an=np.array([14.0, 15.0, 16.0], np.float32),
        feather_clamp=(14.0, 16.0),
        NONPURE_BLACK_THR = 20,
    ),
    "very_dark": dict(
        p50_proxy=77.5,
        p50_bin_base=(70.0, 85.0),
        p50_anchors=np.array([70.0, 77.5, 85.0], np.float32),
        feather_224_an=np.array([16.0, 17.0, 18.0], np.float32),
        feather_clamp=(16.0, 18.0),
        NONPURE_BLACK_THR = 40,
    ),
    "dark": dict(
        p50_proxy=95.0,
        p50_bin_base=(85.0, 105.0),
        p50_anchors=np.array([85.0, 95.0, 105.0], np.float32),
        feather_224_an=np.array([18.0, 18.0, 18.0], np.float32),
        feather_clamp=(18.0, 18.0),
        NONPURE_BLACK_THR = 50,
    ),
    "normal": dict(
        p50_proxy=117.0,
        p50_bin_base=(105.0, 129.0),
        p50_anchors=np.array([105.0, 117.0, 129.0], np.float32),
        feather_224_an=np.array([18.0, 18.0, 18.0], np.float32),
        feather_clamp=(18.0, 18.0),
        NONPURE_BLACK_THR = 60,
    ),
    "bright": dict(
        p50_proxy=147.0,
        p50_bin_base=(129.0, 165.0),
        p50_anchors=np.array([129.0, 147.0, 165.0], np.float32),
        feather_224_an=np.array([20.0, 20.0, 20.0], np.float32),
        feather_clamp=(20.0, 20.0),
        NONPURE_BLACK_THR = 70,
    ),
    "very_bright": dict(
        p50_proxy=210.0,
        p50_bin_base=(165.0, 255.0),
        p50_anchors=np.array([165.0, 210.0, 255.0], np.float32),
        feather_224_an=np.array([22.0, 21.0, 20.0], np.float32),
        feather_clamp=(20.0, 22.0),
        NONPURE_BLACK_THR = 80,
    ),
}
CATEGORY_ALIASES = {
    # CamelCase labels produced by calculate_p50.py
    "extremelydark": "extremely_dark",
    "verydark": "very_dark",
    "verybright": "very_bright",

    # English variants
    "extremely dark": "extremely_dark",
    "extremely_dark": "extremely_dark",
    "extremely-dark": "extremely_dark",

    "very dark": "very_dark",
    "very_dark": "very_dark",
    "very-dark": "very_dark",

    "dark": "dark",
    "normal": "normal",
    "bright": "bright",

    "very bright": "very_bright",
    "very_bright": "very_bright",
    "very-bright": "very_bright",

    # Spanish variants
    "extremadamente oscuro": "extremely_dark",
    "extremadamente oscura": "extremely_dark",
    "muy oscuro": "very_dark",
    "muy oscura": "very_dark",
    "oscuro": "dark",
    "oscura": "dark",
    "brillante": "bright",
    "muy brillante": "very_bright",
}

# Store outputs in brightness-category subdirectories.
SAVE_BY_BRIGHTNESS_CATEGORY = True

# Folder names for normalized brightness categories.
FOLDER_NAME_BY_KEY = {
    "extremely_dark": "extremely_dark",
    "very_dark": "very_dark",
    "dark": "dark",
    "normal": "normal",
    "bright": "bright",
    "very_bright": "very_bright",
}

ASYM_FEATHER_SIDE = 'NONE'
ASYM_EXTRA_AT_224 = 0.0
ASYM_EXTRAS_BY_SIDE = None

DEFAULT_NONPURE_BLACK_THR = 25

# Output variants
SAVE_HARD_MASK_BLACK = True
SAVE_FIXED_FEATHER = True

# Fixed feather width is defined at a 224-pixel reference scale and
# rescaled to the full-resolution image. Set FIXED_FEATHER_FULL_PX to
# override this behavior with a fixed full-resolution width.
FIXED_FEATHER_AT_224 = 15
FIXED_FEATHER_FULL_PX = None

SAVE_ADAPTIVE_FEATHER = True

# File and image utilities
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def collect_imgs(root):
    valid_exts = {
        ".jpg", ".jpeg", ".png",
        ".bmp", ".tif", ".tiff"
    }

    return sorted(
        str(p)
        for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    )

def imread_rgb(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_png(path_out, img_rgb, compression=PNG_COMPRESSION):
    """Save an RGB image as lossless PNG."""
    ensure_dir(Path(path_out).parent)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(
        str(path_out),
        bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]
    )
    if not ok:
        raise IOError(f"No se pudo guardar la imagen PNG: {path_out}")

def save_mask_png(path_out, mask_u8, compression=PNG_COMPRESSION):
    """Save a binary mask as lossless uint8 PNG."""
    ensure_dir(Path(path_out).parent)
    mask_u8 = mask_u8.astype(np.uint8)
    ok = cv2.imwrite(
        str(path_out),
        mask_u8,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]
    )
    if not ok:
        raise IOError(f"No se pudo guardar la máscara PNG: {path_out}")

def add_padding(img_rgb, pad_pct=0.08, color=(0,0,0)):
    h,w = img_rgb.shape[:2]
    pad = int(round(pad_pct*max(h,w)))
    if pad<=0: return img_rgb,0
    canvas = np.full((h+2*pad, w+2*pad, 3), color, dtype=np.uint8)
    canvas[pad:pad+h, pad:pad+w] = img_rgb
    return canvas, pad

def get_LV(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    V = hsv[:,:,2].astype(np.float32)
    L = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[:,:,0].astype(np.float32)
    return L, V

# Contour utilities
def extract_contour(mask_u8):
    cnts,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea).reshape(-1,2).astype(np.int32)

def smooth_poly(pts, eps_ratio=0.003):
    if pts is None or len(pts)<5: return pts
    peri = cv2.arcLength(pts.reshape(-1,1,2), True)
    eps = float(eps_ratio) * max(peri,1.0)
    approx = cv2.approxPolyDP(pts.reshape(-1,1,2), eps, True)
    return approx.reshape(-1,2)

def draw_outline(img_rgb, pts, color=(255,0,0), thickness=3):
    if pts is None: return img_rgb.copy()
    out = img_rgb.copy()
    cv2.polylines(out, [pts.reshape(-1,1,2)], True, color, thickness, lineType=cv2.LINE_AA)
    return out

# Inscribed-circle and P50 utilities
def circle_from_mask_inscribed(mask_u8):
    inside = (mask_u8>0).astype(np.uint8)
    if inside.max()==0: return None
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
    cx, cy = int(maxLoc[0]), int(maxLoc[1])
    r = max(1, int(round(maxVal - 2)))
    h,w = mask_u8.shape[:2]
    r = int(np.clip(r, 1, int(0.56*min(h,w))))
    cx = int(np.clip(cx, r, w-r-1)); cy = int(np.clip(cy, r, h-r-1))
    return cx,cy,r

def p50_in_disk_L(L, cx, cy, r, shrink_pct=0.08):
    h,w = L.shape[:2]
    r_stats = max(1, int(round(r*(1.0-shrink_pct))))
    x1=max(0,cx-r_stats); y1=max(0,cy-r_stats)
    x2=min(w,cx+r_stats); y2=min(h,cy+r_stats)
    Lroi = L[y1:y2, x1:x2]
    if Lroi.size==0: return float(np.percentile(L,50))
    yy,xx = np.ogrid[:Lroi.shape[0], :Lroi.shape[1]]
    mask=(xx-(cx-x1))**2 + (yy-(cy-y1))**2 <= (r_stats*r_stats)
    vals=Lroi[mask]
    if vals.size==0: return float(np.percentile(L,50))
    return float(np.percentile(vals,50))

def p50_bin_edges(p50, step=3.0):
    start = step*np.floor(p50/step)
    return float(start), float(start+step)

def feather_at_224_from_p50(p50):
    if P50_BIN_BASE[0] <= p50 < P50_BIN_BASE[1]:
        f = float(np.interp(p50, P50_ANCHORS, FEATHER_224_AN))
        return int(round(np.clip(f, FEATHER_CLAMP[0], FEATHER_CLAMP[1])))
    return int(round(FEATHER_CLAMP[0] if p50 < P50_BIN_BASE[0] else FEATHER_CLAMP[1]))

def _norm_key(x):
    """Normalize filenames and paths for CSV-to-image matching."""
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\\", "/")
    return s.lower()

def _norm_stem(x):
    """Normalize a path without its extension for robust matching."""
    s = _norm_key(x)
    if not s:
        return ""
    return str(Path(s).with_suffix(""))

def normalize_brightness_category(x):
    """Normalize a brightness-category label to a profile key."""
    if pd.isna(x):
        return None
    raw = str(x).strip()
    if not raw:
        return None
    s = raw.lower().replace("-", "_").replace("/", "_")
    s = "_".join(s.split())
    s_space = s.replace("_", " ")
    return CATEGORY_ALIASES.get(s, CATEGORY_ALIASES.get(s_space, s))

def get_profile_from_category(category):
    key = normalize_brightness_category(category)
    if key is None:
        return None, None
    profile = BRIGHTNESS_FEATHER_PROFILES.get(key)
    return profile, key

def feather_at_224_from_profile(profile):
    """Return the 224-pixel-reference feather width for a profile."""
    if profile is None:
        profile = DEFAULT_BRIGHTNESS_PROFILE
    p50_proxy = float(profile.get("p50_proxy", DEFAULT_BRIGHTNESS_PROFILE["p50_proxy"]))
    anchors = np.asarray(profile.get("p50_anchors", DEFAULT_BRIGHTNESS_PROFILE["p50_anchors"]), dtype=np.float32)
    values = np.asarray(profile.get("feather_224_an", DEFAULT_BRIGHTNESS_PROFILE["feather_224_an"]), dtype=np.float32)
    clamp = profile.get("feather_clamp", DEFAULT_BRIGHTNESS_PROFILE["feather_clamp"])
    f = float(np.interp(p50_proxy, anchors, values))
    return int(round(np.clip(f, float(clamp[0]), float(clamp[1]))))

def profile_bin_edges(profile):
    if profile is None:
        profile = DEFAULT_BRIGHTNESS_PROFILE
    b0, b1 = profile.get("p50_bin_base", DEFAULT_BRIGHTNESS_PROFILE["p50_bin_base"])
    return float(b0), float(b1)

def load_brightness_lookup(csv_path):
    """
    Build a brightness-category lookup from the labels CSV.

    Matching supports relative paths, basenames, and extension-free stems.
    """
    csv_path = str(csv_path).strip()
    if not csv_path:
        return {}, dict(enabled=False, note="LABELS_CSV vacío; se usará fallback por P50 calculado.")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        return {}, dict(enabled=False, note=f"No existe LABELS_CSV: {csv_file}; se usará fallback por P50 calculado.")

    df = pd.read_csv(csv_file)
    if CSV_FILENAME_COL not in df.columns:
        raise ValueError(f"El CSV no tiene la columna '{CSV_FILENAME_COL}'. Columnas disponibles: {list(df.columns)}")
    if CSV_BRIGHTNESS_COL not in df.columns:
        raise ValueError(f"El CSV no tiene la columna '{CSV_BRIGHTNESS_COL}'. Columnas disponibles: {list(df.columns)}")

    lookup = {}
    for _, row in df.iterrows():
        fname = row[CSV_FILENAME_COL]
        cat = row[CSV_BRIGHTNESS_COL]
        if pd.isna(fname) or pd.isna(cat):
            continue

        key_full = _norm_key(fname)
        key_base = _norm_key(Path(str(fname)).name)
        key_stem_full = _norm_stem(fname)
        key_stem_base = _norm_stem(Path(str(fname)).name)

        for k in {key_full, key_base, key_stem_full, key_stem_base}:
            if k:
                lookup[k] = str(cat).strip()

    return lookup, dict(enabled=True, note=f"CSV cargado: {csv_file} | filas={len(df)} | claves_lookup={len(lookup)}")

def find_brightness_category_for_image(path_obj, rel, lookup):
    """Find the brightness category associated with an input image."""
    if not lookup:
        return None
    candidates = [
        _norm_key(rel),
        _norm_key(Path(rel).name),
        _norm_stem(rel),
        _norm_stem(Path(rel).name),
        _norm_key(path_obj.name),
        _norm_stem(path_obj.name),
    ]
    for k in candidates:
        if k in lookup:
            return lookup[k]
    return None

def sanitize_folder_name(name):
    """Convert a category label to a filesystem-safe folder name."""
    if name is None or pd.isna(name):
        return "unknown"
    s = str(name).strip().lower()
    if not s:
        return "unknown"

    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
        "Á": "a", "É": "e", "Í": "i", "Ó": "o", "Ú": "u", "Ü": "u", "Ñ": "n",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)

    for ch in [" ", "-", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        s = s.replace(ch, "_")

    # Collapse repeated underscores.
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("._")
    return s if s else "unknown"

def get_category_folder(brightness_category_key=None, brightness_category_raw=None):
    """Return the output folder associated with a brightness category."""
    if not SAVE_BY_BRIGHTNESS_CATEGORY:
        return None

    if brightness_category_key:
        key = str(brightness_category_key).strip().lower()
        if key in FOLDER_NAME_BY_KEY:
            return FOLDER_NAME_BY_KEY[key]
        return sanitize_folder_name(key)

    if brightness_category_raw:
        normalized = normalize_brightness_category(brightness_category_raw)
        if normalized in FOLDER_NAME_BY_KEY:
            return FOLDER_NAME_BY_KEY[normalized]
        return sanitize_folder_name(brightness_category_raw)

    return "unknown"

def out_path_by_category(base_dir, category_folder, rel, suffix=".png"):
    """Build an output path, optionally grouped by brightness category."""
    rel_path = Path(rel).with_suffix(suffix)
    if SAVE_BY_BRIGHTNESS_CATEGORY and category_folder:
        return base_dir / category_folder / rel_path
    return base_dir / rel_path

def rel_path_by_category(base_name, category_folder, rel, suffix=".png"):
    """Build the relative output path stored in the metadata CSV."""
    rel_path = Path(rel).with_suffix(suffix)
    if SAVE_BY_BRIGHTNESS_CATEGORY and category_folder:
        return str(Path(base_name) / category_folder / rel_path)
    return str(Path(base_name) / rel_path)

# Alpha masks and black-background compositing
def alpha_from_mask(mask_u8, feather_px):
    h,w = mask_u8.shape[:2]
    if feather_px<=0: return (mask_u8>0).astype(np.float32)
    inside = (mask_u8>0).astype(np.uint8)
    ys,xs = np.where(inside)
    if xs.size==0: return np.zeros((h,w), np.float32)
    margin = int(feather_px*2+2)
    x1=max(0,xs.min()-margin); x2=min(w,xs.max()+margin+1)
    y1=max(0,ys.min()-margin); y2=min(h,ys.max()+margin+1)
    dist_roi = cv2.distanceTransform(inside[y1:y2, x1:x2], cv2.DIST_L2, 5)
    t = np.clip(dist_roi/float(feather_px), 0.0, 1.0)
    a = 0.5*(1.0 - np.cos(np.pi*t)).astype(np.float32)
    a = cv2.GaussianBlur(a, (0,0), 0.7)
    alpha = np.zeros((h,w), np.float32)
    alpha[y1:y2, x1:x2] = a
    return alpha

def _srgb_to_linear(u8):
    x = (u8.astype(np.float32)/255.0)
    return np.where(x<=0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

def _linear_to_srgb(lin):
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x<0.0031308, 12.92*x, 1.055*np.power(x,1/2.4)-0.055)
    return np.clip(s*255.0, 0, 255).astype(np.uint8)

def composite_black(img_rgb, alpha):
    lin = _srgb_to_linear(img_rgb)
    out_lin = lin*alpha[...,None]
    return _linear_to_srgb(out_lin)

# Optional asymmetric feathering
def _make_side_extra_map(h, w, side, extra_px):
    if extra_px <= 0 or side == 'NONE':
        return np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    E = np.zeros((h, w), np.float32)
    pi = np.pi
    def ramp01(t):
        t = np.clip(t, 0.0, 1.0)
        return 0.5 * (1.0 - np.cos(pi * t))
    if side == 'LEFT':
        t = 1.0 - (xx / max(1.0, extra_px))
        E = extra_px * ramp01(t)
    elif side == 'RIGHT':
        t = (xx - (w - extra_px)) / max(1.0, extra_px)
        E = extra_px * ramp01(t)
    elif side == 'TOP':
        t = 1.0 - (yy / max(1.0, extra_px))
        E = extra_px * ramp01(t)
    elif side == 'BOTTOM':
        t = (yy - (h - extra_px)) / max(1.0, extra_px)
        E = extra_px * ramp01(t)
    return E.astype(np.float32)

def asymmetric_alpha_from_mask(mask_u8, feather_px_base,
                               side='NONE', extra_px=0.0,
                               extras_by_side=None):
    h, w = mask_u8.shape[:2]
    inside = (mask_u8 > 0).astype(np.uint8)
    if inside.max() == 0 or feather_px_base <= 0:
        return inside.astype(np.float32)

    ys, xs = np.where(inside)
    margin = int(feather_px_base * 2 + 2)
    x1 = max(0, xs.min() - margin); x2 = min(w, xs.max() + margin + 1)
    y1 = max(0, ys.min() - margin); y2 = min(h, ys.max() + margin + 1)

    inside_roi = inside[y1:y2, x1:x2]
    dist_roi = cv2.distanceTransform(inside_roi, cv2.DIST_L2, 5).astype(np.float32)

    E_full = np.zeros((h, w), np.float32)
    if extras_by_side is not None:
        for s, ex in extras_by_side.items():
            if ex and ex > 0:
                E_full += _make_side_extra_map(h, w, s, float(ex))
    else:
        if extra_px and extra_px > 0:
            E_full += _make_side_extra_map(h, w, side, float(extra_px))

    E_roi = E_full[y1:y2, x1:x2]
    denom = np.maximum(1e-6, feather_px_base + E_roi)
    t = np.clip(dist_roi / denom, 0.0, 1.0)
    a = 0.5 * (1.0 - np.cos(np.pi * t))
    a = cv2.GaussianBlur(a, (0, 0), 0.7).astype(np.float32)

    alpha = np.zeros((h, w), np.float32)
    alpha[y1:y2, x1:x2] = a
    return alpha

# FOV segmentation from non-black pixels
def mask_from_nonpure_black(img_rgb, thr=6, min_hole_pct=0.001):
    """
    Estimate the retinal FOV from pixels whose maximum RGB value exceeds
    the threshold, retain the largest connected component, and fill
    sufficiently large internal holes.
    """
    h, w = img_rgb.shape[:2]

    # Threshold non-black pixels.
    m = (img_rgb.max(axis=2) > int(thr)).astype(np.uint8) * 255

    # Join small neighboring regions.
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Keep the largest connected component as the retinal FOV.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (m > 0).astype(np.uint8), connectivity=8
    )
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = 1 + np.argmax(areas)
        m = (labels == idx).astype(np.uint8) * 255

    # Fill internal holes in the retained component.
    inv = cv2.bitwise_not(m)
    flood = inv.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)

    # Keep only holes above the minimum relative area.
    area_min = int(max(1, min_hole_pct * h * w))
    num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(
        (holes > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(holes)
    for i in range(1, num_h):
        if stats_h[i, cv2.CC_STAT_AREA] >= area_min:
            keep[labels_h == i] = 255

    m_filled = cv2.bitwise_or(m, keep)

    return m_filled

def build_fov_mask(
    img_rgb,
    nonpure_black_thr=DEFAULT_NONPURE_BLACK_THR
):
    mask = mask_from_nonpure_black(
        img_rgb,
        nonpure_black_thr
    )

    return mask, {
        "mode": "nonpure_black",
        "nonpure_black_thr": int(nonpure_black_thr)
    }

# Contrast enhancement used only for FOV estimation
def global_contrast_for_mask(img_rgb, plow=2.0, phigh=98.0, sat_gain=1.05):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    lo = np.percentile(L, plow)
    hi = np.percentile(L, phigh)
    if hi <= lo:
        lo, hi = 5, 245
    Lf = np.clip((L.astype(np.float32)-lo) * (255.0/max(1.0, hi-lo)), 0, 255).astype(np.uint8)
    rgb_l = cv2.cvtColor(cv2.merge([Lf, A, B]), cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv)
    S = np.clip(S.astype(np.float32)*sat_gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([H,S,V]), cv2.COLOR_HSV2RGB)

# Command-line interface

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate hard-mask, fixed-feather, and adaptive-feather "
            "retinal image variants for the ablation study."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Root directory containing the original retinal images."
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=(
            "CSV file containing image identifiers and "
            "brightness_category values."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root directory where the processed datasets will be saved."
    )

    return parser.parse_args()

# Main processing pipeline

def main():
    args = parse_args()

    source_dir = args.images
    labels_csv = args.labels
    out_dir = args.output

    # Validate inputs.
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {source_dir}"
        )

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels CSV does not exist: {labels_csv}"
        )

    # Prepare output directories.
    d_hard = out_dir / "images_hard_mask_black"
    d_fixed = out_dir / "images_fixed_feather"
    d_adaptive = out_dir / "images_adaptive_feather"

    # Optional QA directories.
    d_overlay = out_dir / "qa_outline_hr"
    d_mask = out_dir / "qa_mask_hr"

    dirs_to_create = []

    if SAVE_HARD_MASK_BLACK:
        dirs_to_create.append(d_hard)

    if SAVE_FIXED_FEATHER:
        dirs_to_create.append(d_fixed)

    if SAVE_ADAPTIVE_FEATHER:
        dirs_to_create.append(d_adaptive)

    if SAVE_OVERLAY:
        dirs_to_create.append(d_overlay)

    if SAVE_MASK:
        dirs_to_create.append(d_mask)

    for directory in dirs_to_create:
        ensure_dir(directory)

    rows = []

    # Load brightness-category lookup.
    brightness_lookup = {}
    brightness_lookup_info = dict(
        enabled=False,
        note="USE_BRIGHTNESS_CSV=False"
    )

    if USE_BRIGHTNESS_CSV:
        brightness_lookup, brightness_lookup_info = (
            load_brightness_lookup(labels_csv)
        )
        print(brightness_lookup_info["note"])

    # Discover input images.
    image_files = collect_imgs(source_dir)

    if not image_files:
        raise RuntimeError(
            f"No images were found in the image directory: {source_dir}"
        )

    print(f"Images found: {len(image_files)}")

    # Process each image independently.
    for p in tqdm(
        image_files,
        desc="Hard mask + fixed feather + adaptive feather"
    ):
        p = Path(p)
        rel = str(p.relative_to(source_dir))

        try:
            img = imread_rgb(str(p))

            if img is None:
                rows.append(
                    dict(
                        rel_in=rel,
                        status="read_error"
                    )
                )
                continue

            # Preserve the padding used in the experimental pipeline.
            img, _ = add_padding(
                img,
                pad_pct=PAD_PCT,
                color=PAD_COLOR
            )

            h, w = img.shape[:2]

            # Initialize brightness-profile state.
            brightness_category_raw = None
            brightness_category_key = None
            brightness_profile = None

            p50 = None
            p50_est = None
            p50_source = "computed_fallback"
            L_est = None

            nonpure_black_thr = DEFAULT_NONPURE_BLACK_THR

            # Match the image to its CSV brightness category.
            if USE_BRIGHTNESS_CSV and brightness_lookup:

                brightness_category_raw = (
                    find_brightness_category_for_image(
                        p,
                        rel,
                        brightness_lookup
                    )
                )

                if brightness_category_raw is None:

                    if REQUIRE_CSV_MATCH:
                        rows.append(
                            dict(
                                rel_in=rel,
                                status="csv_missing_brightness_category",
                                note=(
                                    "The image was not found in the labels "
                                    "CSV using filename/path/stem matching."
                                )
                            )
                        )
                        continue

                else:
                    (
                        brightness_profile,
                        brightness_category_key
                    ) = get_profile_from_category(
                        brightness_category_raw
                    )

                    if brightness_profile is None:

                        if UNKNOWN_CATEGORY_POLICY == "default_profile":
                            brightness_profile = (
                                DEFAULT_BRIGHTNESS_PROFILE
                            )

                            p50_source = (
                                "csv_unknown_category_default_profile"
                            )

                        else:
                            p50_source = (
                                "computed_fallback_unknown_category"
                            )

                    else:
                        p50_source = "csv_brightness_category"

            # Select the output category folder.
            category_folder = get_category_folder(
                brightness_category_key=brightness_category_key,
                brightness_category_raw=brightness_category_raw
            )

            # Enhance contrast only for FOV estimation.
            img_boost = global_contrast_for_mask(
                img,
                **GLOBAL_CONTRAST_PARAMS
            )

            # Build one shared FOV mask for all processed variants.
            if brightness_profile is not None:

                p50_est = float(
                    brightness_profile.get(
                        "p50_proxy",
                        DEFAULT_BRIGHTNESS_PROFILE["p50_proxy"]
                    )
                )

                # Category-specific FOV threshold.
                nonpure_black_thr = int(
                    brightness_profile.get(
                        "NONPURE_BLACK_THR",
                        DEFAULT_NONPURE_BLACK_THR
                    )
                )

                mask, qa_stats = build_fov_mask(
                    img_boost,
                    nonpure_black_thr
                )

            else:
                # Fallback when no valid brightness profile exists
                L_est, _ = get_LV(img_boost)

                cx0 = w // 2
                cy0 = h // 2
                r0 = int(0.42 * min(h, w))

                p50_est = p50_in_disk_L(
                    L_est,
                    cx0,
                    cy0,
                    r0,
                    shrink_pct=0.10
                )

                nonpure_black_thr = DEFAULT_NONPURE_BLACK_THR

                mask, qa_stats = build_fov_mask(
                    img_boost,
                    nonpure_black_thr
                )

            # Seal small mask discontinuities.
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (5, 5)
                )
            )

            # Derive contour and inscribed-circle metadata.
            cnt = extract_contour(mask)
            cnt_s = smooth_poly(cnt, 0.003)

            circ = circle_from_mask_inscribed(mask)

            if circ is None:
                cx = w // 2
                cy = h // 2
                r = int(0.42 * min(h, w))
            else:
                cx, cy, r = circ

            # Determine the adaptive feather width.
            if brightness_profile is not None:

                p50 = float(
                    brightness_profile.get(
                        "p50_proxy",
                        DEFAULT_BRIGHTNESS_PROFILE["p50_proxy"]
                    )
                )

                f224 = feather_at_224_from_profile(
                    brightness_profile
                )

                b0, b1 = profile_bin_edges(
                    brightness_profile
                )

            else:
                if L_est is None:
                    L_est, _ = get_LV(img_boost)

                p50 = p50_in_disk_L(
                    L_est,
                    cx,
                    cy,
                    r,
                    shrink_pct=0.08
                )

                f224 = feather_at_224_from_p50(p50)

                b0, b1 = p50_bin_edges(
                    p50,
                    3.0
                )

            # Scale feather widths from the 224-pixel reference
            # to the original image resolution.
            scale_full = max(h, w) / 224.0

            # Variant 1: hard mask.
            out_hard_rel = None

            if SAVE_HARD_MASK_BLACK:

                alpha_hard = (
                    mask > 0
                ).astype(np.float32)

                composed_hard = composite_black(
                    img,
                    alpha_hard
                )

                out_hard = out_path_by_category(
                    d_hard,
                    category_folder,
                    rel,
                    suffix=".png"
                )

                save_png(
                    out_hard,
                    composed_hard
                )

                out_hard_rel = rel_path_by_category(
                    "images_hard_mask_black",
                    category_folder,
                    rel,
                    suffix=".png"
                )

            # Variant 2: fixed feather.
            out_fixed_rel = None
            fixed_feather_full = None

            if SAVE_FIXED_FEATHER:

                if FIXED_FEATHER_FULL_PX is None:

                    fixed_feather_full = max(
                        1,
                        int(
                            round(
                                float(FIXED_FEATHER_AT_224)
                                * scale_full
                            )
                        )
                    )

                else:
                    fixed_feather_full = max(
                        1,
                        int(
                            round(
                                float(FIXED_FEATHER_FULL_PX)
                            )
                        )
                    )

                # Fixed feather is identical across brightness categories.
                alpha_fixed = alpha_from_mask(
                    mask,
                    fixed_feather_full
                )

                composed_fixed = composite_black(
                    img,
                    alpha_fixed
                )

                out_fixed = out_path_by_category(
                    d_fixed,
                    category_folder,
                    rel,
                    suffix=".png"
                )

                save_png(
                    out_fixed,
                    composed_fixed
                )

                out_fixed_rel = rel_path_by_category(
                    "images_fixed_feather",
                    category_folder,
                    rel,
                    suffix=".png"
                )

            # Variant 3: adaptive feather.
            out_adaptive_rel = None
            f_full = None

            if SAVE_ADAPTIVE_FEATHER:

                f_full = max(
                    2,
                    int(round(f224 * scale_full))
                )

                if ASYM_EXTRAS_BY_SIDE is not None:

                    extras_full = {
                        key: float(value) * scale_full
                        for key, value
                        in ASYM_EXTRAS_BY_SIDE.items()
                    }

                    alpha_adaptive = (
                        asymmetric_alpha_from_mask(
                            mask,
                            f_full,
                            extras_by_side=extras_full
                        )
                    )

                else:

                    extra_full = (
                        float(ASYM_EXTRA_AT_224)
                        * scale_full
                    )

                    alpha_adaptive = (
                        asymmetric_alpha_from_mask(
                            mask,
                            f_full,
                            side=ASYM_FEATHER_SIDE,
                            extra_px=extra_full
                        )
                    )

                composed_adaptive = composite_black(
                    img,
                    alpha_adaptive
                )

                out_adaptive = out_path_by_category(
                    d_adaptive,
                    category_folder,
                    rel,
                    suffix=".png"
                )

                save_png(
                    out_adaptive,
                    composed_adaptive
                )

                out_adaptive_rel = rel_path_by_category(
                    "images_adaptive_feather",
                    category_folder,
                    rel,
                    suffix=".png"
                )

            # Save optional QA outputs.
            if SAVE_OVERLAY:

                overlay = draw_outline(
                    img,
                    cnt_s,
                    color=(255, 0, 0),
                    thickness=3
                )

                save_png(
                    out_path_by_category(
                        d_overlay,
                        category_folder,
                        rel,
                        suffix=".png"
                    ),
                    overlay
                )

            if SAVE_MASK:

                save_mask_png(
                    out_path_by_category(
                        d_mask,
                        category_folder,
                        rel,
                        suffix=".png"
                    ),
                    mask
                )

            # Record per-image processing metadata.
            rows.append(
                dict(
                    rel_in=str(rel),

                    out_hard_mask_black=out_hard_rel,
                    out_fixed_feather=out_fixed_rel,
                    out_adaptive_feather=out_adaptive_rel,

                    w=w,
                    h=h,

                    cx=int(cx),
                    cy=int(cy),
                    r=int(r),

                    brightness_category_raw=brightness_category_raw,
                    brightness_category_key=brightness_category_key,
                    brightness_output_folder=category_folder,

                    p50_source=p50_source,

                    p50_est_or_proxy=(
                        round(float(p50_est), 1)
                        if p50_est is not None
                        else None
                    ),

                    p50_bin_start=int(round(b0)),
                    p50_bin_end=int(round(b1)),

                    nonpure_black_thr=int(
                        nonpure_black_thr
                    ),

                    fixed_feather_at_224=(
                        None
                        if FIXED_FEATHER_FULL_PX is not None
                        else int(FIXED_FEATHER_AT_224)
                    ),

                    fixed_feather_full=(
                        int(fixed_feather_full)
                        if fixed_feather_full is not None
                        else None
                    ),

                    adaptive_feather_at_224=int(f224),

                    adaptive_feather_full=(
                        int(f_full)
                        if f_full is not None
                        else None
                    ),

                    mode=qa_stats.get("mode"),

                    status="ok"
                )
            )

        except Exception as exc:

            rows.append(
                dict(
                    rel_in=rel,
                    status="error",
                    note=str(exc)
                )
            )

            tqdm.write(
                f"Error processing {rel}: {exc}"
            )

    # Save processing metadata.
    df = pd.DataFrame(rows)

    ensure_dir(out_dir)

    metadata_path = (
        out_dir
        / "_meta_delineado_tres_salidas.csv"
    )

    df.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8"
    )

    # Print a concise processing summary.
    n_ok = int(
        (df["status"] == "ok").sum()
    ) if "status" in df.columns else 0

    n_errors = len(df) - n_ok

    print("\nProcessing completed.")
    print(f"Images processed successfully: {n_ok}")
    print(f"Images with errors:            {n_errors}")

    if SAVE_HARD_MASK_BLACK:
        print(f"Hard-mask images:              {d_hard}")

    if SAVE_FIXED_FEATHER:
        print(f"Fixed-feather images:          {d_fixed}")

    if SAVE_ADAPTIVE_FEATHER:
        print(f"Adaptive-feather images:       {d_adaptive}")

    if SAVE_OVERLAY:
        print(f"QA overlays:                   {d_overlay}")

    if SAVE_MASK:
        print(f"QA masks:                      {d_mask}")

    print(f"Metadata CSV:                  {metadata_path}")

if __name__ == "__main__":
    cv2.setUseOptimized(True)
    main()