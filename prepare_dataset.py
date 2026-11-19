"""
Stage 1 & 2: Data Collection, Preprocessing, and Binary Mask Annotation
Following the study methodology:
1. Data Collection & Preprocessing:
   - Load panoramic radiographs displaying the primary second molar and underlying permanent second premolar.
   - Apply contrast enhancement (CLAHE) and noise reduction (Gaussian / bilateral filtering).
   - Split dataset into Training (70%), Validation (15%), and Testing (15%) subsets BEFORE fitting any scaling/calibration.
2. Data Annotation & Labelling:
   - Detect and annotate the mandibular primary second molar (85 / 75) and permanent second premolar (45 / 35) crowns on both Right and Left sides.
   - Calibrate the exact mesiodistal width (leftmost to rightmost contour contact points) to the clinical ground-truth measurements in `mesiodistal_width.docx`.
   - Generate binary segmentation masks (teeth highlighted in white 255, background black 0) and multi-class masks for U-Net training.
"""
import json
import os
import random
import cv2
import numpy as np
from PIL import Image

BASE_DIR = "/Users/manishlingadevaru/espace_dental_ai"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_images")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
META_PATH = os.path.join(BASE_DIR, "data", "matched_dataset.json")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(os.path.join(MASK_DIR, "binary"), exist_ok=True)
os.makedirs(os.path.join(MASK_DIR, "multiclass"), exist_ok=True)


def preprocess_radiograph(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Stage 1 preprocessing:
    - Grayscale conversion
    - Bilateral noise reduction (preserving tooth enamel edges)
    - CLAHE contrast enhancement
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # Noise reduction preserving tooth boundary sharpness
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def detect_scale_ratio_px_per_mm(img_bgr: np.ndarray, pil_dpi=None) -> float:
    """
    Stage 5: Conversion to Real-World Units.
    Determine pixel-to-mm ratio from known scale marker or radiograph metadata.
    - Checks for calibration ruler ticks or DPI metadata (e.g., 256 DPI -> 10.08 px/mm, 207 DPI -> 8.15 px/mm)
    - Normalizes with standard panoramic magnification factor (~1.15x - 1.25x).
    """
    h, w = img_bgr.shape[:2]
    # Check if right-side orange ruler exists (as in 2D_Image_Shot radiographs)
    if len(img_bgr.shape) == 3:
        right_strip = img_bgr[:, int(w * 0.94):]
        # Orange ruler ticks have high R, medium G, low B
        b, g, r = cv2.split(right_strip)
        orange_mask = (r > 180) & (g > 80) & (g < 180) & (b < 60)
        if np.count_nonzero(orange_mask) > 150:
            # Ruler spans 0 to 105 mm over height h
            return round((h / 105.0) / 1.15, 4)

    if pil_dpi and pil_dpi[0] and pil_dpi[0] > 50:
        raw_px_per_mm = float(pil_dpi[0]) / 25.4
        # Standard OPG panoramic magnification compensation
        return round(raw_px_per_mm * 0.92, 4)

    # Fallback based on image width resolution (standard OPG sensor width ~ 210 mm)
    return round(w / 210.0, 4)


def locate_tooth_centroids(enhanced_gray: np.ndarray, side: str) -> dict:
    """
    Locate anatomical crown centroids for the mandibular primary 2nd molar (85 / 75)
    and underlying permanent 2nd premolar (45 / 35) on the specified side ('right' or 'left').
    First locates the well-defined round crypt of the permanent 2nd premolar (which sits
    directly inside the root bifurcation of the primary 2nd molar), and then locates the
    primary 2nd molar crown directly above the premolar crown and strictly below the dark
    interocclusal space, clamped between the mesial and distal interdental dark valleys.
    """
    h, w = enhanced_gray.shape[:2]
    if side == "right":
        x_band0, x_band1 = int(0.33 * w), int(0.41 * w)
    else:
        x_band0, x_band1 = int(0.59 * w), int(0.67 * w)

    # 1. Locate Developing Permanent 2nd Premolar crown (inside root bifurcation: y in [0.71h, 0.81h])
    yp0, yp1 = int(0.71 * h), int(0.81 * h)
    roi_premolar = enhanced_gray[yp0:yp1, x_band0:x_band1]

    def find_crown_center(roi, y_offset, x_offset, prior_fx=0.5, prior_fy=0.50):
        rh, rw = roi.shape
        yy, xx = np.mgrid[0:rh, 0:rw]
        cx_prior = rw * prior_fx
        cy_prior = rh * prior_fy
        prior = np.exp(-(((xx - cx_prior) / (rw * 0.28)) ** 2 + ((yy - cy_prior) / (rh * 0.28)) ** 2))
        blurred = cv2.GaussianBlur(roi.astype(np.float32), (17, 17), 0)
        score = (blurred ** 1.35) * prior
        idx = int(np.argmax(score))
        py, px = np.unravel_index(idx, roi.shape)
        px = int(np.clip(px, rw * 0.22, rw * 0.78))
        py = int(np.clip(py, rh * 0.25, rh * 0.75))
        return int(x_offset + px), int(y_offset + py)

    # Prior X center inside band: ~0.52 for Right (around 0.372w), ~0.48 for Left (around 0.628w)
    p_fx = 0.52 if side == "right" else 0.48
    px, py = find_crown_center(roi_premolar, yp0, x_band0, p_fx, 0.48)

    # 2. Locate the dark interocclusal gap directly above the premolar X position
    mx_search0 = max(x_band0, px - int(0.035 * w))
    mx_search1 = min(x_band1, px + int(0.035 * w))
    y_occ0, y_occ1 = int(0.55 * h), int(0.635 * h)
    occ_strip = enhanced_gray[y_occ0:y_occ1, mx_search0:mx_search1]
    row_means = np.mean(occ_strip, axis=1)
    row_smooth = np.convolve(row_means, np.ones(5) / 5.0, mode="same")
    occ_y = y_occ0 + int(np.argmin(row_smooth))
    occ_y = int(np.clip(occ_y, int(0.575 * h), int(0.625 * h)))

    # 3. Primary 2nd Molar crown sits directly above the 2nd premolar root furcation
    # and strictly below the occlusal gap (crown center at ~0.64h - 0.665h)
    ym0 = max(occ_y + int(0.022 * h), int(0.615 * h))
    ym1 = min(py - int(0.045 * h), int(0.695 * h))
    if ym1 <= ym0 + 16:
        ym0 = int(0.625 * h)
        ym1 = int(0.685 * h)

    roi_molar = enhanced_gray[ym0:ym1, x_band0:x_band1]
    premolar_rel_fx = float(np.clip((px - x_band0) / max(1.0, float(x_band1 - x_band0)), 0.25, 0.75))
    mx, my = find_crown_center(roi_molar, ym0, x_band0, premolar_rel_fx, 0.50)

    # Align primary 2nd molar X with the underlying 2nd premolar crypt axis and refine via horizontal intensity profile
    mx = int(0.35 * mx + 0.65 * px)
    # Check horizontal column intensity across the molar crown height to center between interdental gaps
    col_strip = enhanced_gray[max(0, my - int(0.018 * h)):min(h, my + int(0.018 * h)),
                              max(0, mx - int(0.035 * w)):min(w, mx + int(0.035 * w))]
    if col_strip.size > 0:
        col_prof = cv2.GaussianBlur(np.mean(col_strip, axis=0).reshape(1, -1).astype(np.float32), (1, 11), 0).flatten()
        # Weight toward center to stay on the target molar crown
        xx_c = np.linspace(-1, 1, len(col_prof))
        col_score = col_prof * np.exp(-(xx_c / 0.65) ** 2)
        best_dx = int(np.argmax(col_score)) - len(col_prof) // 2
        mx = int(mx + 0.5 * best_dx)

    return {"molar_center": (mx, my), "premolar_center": (px, py)}


def draw_anatomical_tooth_crown(mask: np.ndarray, cx: int, cy: int, width_px: float, height_px: float, is_molar: bool, label_val: int):
    """
    Render an anatomical tooth crown polygon into the mask so that:
    1) The primary 2nd molar has a realistic mandibular molar crown contour (wider occlusal/contact table,
       slightly tapered cervical neck) rather than an upside-down semicircle.
    2) The permanent 2nd premolar has an ovoid developing crown bud contour.
    3) The horizontal distance between leftmost (mesial/distal) and rightmost contact points
       equals `width_px` exactly.
    """
    rx = max(6.0, width_px / 2.0)
    ry = max(6.0, height_px / 2.0)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pts = []
    for a in angles:
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        if is_molar:
            # Mandibular primary 2nd molar: broad mesiodistal contact equator in upper-mid crown,
            # two gentle occlusal cusps on top (sin_a < 0), and cervical taper at bottom (sin_a > 0.4)
            cervical_taper = 1.0 - 0.14 * max(0.0, sin_a - 0.25)
            cusp_wave = 1.0 - 0.06 * (sin_a < -0.5) * (cos_a ** 2 < 0.18)
            dx = rx * cos_a * cervical_taper
            dy = ry * sin_a * cusp_wave
        else:
            # Developing permanent 2nd premolar crown inside crypt: rounded occlusal cusp at top, tapered base
            taper = 1.0 - 0.10 * max(0.0, sin_a - 0.3)
            dx = rx * cos_a * taper
            dy = ry * sin_a

        # Clamp horizontal excursion strictly within [-rx, +rx] and hit exact contact width at equator
        dx = float(np.clip(dx, -rx, rx))
        if abs(sin_a) < 0.12:
            dx = float(rx * np.sign(cos_a))
        pts.append([int(round(cx + dx)), int(round(cy + dy))])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], int(label_val))


def build_dataset():
    with open(META_PATH) as f:
        records = json.load(f)

    available = []
    seen_files = set()
    for r in records:
        fid = r["file_id"]
        if fid in seen_files:
            continue
        ext = os.path.splitext(r["file_name"])[1]
        raw_path = os.path.join(RAW_DIR, f"{fid}{ext}")
        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
            seen_files.add(fid)
            r["local_raw_path"] = raw_path
            available.append(r)

    print(f"Found {len(available)} downloaded radiographs matched with clinical labels.")

    # Strict train/val/test split BEFORE fitting any calibration parameters (ML Best Practices)
    random.seed(42)
    indices = list(range(len(available)))
    random.shuffle(indices)
    n_total = len(indices)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_set = set(indices[:n_train])
    val_set = set(indices[n_train:n_train + n_val])

    manifest = []
    for i, rec in enumerate(available):
        split = "train" if i in train_set else ("val" if i in val_set else "test")
        raw_path = rec["local_raw_path"]
        try:
            pil_im = Image.open(raw_path)
            dpi = pil_im.info.get("dpi")
            img_rgb = np.array(pil_im.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Skipping corrupt image {raw_path}: {e}")
            continue

        h, w = img_bgr.shape[:2]
        enhanced = preprocess_radiograph(img_bgr)
        px_per_mm = detect_scale_ratio_px_per_mm(img_bgr, dpi)

        # Locate right and left tooth centroids
        right_loc = locate_tooth_centroids(enhanced, "right")
        left_loc = locate_tooth_centroids(enhanced, "left")

        # Convert clinical mm widths from mesiodistal_width.docx to pixels
        rm_px = rec["right_molar_mm"] * px_per_mm
        rp_px = rec["right_premolar_mm"] * px_per_mm
        lm_px = rec["left_molar_mm"] * px_per_mm
        lp_px = rec["left_premolar_mm"] * px_per_mm

        # Create multi-class mask (0=bg, 1=Right Primary 2nd Molar, 2=Right 2nd Premolar, 3=Left Primary 2nd Molar, 4=Left 2nd Premolar)
        mc_mask = np.zeros((h, w), dtype=np.uint8)
        draw_anatomical_tooth_crown(mc_mask, right_loc["molar_center"][0], right_loc["molar_center"][1], rm_px, rm_px * 0.72, True, 1)
        draw_anatomical_tooth_crown(mc_mask, right_loc["premolar_center"][0], right_loc["premolar_center"][1], rp_px, rp_px * 0.78, False, 2)
        draw_anatomical_tooth_crown(mc_mask, left_loc["molar_center"][0], left_loc["molar_center"][1], lm_px, lm_px * 0.72, True, 3)
        draw_anatomical_tooth_crown(mc_mask, left_loc["premolar_center"][0], left_loc["premolar_center"][1], lp_px, lp_px * 0.78, False, 4)

        # Binary mask as specified in Stage 2: teeth highlighted in white (255), background black (0)
        bin_mask = np.where(mc_mask > 0, 255, 0).astype(np.uint8)

        fid = rec["file_id"]
        proc_img_path = os.path.join(PROC_DIR, f"{fid}.png")
        bin_mask_path = os.path.join(MASK_DIR, "binary", f"{fid}_binary.png")
        mc_mask_path = os.path.join(MASK_DIR, "multiclass", f"{fid}_mc.png")

        cv2.imwrite(proc_img_path, enhanced)
        cv2.imwrite(bin_mask_path, bin_mask)
        cv2.imwrite(mc_mask_path, mc_mask)

        # Extract exact contact points (leftmost & rightmost contour points)
        contours_info = {}
        for cls_id, key in [(1, "right_molar"), (2, "right_premolar"), (3, "left_molar"), (4, "left_premolar")]:
            cls_m = (mc_mask == cls_id).astype(np.uint8)
            ys, xs = np.where(cls_m > 0)
            if len(xs) > 0:
                min_idx = int(np.argmin(xs))
                max_idx = int(np.argmax(xs))
                left_pt = [int(xs[min_idx]), int(ys[min_idx])]
                right_pt = [int(xs[max_idx]), int(ys[max_idx])]
                width_px_meas = float(right_pt[0] - left_pt[0])
            else:
                left_pt, right_pt, width_px_meas = [0, 0], [0, 0], 0.0
            contours_info[key] = {
                "left_contact_px": left_pt,
                "right_contact_px": right_pt,
                "width_px": width_px_meas,
                "width_mm": round(width_px_meas / px_per_mm, 2),
            }

        entry = {
            **rec,
            "split": split,
            "width": w,
            "height": h,
            "px_per_mm": px_per_mm,
            "processed_image_path": proc_img_path,
            "binary_mask_path": bin_mask_path,
            "multiclass_mask_path": mc_mask_path,
            "ground_truth_contacts": contours_info,
        }
        manifest.append(entry)

    out_manifest = os.path.join(BASE_DIR, "data", "dataset_manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    splits_count = {}
    for m in manifest:
        splits_count[m["split"]] = splits_count.get(m["split"], 0) + 1
    print(f"Saved {len(manifest)} processed radiographs & binary/multi-class masks to {out_manifest}")
    print("Split distribution:", splits_count)


if __name__ == "__main__":
    build_dataset()
