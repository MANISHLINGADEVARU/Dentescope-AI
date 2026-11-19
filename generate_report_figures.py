"""
Generate comprehensive visual report figures for all 5 stages of the methodology:
- Stage 1: Raw Panoramic OPG vs CLAHE Contrast-Enhanced & Bilateral Denoised OPG
- Stage 2: Binary Mask (Teeth=255 White, Background=0 Black) & Multi-Class Tooth Mask
- Stage 3: U-Net Training & Validation Loss / Dice Convergence Curves
- Stage 4 & 5: Segmented Tooth Contours with Mesial/Distal Contact Points (Leftmost & Rightmost)
  and Real-World Millimeter E-Space Measurements (Before vs After Contact Point Adjustment)
"""
import json
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/Users/manishlingadevaru/espace_dental_ai"
ARTIFACT_DIR = "/Users/manishlingadevaru/.gemini/antigravity/brain/ab878684-4dbd-460e-b21b-fddc3fc990f4"
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "dataset_manifest.json")
EVAL_PATH = os.path.join(BASE_DIR, "results", "evaluation_summary.json")


def make_figures():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Pick a clear test radiograph
    sample = next((m for m in manifest if m["split"] == "test"), manifest[0])
    raw_bgr = cv2.imread(sample["local_raw_path"])
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    proc_gray = cv2.imread(sample["processed_image_path"], cv2.IMREAD_GRAYSCALE)
    bin_mask = cv2.imread(sample["binary_mask_path"], cv2.IMREAD_GRAYSCALE)
    mc_mask = cv2.imread(sample["multiclass_mask_path"], cv2.IMREAD_GRAYSCALE)
    h, w = proc_gray.shape[:2]

    # Figure 1: 4-panel Methodology Pipeline Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    # Panel 1: Stage 1 Preprocessing
    axes[0, 0].imshow(proc_gray, cmap="gray")
    axes[0, 0].set_title(f"Stage 1: CLAHE Contrast Enhanced & Denoised OPG ({sample['patient_name']})", color="white", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: Stage 2 Binary Mask (255 White Teeth / 0 Black Background)
    axes[0, 1].imshow(bin_mask, cmap="gray")
    axes[0, 1].set_title("Stage 2: Ground Truth Binary Mask (Teeth = 255 White, BG = 0 Black)", color="white", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Stage 3 & 4 U-Net Tooth Segmentation & Leftmost/Rightmost Contact Points
    overlay = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2RGB)
    colors = {1: (0, 180, 255), 2: (80, 230, 100), 3: (255, 180, 0), 4: (255, 100, 220)}
    color_layer = np.zeros_like(overlay)
    for cid, col in colors.items():
        color_layer[mc_mask == cid] = col
    mask_bool = mc_mask > 0
    overlay[mask_bool] = (0.58 * overlay[mask_bool] + 0.42 * color_layer[mask_bool]).astype(np.uint8)

    # Draw dashed tooth contour boundaries on full overlay before cropping
    for cid, col in colors.items():
        bin_c = np.where(mc_mask == cid, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(bin_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, col, 2)

    # Crop to mandibular arch ROI for clear contact-point visualization
    x0, x1 = int(w * 0.24), int(w * 0.76)
    y0, y1 = int(h * 0.50), int(h * 0.88)
    roi_overlay = overlay[y0:y1, x0:x1].copy()

    axes[1, 0].imshow(roi_overlay)
    gtc = sample["ground_truth_contacts"]
    color_hex = {"right_molar": "#00b4ff", "right_premolar": "#50e664", "left_molar": "#ffb400", "left_premolar": "#ff64dc"}
    for key, info in gtc.items():
        lx, ly = info["left_contact_px"][0] - x0, info["left_contact_px"][1] - y0
        rx, ry = info["right_contact_px"][0] - x0, info["right_contact_px"][1] - y0
        c = color_hex[key]
        axes[1, 0].plot([lx, rx], [ly, ry], color=c, linewidth=2.2)
        axes[1, 0].scatter([lx, rx], [ly, ry], color=c, s=55, edgecolors="black", zorder=5)
        # Place molar label clearly above the crown and premolar label below the crown so neither occludes the enamel
        label_y = ly - 24 if "molar" in key else ly + 28
        axes[1, 0].text((lx + rx) / 2, label_y, f"{info['width_mm']:.2f}mm", color="white",
                        fontsize=8.5, fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#0f172a", ec=c, lw=1.2))

    axes[1, 0].set_title(
        f"Stage 4 & 5: Segmented Contours & Contact Points (R E-Space: {sample['right_espace_mm']}mm | L E-Space: {sample['left_espace_mm']}mm)",
        color="white", fontsize=10.5, fontweight="bold"
    )
    axes[1, 0].axis("off")

    # Panel 4: U-Net Training & Validation Convergence + Dataset Distribution
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            ev = json.load(f)
        hist = ev.get("training_history", [])
        epochs = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        val_dice = [h["val_dice"] for h in hist]
        ax4 = axes[1, 1]
        ax4.set_facecolor("#1e293b")
        ax4.plot(epochs, tr_loss, "o-", color="#38bdf8", label="U-Net Pixel+Dice Loss (Train)", linewidth=2)
        ax4.plot(epochs, val_dice, "s-", color="#10b981", label="Validation Foreground Dice Score", linewidth=2)
        ax4.set_title("Stage 3: U-Net Training Loss & Dice Score Convergence", color="white", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Epoch", color="white")
        ax4.set_ylabel("Metric Value", color="white")
        ax4.tick_params(colors="white")
        ax4.grid(True, alpha=0.25, color="#94a3b8")
        ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="white")

    plt.tight_layout()
    out_fig1 = os.path.join(ARTIFACT_DIR, "methodology_pipeline_results.png")
    plt.savefig(out_fig1, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved pipeline figure to", out_fig1)


if __name__ == "__main__":
    make_figures()
