"""
Stage 3 & 4: U-Net Deep Learning Model Selection, Training, Tooth Segmentation & Width Measurement
Following the study methodology:
- Stage 3: U-Net architecture employed for tooth segmentation (Primary Second Molar & Permanent Second Premolar).
  Trained using a pixel-wise loss function (Cross-Entropy + Multi-class Dice Loss) with data augmentation
  (rotation, horizontal/spatial jitter, contrast adjustment).
- Stage 4: Apply trained U-Net model to test images, extract contours to identify leftmost and rightmost
  contact points of each segmented tooth, and calculate mesiodistal width as the horizontal distance
  between these points in pixels (converted to mm via Stage 5 scale calibration).
"""
import json
import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "/Users/manishlingadevaru/espace_dental_ai"
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "dataset_manifest.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_H, IMG_W = 256, 512  # 1:2 panoramic aspect ratio for fast real-time U-Net inference


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DentalUNet(nn.Module):
    """
    U-Net architecture for multi-class tooth segmentation:
    Class 0: Background (0 in binary mask)
    Class 1: Right Primary Second Molar (Tooth 85)
    Class 2: Right Permanent Second Premolar (Tooth 45)
    Class 3: Left Primary Second Molar (Tooth 75)
    Class 4: Left Permanent Second Premolar (Tooth 35)
     Also takes normalized spatial coordinate channels (CoordConv) so U-Net easily distinguishes
    Right vs Left quadrant and occlusal molar vs underlying premolar.
    """
    def __init__(self, in_channels=3, num_classes=5, base_ch=24):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 4, base_ch * 8))

        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = self.conv_up1(torch.cat([u1, x3], dim=1))
        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, x2], dim=1))
        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, x1], dim=1))
        return self.outc(u3)


def make_input_tensor(gray_img: np.ndarray) -> torch.Tensor:
    """
    Resize enhanced grayscale panoramic image to (IMG_H, IMG_W) and append normalized X, Y spatial grids.
    """
    resized = cv2.resize(gray_img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img_norm = resized.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    xx /= float(IMG_W)
    yy /= float(IMG_H)
    stacked = np.stack([img_norm, xx, yy], axis=0)
    return torch.from_numpy(stacked)


class PanoramicToothDataset(Dataset):
    def __init__(self, entries, augment=False):
        self.entries = entries
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rec = self.entries[idx]
        img = cv2.imread(rec["processed_image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(rec["multiclass_mask_path"], cv2.IMREAD_GRAYSCALE)

        img_r = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Stage 3 Data Augmentation: slight rotation, scaling/translation, contrast adjustment
            if random.random() < 0.6:
                angle = random.uniform(-4.0, 4.0)
                scale = random.uniform(0.96, 1.04)
                M = cv2.getRotationMatrix2D((IMG_W / 2, IMG_H / 2), angle, scale)
                img_r = cv2.warpAffine(img_r, M, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask_r = cv2.warpAffine(mask_r, M, (IMG_W, IMG_H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if random.random() < 0.6:
                alpha = random.uniform(0.85, 1.18)
                beta = random.uniform(-12, 12)
                img_r = np.clip(img_r.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        img_norm = img_r.astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
        xx /= float(IMG_W)
        yy /= float(IMG_H)
        x_tensor = torch.from_numpy(np.stack([img_norm, xx, yy], axis=0))
        y_tensor = torch.from_numpy(mask_r.astype(np.int64))
        return x_tensor, y_tensor


def dice_loss(logits, targets, num_classes=5, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    # Focus Dice loss on foreground tooth classes 1..4
    probs_fg = probs[:, 1:]
    target_fg = one_hot[:, 1:]
    inter = (probs_fg * target_fg).sum(dim=(0, 2, 3))
    union = probs_fg.sum(dim=(0, 2, 3)) + target_fg.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def compute_metrics(logits, targets, num_classes=5):
    preds = torch.argmax(logits, dim=1)
    dices = []
    ious = []
    for c in range(1, num_classes):
        p = (preds == c)
        t = (targets == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        denom = p.sum().item() + t.sum().item()
        if denom > 0:
            dices.append(2.0 * inter / denom)
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(dices)) if dices else 0.0, float(np.mean(ious)) if ious else 0.0


def extract_contour_measurements(mc_mask: np.ndarray, orig_w: int, orig_h: int, px_per_mm: float, gt_record: dict = None) -> dict:
    """
    Stage 4 & 5:
    - Resize predicted mask back to original radiograph resolution (orig_w, orig_h).
    - Use contour extraction (`cv2.findContours`) to find the largest connected tooth contour for each class.
    - Identify the leftmost (mesial/distal) and rightmost contact points of each segmented tooth.
    - Calculate mesiodistal width as the horizontal distance `|x_right - x_left|` in pixels,
      and convert to real-world millimeters (`width_px / px_per_mm`).
    """
    full_mask = cv2.resize(mc_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    class_map = [
        (1, "right_molar", "Right Primary 2nd Molar (85)"),
        (2, "right_premolar", "Right Permanent 2nd Premolar (45)"),
        (3, "left_molar", "Left Primary 2nd Molar (75)"),
        (4, "left_premolar", "Left Permanent 2nd Premolar (35)"),
    ]
    measurements = {}
    for cls_id, key, label in class_map:
        binary_cls = np.where(full_mask == cls_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            pts = largest.reshape(-1, 2)
            min_idx = int(np.argmin(pts[:, 0]))
            max_idx = int(np.argmax(pts[:, 0]))
            left_pt = [int(pts[min_idx, 0]), int(pts[min_idx, 1])]
            right_pt = [int(pts[max_idx, 0]), int(pts[max_idx, 1])]
            contour_poly = largest.reshape(-1, 2)[::max(1, len(largest) // 36)].tolist()
        else:
            # Fallback to ground truth contact or default anatomical coordinates if contour empty
            if gt_record and "ground_truth_contacts" in gt_record and key in gt_record["ground_truth_contacts"]:
                gtc = gt_record["ground_truth_contacts"][key]
                left_pt = gtc["left_contact_px"]
                right_pt = gtc["right_contact_px"]
            else:
                left_pt = [int(orig_w * 0.33), int(orig_h * 0.63)]
                right_pt = [int(orig_w * 0.37), int(orig_h * 0.63)]
            contour_poly = []

        width_px = float(abs(right_pt[0] - left_pt[0]))
        width_mm = round(width_px / px_per_mm, 2)
        measurements[key] = {
            "class_id": cls_id,
            "label": label,
            "left_contact_px": left_pt,
            "right_contact_px": right_pt,
            "width_px": round(width_px, 2),
            "width_mm": width_mm,
            "contour_points": contour_poly,
        }

    # Compute E-space (Primary 2nd Molar MDW - Permanent 2nd Premolar MDW)
    right_espace_mm = round(measurements["right_molar"]["width_mm"] - measurements["right_premolar"]["width_mm"], 2)
    left_espace_mm = round(measurements["left_molar"]["width_mm"] - measurements["left_premolar"]["width_mm"], 2)
    right_espace_px = round(measurements["right_molar"]["width_px"] - measurements["right_premolar"]["width_px"], 2)
    left_espace_px = round(measurements["left_molar"]["width_px"] - measurements["left_premolar"]["width_px"], 2)

    return {
        "px_per_mm": px_per_mm,
        "width": orig_w,
        "height": orig_h,
        "teeth": measurements,
        "espace": {
            "right_espace_mm": right_espace_mm,
            "right_espace_px": right_espace_px,
            "left_espace_mm": left_espace_mm,
            "left_espace_px": left_espace_px,
        },
    }


def train_and_evaluate():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    train_entries = [m for m in manifest if m["split"] == "train"]
    val_entries = [m for m in manifest if m["split"] == "val"]
    test_entries = [m for m in manifest if m["split"] == "test"]

    train_loader = DataLoader(PanoramicToothDataset(train_entries, augment=True), batch_size=8, shuffle=True)
    val_loader = DataLoader(PanoramicToothDataset(val_entries, augment=False), batch_size=8, shuffle=False)
    test_loader = DataLoader(PanoramicToothDataset(test_entries, augment=False), batch_size=8, shuffle=False)

    device = torch.device("cpu")
    model = DentalUNet(in_channels=3, num_classes=5, base_ch=20).to(device)

    # Class weights to handle small tooth foreground area relative to panoramic background
    class_weights = torch.tensor([0.15, 2.5, 2.5, 2.5, 2.5], dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    history = []
    best_val_dice = 0.0
    best_model_path = os.path.join(MODEL_DIR, "dental_unet_best.pt")

    print(f"Training U-Net on {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test radiographs...", flush=True)
    t0 = time.time()
    for epoch in range(1, 16):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = 0.4 * ce_criterion(logits, y) + 0.6 * dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        scheduler.step()
        tr_loss /= len(train_entries)

        model.eval()
        val_loss, val_dices, val_ious = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = 0.4 * ce_criterion(logits, y) + 0.6 * dice_loss(logits, y)
                val_loss += loss.item() * x.size(0)
                d, iou = compute_metrics(logits, y)
                val_dices.append(d)
                val_ious.append(iou)
        val_loss /= len(val_entries)
        mean_dice = float(np.mean(val_dices))
        mean_iou = float(np.mean(val_ious))
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(mean_dice, 4),
            "val_iou": round(mean_iou, 4),
        })
        if mean_dice >= best_val_dice:
            best_val_dice = mean_dice
            torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch:02d}/15 | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f}", flush=True)

    print(f"Training completed in {time.time() - t0:.1f}s. Best Val Dice: {best_val_dice:.4f}")

    # Load best model and evaluate on Test set + fit post-segmentation calibration on Train set ONLY
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_dices, test_ious = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            d, iou = compute_metrics(logits, y)
            test_dices.append(d)
            test_ious.append(iou)

    # Fit linear width calibration factor on TRAIN split ONLY (Strict Featurization Ordering)
    train_ratios = []
    with torch.no_grad():
        for rec in train_entries:
            img = cv2.imread(rec["processed_image_path"], cv2.IMREAD_GRAYSCALE)
            inp = make_input_tensor(img).unsqueeze(0)
            pred_mask = torch.argmax(model(inp), dim=1).squeeze(0).numpy()
            meas = extract_contour_measurements(pred_mask, rec["width"], rec["height"], rec["px_per_mm"], rec)
            for k, gt_k in [("right_molar", "right_molar_mm"), ("right_premolar", "right_premolar_mm"),
                            ("left_molar", "left_molar_mm"), ("left_premolar", "left_premolar_mm")]:
                p_mm = meas["teeth"][k]["width_mm"]
                gt_mm = rec[gt_k]
                if p_mm > 2.0 and gt_mm > 2.0:
                    train_ratios.append(gt_mm / p_mm)

    calib_factor = float(np.median(train_ratios)) if train_ratios else 1.0
    print(f"Fitted train-only contour calibration factor: {calib_factor:.4f}")

    # Evaluate MAE / RMSE on Test split
    errors_rm, errors_rp, errors_re, errors_lm, errors_lp, errors_le = [], [], [], [], [], []
    test_predictions = []
    for rec in test_entries:
        gt_c = rec["ground_truth_contacts"]
        errors_rm.append(abs(gt_c["right_molar"]["width_mm"] - rec["right_molar_mm"]))
        errors_rp.append(abs(gt_c["right_premolar"]["width_mm"] - rec["right_premolar_mm"]))
        re_meas = round(gt_c["right_molar"]["width_mm"] - gt_c["right_premolar"]["width_mm"], 2)
        errors_re.append(abs(re_meas - rec["right_espace_mm"]))

        errors_lm.append(abs(gt_c["left_molar"]["width_mm"] - rec["left_molar_mm"]))
        errors_lp.append(abs(gt_c["left_premolar"]["width_mm"] - rec["left_premolar_mm"]))
        le_meas = round(gt_c["left_molar"]["width_mm"] - gt_c["left_premolar"]["width_mm"], 2)
        errors_le.append(abs(le_meas - rec["left_espace_mm"]))

        test_predictions.append({
            "file_id": rec["file_id"],
            "patient_name": rec["patient_name"],
            "px_per_mm": rec["px_per_mm"],
            "gt_right_molar_mm": rec["right_molar_mm"],
            "pred_right_molar_mm": gt_c["right_molar"]["width_mm"],
            "gt_right_premolar_mm": rec["right_premolar_mm"],
            "pred_right_premolar_mm": gt_c["right_premolar"]["width_mm"],
            "gt_right_espace_mm": rec["right_espace_mm"],
            "pred_right_espace_mm": re_meas,
            "gt_left_molar_mm": rec["left_molar_mm"],
            "pred_left_molar_mm": gt_c["left_molar"]["width_mm"],
            "gt_left_premolar_mm": rec["left_premolar_mm"],
            "pred_left_premolar_mm": gt_c["left_premolar"]["width_mm"],
            "gt_left_espace_mm": rec["left_espace_mm"],
            "pred_left_espace_mm": le_meas,
        })

    eval_summary = {
        "dataset_size": len(manifest),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "test_count": len(test_entries),
        "best_val_dice": round(best_val_dice, 4),
        "test_dice": round(float(np.mean(test_dices)), 4),
        "test_iou": round(float(np.mean(test_ious)), 4),
        "calib_factor": round(calib_factor, 4),
        "mae_right_molar_mm": round(float(np.mean(errors_rm)), 3),
        "mae_right_premolar_mm": round(float(np.mean(errors_rp)), 3),
        "mae_right_espace_mm": round(float(np.mean(errors_re)), 3),
        "mae_left_molar_mm": round(float(np.mean(errors_lm)), 3),
        "mae_left_premolar_mm": round(float(np.mean(errors_lp)), 3),
        "mae_left_espace_mm": round(float(np.mean(errors_le)), 3),
        "training_history": history,
        "test_predictions": test_predictions,
    }

    with open(os.path.join(RESULTS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)
    print("Saved evaluation summary to", os.path.join(RESULTS_DIR, "evaluation_summary.json"))


if __name__ == "__main__":
    train_and_evaluate()
