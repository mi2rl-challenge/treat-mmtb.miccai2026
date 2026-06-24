"""
TREAT-MMTB 2026 — Task 1 : evaluation
=====================================

Official scoring script. Safe to share with participants (it contains no
ground truth); only the test-set GT (CSV + masks) is kept private.

  - Detection accuracy : from the CSVs (our_id, cavity 0/1)
  - DSC                : from the masks (<our_id>.nii.gz)

  final_score = 0.7 * detection_accuracy + 0.3 * mean_dice

Run (organizer side), after collecting a participant's /output:
  python evaluate.py \
      --gt-csv        /path/to/gt.csv \
      --pred-csv      ./output/prediction.csv \
      --gt-mask-dir   /path/to/gt_masks \
      --pred-mask-dir ./output
"""

import os
import csv
import glob
import argparse

import numpy as np
import SimpleITK as sitk


# =============================================================================
# Detection (CSV)
# =============================================================================
def to_binary_cavity(value) -> int:
    """Map a cavity label to 0/1. Accepts 0/1 and large/medium/small/none."""
    s = str(value).strip().lower()
    if s in ("0", "none", "no", "false", ""):
        return 0
    return 1


def read_cavity_csv(path: str) -> dict:
    """Read a CSV with columns our_id,cavity into {our_id: 0/1}."""
    table = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row["our_id"].strip()] = to_binary_cavity(row["cavity"])
    return table


def detection_accuracy(gt_csv: str, pred_csv: str) -> float:
    gt = read_cavity_csv(gt_csv)
    pred = read_cavity_csv(pred_csv)

    correct = []
    for our_id, gt_flag in gt.items():
        if our_id not in pred:
            raise KeyError(f"Missing prediction in CSV for: {our_id}")
        correct.append(float(gt_flag == pred[our_id]))
    return float(np.mean(correct))


# =============================================================================
# Segmentation (masks)
# =============================================================================
def load_binary_mask(path: str) -> np.ndarray:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(path))
    mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    if pred_sum == 0 and gt_sum == 0:
        return np.nan
    if pred_sum == 0 and gt_sum > 0:
        return 0.0
    if pred_sum > 0 and gt_sum == 0:
        return 0.0
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred_sum + gt_sum)


def mean_dice(gt_mask_dir: str, pred_mask_dir: str) -> float:
    gt_paths = sorted(glob.glob(os.path.join(gt_mask_dir, "*.nii.gz")))
    if not gt_paths:
        raise RuntimeError(f"No GT masks found in: {gt_mask_dir}")

    dices = []
    for gt_path in gt_paths:
        our_id = os.path.basename(gt_path).replace(".nii.gz", "")
        pred_path = os.path.join(pred_mask_dir, f"{our_id}.nii.gz")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction mask for {our_id}: {pred_path}")
        dices.append(dice_score(load_binary_mask(pred_path), load_binary_mask(gt_path)))

    # both-empty cases are NaN and excluded by nanmean.
    # (Alternative: score DSC only on GT-positive cases — see note in chat.)
    return float(np.nanmean(dices))


# =============================================================================
# Entry point
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 evaluation")
    parser.add_argument("--gt-csv", required=True, help="GT CSV (our_id,cavity)")
    parser.add_argument("--pred-csv", required=True, help="predicted CSV (our_id,cavity)")
    parser.add_argument("--gt-mask-dir", required=True, help="GT mask dir (*.nii.gz)")
    parser.add_argument("--pred-mask-dir", required=True, help="predicted mask dir (*.nii.gz)")
    return parser.parse_args()


def main():
    args = parse_args()
    det_acc = detection_accuracy(args.gt_csv, args.pred_csv)
    dsc = mean_dice(args.gt_mask_dir, args.pred_mask_dir)
    final_score = 0.7 * det_acc + 0.3 * dsc

    print(f"Detection Accuracy : {det_acc:.4f}")
    print(f"Mean Dice          : {dsc:.4f}")
    print(f"Final Score        : {final_score:.4f}")


if __name__ == "__main__":
    main()
