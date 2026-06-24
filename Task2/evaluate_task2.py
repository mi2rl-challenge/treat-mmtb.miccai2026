"""
TREAT-MMTB 2026 — Task 2 : evaluation
=====================================

Official scoring script. Safe to share with participants (it contains no
ground truth); only the test-set GT CSV is kept private.

Score = binary F1-score, with TB as the positive class (matching the baseline).

Run (organizer side), after collecting a participant's /output:
  python evaluate.py --gt-csv /path/to/gt.csv --pred-csv ./output/prediction.csv

CSV format (both files): columns  filename,TB/Normal
  - filename  : the PNG file name (e.g. abc.png)
  - TB/Normal : "TB" or "Normal"
If your GT CSV uses a different filename column header, pass --gt-filename-col.
"""

import csv
import argparse


def to_label(value) -> int:
    """Map TB/Normal to 1/0 (TB = positive)."""
    s = str(value).strip().lower()
    if s in ("tb", "1"):
        return 1
    if s in ("normal", "0"):
        return 0
    raise ValueError(f"Unexpected label: {value!r}")


def read_csv(path: str, filename_col: str, label_col: str = "TB/Normal") -> dict:
    """Read a CSV into {filename: 0/1}."""
    table = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row[filename_col].strip()] = to_label(row[label_col])
    return table


def f1_score(gt: dict, pred: dict) -> dict:
    tp = fp = fn = tn = 0
    for filename, g in gt.items():
        if filename not in pred:
            raise KeyError(f"Missing prediction for: {filename}")
        p = pred[filename]
        if p == 1 and g == 1:
            tp += 1
        elif p == 1 and g == 0:
            fp += 1
        elif p == 0 and g == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 evaluation (F1)")
    parser.add_argument("--gt-csv", required=True, help="GT CSV (filename, TB/Normal)")
    parser.add_argument("--pred-csv", required=True, help="predicted CSV (filename, TB/Normal)")
    parser.add_argument("--gt-filename-col", default="filename",
                        help="filename column header in the GT CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    gt = read_csv(args.gt_csv, filename_col=args.gt_filename_col)
    pred = read_csv(args.pred_csv, filename_col="filename")
    r = f1_score(gt, pred)

    print(f"Precision : {r['precision']:.4f}")
    print(f"Recall    : {r['recall']:.4f}")
    print(f"F1 Score  : {r['f1']:.4f}")


if __name__ == "__main__":
    main()
