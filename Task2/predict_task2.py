"""
TREAT-MMTB 2026 — Task 2 : TB diagnosis inference (participant side)
==================================================================

Your container does INFERENCE ONLY: read test PNGs -> predict TB/Normal ->
write a single prediction CSV. No ground truth, no scoring inside the container.

You fill in:
  1. load_model()       — define your architecture and load baked-in weights
  2. build_transform()  — your preprocessing (must match how you trained)

A runnable ResNet50 example (the baseline) is provided below; replace it with
your own model.

I/O contract (paths are FIXED — do not change them):
  Input  (mounted at /input, read-only):
      /input/*.png                     # test chest X-ray images
  Output (written to /output):
      /output/prediction.csv           # columns: filename,TB/Normal
                                       # filename = the PNG file name (e.g. abc.png)
                                       # TB/Normal = "TB" or "Normal"

Notes:
  - The "filename" values must match the GT CSV's filename column.
  - Bake your weights into the image (the container runs offline, no network).
"""

import os
import csv
import glob
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# Label encoding (TB is the positive class, matching the baseline).
ID2LABEL = {0: "Normal", 1: "TB"}


# =============================================================================
# Preprocessing — replace with the transform you trained with
# =============================================================================
def build_transform():
    """
    The test images are RAW PNGs. Reproduce here whatever preprocessing your
    model was trained with — if you trained on preprocessed inputs (e.g. the
    ch0/ch1/ch2 mean-std / CLAHE channels), apply that SAME pipeline to the raw
    image first; the torchvision transform below alone does not reproduce it.

    The block below is only the baseline's final normalization. Add a Resize
    (e.g. transforms.Resize((512, 512))) if your training size differs.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =============================================================================
# Model — replace with your own architecture + weight loading
# =============================================================================
def load_model(weights_path: str):
    """
    Example: ResNet50 with a 2-class head (the baseline).
    Loads weights baked into the image.
    """
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ID2LABEL))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    return model


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def predict_all(model, transform, image_paths):
    rows = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Inference"):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(DEVICE)

        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().tolist()

        for path, pred in zip(batch_paths, preds):
            rows.append((os.path.basename(path), ID2LABEL[int(pred)]))
    return rows


# =============================================================================
# Entry point
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 inference")
    parser.add_argument("--input", default="/input", help="folder of test PNGs")
    parser.add_argument("--output", default="/output", help="where to write prediction.csv")
    parser.add_argument("--weights", default="/workspace/weights/model.pth",
                        help="path to weights baked into the image")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
    if not image_paths:
        raise RuntimeError(f"No PNG images found under: {args.input}")

    model = load_model(args.weights)
    transform = build_transform()
    rows = predict_all(model, transform, image_paths)

    csv_path = os.path.join(args.output, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "TB/Normal"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
