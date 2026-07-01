"""
TREAT-MMTB 2026 — Task 1 : inference (participant side)
======================================================

Your container does INFERENCE ONLY and writes two things to /output:
  1. a predicted mask per case      -> <our_id>.nii.gz   (used for DSC)
  2. a single prediction CSV        -> prediction.csv    (used for detection)

No ground truth, no scoring inside the container.

You fill in:
  1. load_model()      — define your architecture and load baked-in weights
  2. DummyModel.predict()   — return (mask, cavity_flag) for one X-ray case
  3. (optional) preprocessing (see the commented example in run_one_case)

I/O contract (paths are FIXED — do not change them):
  Input  (mounted at /input, read-only), one folder per case, X-ray DICOM only:
      /input/<our_id>/*.dcm

  Output (written to /output):
      /output/<our_id>.nii.gz   # predicted BINARY mask (0/1, uint8),
                                # same image grid as the input X-ray DICOM.
                                # Write one for EVERY case
                                # (an empty mask if no cavity).
      /output/prediction.csv    # columns: our_id,cavity
                                # cavity: 1=present, 0=absent

Notes:
  - <our_id> is the input folder name; mask filename and CSV our_id must match it.
  - Detection is scored from the CSV, DSC from the masks. Keep them consistent.
  - Bake your weights into the image (the container runs offline, no network).
"""

import os
import csv
import glob
import argparse

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# =============================================================================
# Data loading for X-ray DICOM
# =============================================================================
def load_xray_dicom(patient_dir: str) -> sitk.Image:
    """
    Read one X-ray DICOM image from /input/<our_id>/.

    This task uses X-ray images, not CT volumes. Therefore, the usual path is:
        one case folder -> one 2D DICOM -> one 2D mask.

    If a case folder contains multiple DICOM files, the first sorted file is used.
    """
    dcm_paths = sorted(glob.glob(os.path.join(patient_dir, "*.dcm")))
    if not dcm_paths:
        dcm_paths = sorted(
            p for p in glob.glob(os.path.join(patient_dir, "*"))
            if os.path.isfile(p)
        )

    if not dcm_paths:
        raise RuntimeError(f"No DICOM files found in: {patient_dir}")

    img = sitk.ReadImage(dcm_paths[0])

    # X-ray should normally be 2D. Some readers may return a 3D image with one
    # slice. Keep it if so, but downstream mask handling will match the dimension.
    if img.GetDimension() not in (2, 3):
        raise RuntimeError(
            f"Unsupported X-ray image dimension {img.GetDimension()} "
            f"for file: {dcm_paths[0]}"
        )

    return img


# =============================================================================
# Preprocessing helpers
# =============================================================================
def normalize_xray(xray_array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Simple X-ray intensity normalization to [0, 1].

    Unlike CT, X-ray does not use HU clipping such as [-1000, 400].
    Replace this with the exact preprocessing used by your model.
    """
    x = xray_array.astype(np.float32)
    lo, hi = np.percentile(x, (0.5, 99.5))
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + eps)


def make_mask_match_reference(mask: np.ndarray, reference: sitk.Image) -> np.ndarray:
    """
    Convert model output to a binary uint8 mask whose array dimension matches
    the SimpleITK reference image.

    SimpleITK array conventions:
      - 2D image -> array shape (Y, X)
      - 3D image -> array shape (Z, Y, X)

    Do not blindly use np.squeeze(), because a one-slice 3D mask (1, Y, X)
    can become 2D and then CopyInformation() will fail.
    """
    mask = np.asarray(mask)
    mask = (mask > 0).astype(np.uint8)

    ref_dim = reference.GetDimension()
    ref_size = reference.GetSize()  # 2D: (X, Y), 3D: (X, Y, Z)

    if ref_dim == 2:
        expected_shape = (ref_size[1], ref_size[0])  # (Y, X)

        # Allow common outputs: (1, Y, X), (Y, X, 1), or (Y, X)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if mask.ndim != 2:
            raise RuntimeError(
                f"Expected 2D mask for 2D X-ray reference, got shape {mask.shape}"
            )

        if mask.shape != expected_shape:
            raise RuntimeError(
                f"Mask shape {mask.shape} does not match reference shape {expected_shape}"
            )

    elif ref_dim == 3:
        expected_shape = (ref_size[2], ref_size[1], ref_size[0])  # (Z, Y, X)

        # If reference is one-slice 3D and model returned 2D, add Z dimension.
        if mask.ndim == 2 and ref_size[2] == 1:
            mask = mask[None, :, :]

        if mask.ndim != 3:
            raise RuntimeError(
                f"Expected 3D mask for 3D reference, got shape {mask.shape}"
            )

        if mask.shape != expected_shape:
            raise RuntimeError(
                f"Mask shape {mask.shape} does not match reference shape {expected_shape}"
            )

    else:
        raise RuntimeError(f"Unsupported reference dimension: {ref_dim}")

    return mask.astype(np.uint8)


# =============================================================================
# Model — replace this block with your own implementation
# =============================================================================
class DummyModel:
    """Placeholder. Returns an empty mask and cavity=0. Replace with your network."""

    def predict(self, xray_array: np.ndarray, xray_itk: sitk.Image = None):
        """
        Args
        ----
        xray_array : np.ndarray
            2D shape (Y, X) for standard X-ray, or 3D shape (1, Y, X) if the
            DICOM was read as a one-slice volume.
        xray_itk : sitk.Image
            Original X-ray image used for geometry.

        Returns
        -------
        mask : np.ndarray
            Binary cavity mask on the SAME grid as xray_array.
        cavity : int
            1 if cavity is present, 0 otherwise.
        """
        mask = np.zeros_like(xray_array, dtype=np.uint8)
        cavity = int(mask.sum() > 0)
        return mask, cavity


def load_model(weights_path: str):
    """
    Define your architecture and load baked-in weights.

    Example
    -------
    # import torch
    # model = MyNet()
    # model.load_state_dict(torch.load(weights_path, map_location="cuda"))
    # model.eval().cuda()
    # return model
    """
    return DummyModel()


# =============================================================================
# Per-case inference
# =============================================================================
def run_one_case(model, patient_dir: str, output_dir: str):
    our_id = os.path.basename(os.path.normpath(patient_dir))

    # 1) Load X-ray DICOM
    xray_itk = load_xray_dicom(patient_dir)
    xray_array = sitk.GetArrayFromImage(xray_itk).astype(np.float32)

    # 2) Optional preprocessing
    # Use the same preprocessing as training. For the dummy model, raw is fine.
    # xray_proc = normalize_xray(xray_array)
    xray_proc = xray_array

    # 3) Model prediction
    mask, cavity = model.predict(xray_proc, xray_itk)

    # 4) Match mask dimension/shape to original X-ray image
    mask = make_mask_match_reference(mask, xray_itk)
    cavity = int(cavity)

    # If there is no separate classifier, derive the flag from the mask.
    # Keep this line if you want CSV and mask to be consistent.
    cavity = int(mask.sum() > 0)

    # 5) Save binary mask with the ORIGINAL X-ray geometry
    pred_itk = sitk.GetImageFromArray(mask)
    pred_itk.CopyInformation(xray_itk)
    sitk.WriteImage(pred_itk, os.path.join(output_dir, f"{our_id}.nii.gz"))

    return our_id, cavity


# =============================================================================
# Entry point
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 X-ray cavity inference")
    parser.add_argument("--input", default="/input", help="test data root (case folders)")
    parser.add_argument("--output", default="/output", help="where to write masks + CSV")
    parser.add_argument(
        "--weights",
        default="/workspace/weights/model.pth",
        help="path to weights baked into the image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.weights)

    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(args.input, "*")) if os.path.isdir(d)
    )
    if not patient_dirs:
        raise RuntimeError(f"No case folders found under: {args.input}")

    rows = [
        run_one_case(model, d, args.output)
        for d in tqdm(patient_dirs, desc="Inference")
    ]

    csv_path = os.path.join(args.output, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["our_id", "cavity"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
