"""
TREAT-MMTB 2026 — Task 1 : inference (participant side)
======================================================

Your container does INFERENCE ONLY and writes two things to /output:
  1. a predicted mask per patient   -> <our_id>.nii.gz   (used for DSC)
  2. a single prediction CSV         -> prediction.csv    (used for detection)

No ground truth, no scoring inside the container.

You fill in:
  1. load_model()      — define your architecture and load baked-in weights
  2. Model.predict()   — return (mask, cavity_flag) for one patient
  3. (optional) preprocessing (see the commented example in run_one_case)

I/O contract (paths are FIXED — do not change them):
  Input  (mounted at /input, read-only), one folder per patient, CT only:
      /input/<our_id>/*.dcm

  Output (written to /output):
      /output/<our_id>.nii.gz   # predicted BINARY mask (0/1, uint8),
                                # same grid as the CT. Write one for EVERY
                                # patient (an empty mask if no cavity).
      /output/prediction.csv    # columns: our_id,cavity  (cavity: 1=present, 0=absent)

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
# Data loading
# =============================================================================
def load_dicom_series(patient_dir: str) -> sitk.Image:
    """Read a DICOM series in `patient_dir` into a single 3D image."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(patient_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {patient_dir}")
    file_names = reader.GetGDCMSeriesFileNames(patient_dir, series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()


# =============================================================================
# (Optional) example preprocessing — replace/adapt to your own pipeline
# =============================================================================
def normalize(ct_array: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """Clip HU to [hu_min, hu_max] and scale to [0, 1]."""
    ct = np.clip(ct_array, hu_min, hu_max)
    return (ct - hu_min) / (hu_max - hu_min)


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_label=True) -> sitk.Image:
    """Resample `moving` onto the grid of `reference` (e.g. prediction back to CT grid)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(moving)


# =============================================================================
# Model — replace this whole block with your own implementation
# =============================================================================
class DummyModel:
    """Placeholder. Returns an empty mask and cavity=0. Replace with your network."""

    def predict(self, ct_array: np.ndarray, ct_itk: sitk.Image = None):
        """
        Args
        ----
        ct_array : np.ndarray, shape (Z, Y, X), HU values
        ct_itk   : the original CT image (for geometry, if you resample)

        Returns
        -------
        mask   : np.ndarray, binary (cavity=1) on the SAME grid as the CT
        cavity : int, 1 if this patient has a cavity else 0

        If you do not have a separate classifier, you may derive the flag from
        the mask: cavity = int(mask.sum() > 0).
        """
        # TODO: run your model here.
        mask = np.zeros_like(ct_array, dtype=np.uint8)
        cavity = 0
        return mask, cavity


def load_model(weights_path: str):
    """
    Define your architecture and load weights baked into the image.

    Example
    -------
    # import torch
    # model = MyNet()
    # model.load_state_dict(torch.load(weights_path, map_location="cuda"))
    # model.eval().cuda()
    # return model
    """
    # TODO: replace with real model loading.
    return DummyModel()


# =============================================================================
# Per-case inference
# =============================================================================
def run_one_case(model, patient_dir: str, output_dir: str):
    our_id = os.path.basename(os.path.normpath(patient_dir))

    ct_itk = load_dicom_series(patient_dir)                         # reference grid
    ct_array = sitk.GetArrayFromImage(ct_itk).astype(np.float32)   # (Z, Y, X), HU

    # ---- preprocessing (example; uncomment / adapt to your pipeline) --------
    # ct_proc = normalize(ct_array, hu_min=-1000.0, hu_max=400.0)
    ct_proc = ct_array

    mask, cavity = model.predict(ct_proc, ct_itk)
    mask = (np.squeeze(mask) > 0).astype(np.uint8)
    cavity = int(cavity)

    # Save mask with the ORIGINAL CT geometry so DSC against the GT is correct.
    pred_itk = sitk.GetImageFromArray(mask)
    pred_itk.CopyInformation(ct_itk)
    sitk.WriteImage(pred_itk, os.path.join(output_dir, f"{our_id}.nii.gz"))

    return our_id, cavity


# =============================================================================
# Entry point
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 inference")
    parser.add_argument("--input", default="/input", help="test data root (patient folders)")
    parser.add_argument("--output", default="/output", help="where to write masks + CSV")
    parser.add_argument("--weights", default="/workspace/weights/model.pth",
                        help="path to weights baked into the image")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    model = load_model(args.weights)

    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(args.input, "*")) if os.path.isdir(d)
    )
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found under: {args.input}")

    rows = [run_one_case(model, d, args.output)
            for d in tqdm(patient_dirs, desc="Inference")]

    # Write the detection CSV: our_id,cavity
    csv_path = os.path.join(args.output, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["our_id", "cavity"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
