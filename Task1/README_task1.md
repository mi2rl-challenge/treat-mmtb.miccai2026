# TREAT-MMTB 2026 — Task 1 Submission (Docker image)

You build a self-contained Docker image that runs **inference only** and writes
predicted masks. You then **submit the image** (as a tar). The organizers run it
on the held-out test set and score the outputs against the ground truth.

## What you submit

- A Docker image saved as a tar: `<team>-task1.tar.gz`
  (your `predict.py`, `requirements.txt`, and **weights are all baked inside**).

You do **not** submit code or weights separately, and you do **not** receive the
ground truth.

## Files in this template

| File | Edit? | Role |
|------|-------|------|
| `Dockerfile` | rarely | Python 3.10 / CUDA 11.8 environment, fixed entrypoint |
| `requirements.txt` | **yes** | add your model's dependencies |
| `predict.py` | **yes** | fill in `load_model()` and `Model.predict()` (returns mask + cavity flag) |
| `weights/` | **yes** | put your trained weights here (e.g. `model.pth`) |

## I/O contract (fixed — do not change paths)

Input (mounted at `/input`, read-only), one folder per patient, **CT only**:

```
/input/
    <our_id>/
        *.dcm        # CT series (one folder = one 3D volume)
```

Input folders are named by `our_id` (the patient id).

Output (written to `/output`):

```
/output/
    <our_id>.nii.gz    # predicted BINARY mask (0/1, uint8), same grid as the CT.
                       # Write one for EVERY patient (empty mask if no cavity).
    prediction.csv     # columns: our_id,cavity   (cavity: 1=present, 0=absent)
```

- Mask filename **must** be `<our_id>.nii.gz` (the input folder name), and the
  `our_id` in `prediction.csv` must match it.
- **Detection** is scored from `prediction.csv`; **DSC** is scored from the masks.
  Keep the two consistent.
- If you resample during inference, resample the prediction **back** to the
  original CT grid before saving (`resample_to_reference` in `predict.py`).

## Rules

- **Weights baked in.** The container runs offline (`--network none`); it cannot
  download anything at run time.
- **Inference only.** No ground truth is provided and no scoring happens inside
  the container.
- **GPU.** Test your image with `--gpus all`.

## Build, test, submit

```bash
# 1. build
docker build -t <team>-task1:latest .

# 2. self-test on a few sample patients (CT only)
mkdir -p sample_out
docker run --rm --gpus all --network none \
    -v /path/to/sample_test:/input:ro \
    -v $PWD/sample_out:/output \
    <team>-task1:latest

# 3. save the image and submit the tar
docker save <team>-task1:latest | gzip > <team>-task1.tar.gz
```

## Scoring (organizer side)

```
final_score = 0.7 × patient-level cavity detection accuracy + 0.3 × DSC
```
