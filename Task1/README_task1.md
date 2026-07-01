# TREAT-MMTB 2026 — Task 1 Submission (Docker image)

You build a self-contained Docker image that runs **inference only** for X-ray cavity segmentation. The organizers will run your image on the held-out test set and score the outputs against the private ground truth.

## What you submit

- A Docker image saved as a tar archive: `<team>-task1.tar.gz`
- Your `predict.py`, `requirements.txt`, model code, and **weights must be baked inside the image**.
- You do **not** submit code or weights separately.
- You do **not** receive the ground truth.

## Files in this template

| File | Edit? | Role |
|------|-------|------|
| `Dockerfile` | rarely | Python 3.10 / CUDA 11.8 environment with fixed entrypoint |
| `requirements.txt` | yes | Add your model dependencies |
| `predict.py` | yes | Fill in `load_model()` and `Model.predict()` |
| `weights/` | yes | Put your trained weights here, for example `weights/model.pth` |

## I/O contract: fixed paths

Input is mounted at `/input` as read-only. Each case is one folder containing X-ray DICOM file(s):

```text
/input/
    <our_id>/
        *.dcm
```

Output must be written to `/output`:

```text
/output/
    <our_id>.nii.gz    # predicted binary cavity mask, 0/1, uint8
                       # same image grid as the input X-ray DICOM
                       # write one mask for every case, including empty masks
    prediction.csv     # columns: our_id,cavity
                       # cavity: 1=present, 0=absent
```

Rules:

- The mask filename must be `<our_id>.nii.gz`, where `<our_id>` is the input folder name.
- The `our_id` values in `prediction.csv` must exactly match the input folder names.
- Detection is scored from `prediction.csv`.
- Dice similarity coefficient is scored from the `.nii.gz` masks.
- Keep the CSV and mask predictions consistent.
- If you resize, crop, pad, or resample during inference, convert the final mask back to the original X-ray DICOM image grid before saving.

## Runtime rules

- The container runs offline with `--network none`; it cannot download model weights, packages, or checkpoints at runtime.
- Ground truth is not provided inside the container.
- Scoring does not run inside the participant container.
- The image should run with GPU using `--gpus all`.

## Build, test, and submit

```bash
# 1. Build
docker build -t <team>-task1:latest .

# 2. Self-test on sample X-ray cases
mkdir -p sample_out
docker run --rm --gpus all --network none \
    -v /path/to/sample_xray_cases:/input:ro \
    -v $PWD/sample_out:/output \
    <team>-task1:latest

# 3. Save the image and submit the tar
docker save <team>-task1:latest | gzip > <team>-task1.tar.gz
```

## Expected output check

After a successful self-test, `/output` should contain:

```text
prediction.csv
<our_id_1>.nii.gz
<our_id_2>.nii.gz
...
```

`prediction.csv` should look like:

```csv
our_id,cavity
case001,1
case002,0
```

## Scoring

Organizer-side scoring:

```text
final_score = 0.7 × patient-level cavity detection accuracy + 0.3 × DSC
```
