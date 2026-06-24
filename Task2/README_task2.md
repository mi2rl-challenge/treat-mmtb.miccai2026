# TREAT-MMTB 2026 — Task 2 Submission (TB diagnosis)

Build a self-contained Docker image that runs **inference only** and writes a
prediction CSV. Submit the image (as a tar). The organizers run it on the
held-out external test set and score the CSV against the ground truth.

## What you submit

- A Docker image saved as a tar: `<team>-task2.tar.gz`
  (`predict.py`, `requirements.txt`, and **weights baked inside**).

## Files in this template

| File | Edit? | Role |
|------|-------|------|
| `Dockerfile` | rarely | Python 3.10 / CUDA 11.8, torch+torchvision, fixed entrypoint |
| `requirements.txt` | **yes** | add any extra dependencies your model needs |
| `predict.py` | **yes** | fill in `load_model()` and `build_transform()` |
| `weights/` | **yes** | put your trained weights here (e.g. `model.pth`) |

A runnable ResNet50 example (the baseline) is included in `predict.py`; replace
it with your own model and preprocessing.

## I/O contract (fixed — do not change paths)

Input (mounted at `/input`, read-only):

```
/input/*.png        # test chest X-ray images (external set, RAW PNGs)
```

The test images are **raw** (not preprocessed). Reproduce your training
preprocessing inside the container — if you trained on preprocessed inputs
(e.g. ch0/ch1/ch2 channels), apply that same pipeline to the raw image in
`build_transform()` (or before it).

Output (written to `/output`):

```
/output/prediction.csv      # columns: filename,TB/Normal
                            # filename  = the PNG file name (e.g. abc.png)
                            # TB/Normal = "TB" or "Normal"
```

- The `filename` values must match the ground-truth CSV's filename column.
- Make sure your preprocessing in `build_transform()` matches how you trained
  (size, normalization).

## Rules

- **Weights baked in.** The container runs offline (`--network none`).
- **Inference only.** No ground truth, no scoring inside the container.
- **GPU.** Test your image with `--gpus all`.

## Build, test, submit

```bash
# 1. build
docker build -t <team>-task2:latest .

# 2. self-test on a few sample PNGs
mkdir -p sample_out
docker run --rm --gpus all --network none \
    -v /path/to/sample_test:/input:ro \
    -v $PWD/sample_out:/output \
    <team>-task2:latest

# 3. save the image and submit the tar
docker save <team>-task2:latest | gzip > <team>-task2.tar.gz
```

## Scoring

```
score = binary F1-score (TB = positive class)
```
