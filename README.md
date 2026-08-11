# HAM10000 PyTorch Multimodal Classifier

This is a compact PyTorch project for training a 7-class HAM10000 skin-lesion classifier from:

- dermatoscopic images
- optional metadata: `age`, `sex`, `localization`

It is meant for research and education, not medical diagnosis.

## Dataset

Download HAM10000 only after reviewing and accepting its terms. The dataset is commonly distributed through Harvard Dataverse / ISIC and is non-commercial in the ISIC challenge context.

Useful references:

- HAM10000 Dataverse DOI: <https://doi.org/10.7910/DVN/DBW86T>
- Dataset paper: <https://doi.org/10.1038/sdata.2018.161>
- ISIC challenge data page: <https://challenge.isic-archive.com/data/>

Expected local layout:

```text
data/ham10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
    ISIC_0024306.jpg
    ...
  HAM10000_images_part_2/
    ...
```

The code searches recursively under `--data-dir`, so a flat `images/` directory also works.

## Install

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Train

```bash
ham10000-train \
  --data-dir /path/to/data/ham10000 \
  --metadata-csv /path/to/data/ham10000/HAM10000_metadata.csv \
  --output-dir runs/ham10000-resnet18 \
  --backbone resnet18 \
  --pretrained \
  --epochs 10 \
  --batch-size 32
```

## Smoke Test

After installing the package, you can create a tiny synthetic dataset to check that your environment can run the code:

```bash
python scripts/create_toy_ham10000.py --output-dir work/toy_ham10000

ham10000-train \
  --data-dir work/toy_ham10000 \
  --metadata-csv work/toy_ham10000/HAM10000_metadata.csv \
  --output-dir runs/toy \
  --backbone resnet18 \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --image-size 64
```

## Understand One Multimodal Batch

This script shows the actual tensors used by the PyTorch model:

```bash
python scripts/walkthrough_multimodal_batch.py \
  --data-dir . \
  --metadata-csv HAM10000_metadata.csv \
  --image-size 64 \
  --batch-size 4
```

It prints:

- image tensor shape, for example `(4, 3, 64, 64)`
- metadata tensor shape, for example `(4, 19)`
- labels shape
- model output shape, for example `(4, 7)`

For image-only training:

```bash
ham10000-train \
  --data-dir /path/to/data/ham10000 \
  --metadata-csv /path/to/data/ham10000/HAM10000_metadata.csv \
  --no-metadata
```

Outputs include:

- `best.pt`
- `last.pt`
- `config.json`
- `history.csv`
- `splits.csv`
- `test_metrics.json`

The split uses `lesion_id` when available so images of the same lesion do not leak across train/validation/test.

## Evaluate

```bash
ham10000-evaluate \
  --checkpoint runs/ham10000-resnet18/best.pt \
  --data-dir /path/to/data/ham10000 \
  --metadata-csv /path/to/data/ham10000/HAM10000_metadata.csv \
  --split test \
  --predictions-csv runs/ham10000-resnet18/test_predictions.csv
```

## Predict One Image

```bash
ham10000-predict \
  --checkpoint runs/ham10000-resnet18/best.pt \
  --image /path/to/ISIC_0024306.jpg \
  --age 55 \
  --sex male \
  --localization back
```

## Labels

```text
akiec  Actinic keratosis / Bowen's disease
bcc    Basal cell carcinoma
bkl    Benign keratosis
df     Dermatofibroma
mel    Melanoma
nv     Melanocytic nevus
vasc   Vascular lesion
```

## Notes

HAM10000 is imbalanced. This trainer uses class-weighted cross entropy and reports balanced accuracy, but serious experiments should also inspect per-class recall, calibration, and patient/lesion-level leakage.
