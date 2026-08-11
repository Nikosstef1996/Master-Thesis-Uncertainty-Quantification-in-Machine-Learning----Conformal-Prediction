# Master Thesis Code: Conformal Prediction and Multimodal Learning

This repository contains the code used for experiments on uncertainty
quantification with conformal prediction in:

- classification problems
- regression problems
- multimodal HAM10000 skin-lesion classification

The repository contains code and notebooks only. Datasets, trained checkpoints,
virtual environments, generated runs, and large feature/model files are ignored.

## Repository Structure

```text
classification/
  CP/                         reusable classification conformal classes
  app/                        classification API example
  notebook/                   Iris and imbalanced-class notebooks
  train.py

regression/
  CP/                         reusable regression conformal classes
  housing_cp_api/             California housing API example
  notebook/                   regression notebook

Multimodal Conformal Prediction/
  conformal_prediction/       shared conformal prediction classes
  ham10000_multimodal/        HAM10000 multimodal ML code

notebooks/
  ham10000_multimodal_conformal_prediction.ipynb
```

## HAM10000 Multimodal Code

The multimodal package supports:

- PyTorch image and metadata classification
- Random Forest on fused image and metadata features
- fusion-strategy comparison
- attention-based fusion
- conformal prediction for Random Forest and attention-fusion models

Main modules:

```text
Multimodal Conformal Prediction/ham10000_multimodal/data.py
Multimodal Conformal Prediction/ham10000_multimodal/model.py
Multimodal Conformal Prediction/ham10000_multimodal/train.py
Multimodal Conformal Prediction/ham10000_multimodal/evaluate.py
Multimodal Conformal Prediction/ham10000_multimodal/predict.py
Multimodal Conformal Prediction/ham10000_multimodal/random_forest.py
Multimodal Conformal Prediction/ham10000_multimodal/fusion_comparison.py
Multimodal Conformal Prediction/ham10000_multimodal/attention_fusion.py
Multimodal Conformal Prediction/ham10000_multimodal/conformal_random_forest.py
Multimodal Conformal Prediction/ham10000_multimodal/conformal_attention_fusion.py
```

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## HAM10000 Dataset

Download HAM10000 only after reviewing and accepting its terms.

Useful references:

- HAM10000 Dataverse DOI: <https://doi.org/10.7910/DVN/DBW86T>
- Dataset paper: <https://doi.org/10.1038/sdata.2018.161>
- ISIC challenge data page: <https://challenge.isic-archive.com/data/>

Expected local layout:

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

These files are intentionally not uploaded to GitHub.

## Example Commands

Train the PyTorch multimodal model:

```bash
ham10000-train \
  --data-dir . \
  --metadata-csv HAM10000_metadata.csv \
  --output-dir runs/ham10000-resnet18 \
  --backbone resnet18 \
  --pretrained \
  --epochs 10 \
  --batch-size 32
```

Train the Random Forest multimodal model:

```bash
ham10000-random-forest \
  --data-dir . \
  --metadata-csv HAM10000_metadata.csv \
  --output-dir outputs/ham10000-random-forest
```

Run fusion comparison:

```bash
ham10000-fusion-comparison \
  --features-npz outputs/ham10000-random-forest/fused_features.npz \
  --output-dir outputs/ham10000-fusion-comparison
```

Run attention fusion:

```bash
ham10000-attention-fusion \
  --features-npz outputs/ham10000-random-forest/fused_features.npz \
  --output-dir outputs/ham10000-attention-fusion
```

Run conformal prediction for attention fusion:

```bash
ham10000-conformal-attention \
  --attention-dir outputs/ham10000-attention-fusion \
  --output-dir outputs/ham10000-attention-fusion/conformal_alpha_0_10 \
  --alpha 0.10
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

HAM10000 is strongly imbalanced. The experiments therefore report both standard
accuracy and balanced accuracy, and conformal prediction is used to quantify
uncertainty through prediction sets.
