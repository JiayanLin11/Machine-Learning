# Knee KL Grading

An ordinal deep learning framework for automated Kellgren-Lawrence (KL 0--4)
grading of knee osteoarthritis from X-ray images.

The project combines a pretrained ConvNeXt-Tiny backbone with Triplet Attention,
Class Distance Weighted Cross-Entropy (CDW-CE), class-imbalance handling, and an
ordinal supervised contrastive auxiliary loss.

## Method

The default model follows this pipeline:

```text
X-ray image
    -> ConvNeXt-Tiny
    -> Triplet Attention
    -> Global average pooling
    -> Classification head (KL 0--4)
    -> Projection head (training only)
```

The ConvNeXt backbone itself is not structurally modified. Triplet Attention and
the two output heads are added after the backbone feature map.

Main components:

- **Triplet Attention** refines the final pre-pooling feature map through
  cross-dimensional channel-spatial interactions.
- **CDW-CE** assigns larger penalties to predictions farther from the true KL
  grade.
- **Class weighting** reduces bias caused by the long-tailed KL distribution.
- **Ordinal contrastive learning** pulls same-grade samples together and gives
  greater separation pressure to negative pairs with larger KL distance.
- **Five-fold probability ensembling** averages test probabilities from the
  five cross-validation models for each seed.

## Project Structure

```text
knee-kl-grading/
├── knee_kl/
│   ├── attention.py       # Attention modules
│   ├── calibration.py     # Temperature scaling
│   ├── checkpoints.py     # Checkpoint loading
│   ├── config.py          # Experiment configuration
│   ├── datasets.py        # Data loading, transforms and folds
│   ├── losses.py          # Classification and contrastive losses
│   ├── metrics.py         # QWK, MAE, F1 and ECE
│   ├── model.py           # Complete model definition
│   ├── train.py           # Training and validation
│   ├── tta.py             # Test-time analysis utilities
│   └── viz.py             # Grad-CAM and t-SNE utilities
├── scripts/
│   └── analysis/          # Result extraction scripts
├── outputs/               # Per-experiment JSON and CSV records
├── main.py                # Experiment scheduler
└── requirements.txt
```

## Installation

Python 3.10 and a CUDA-compatible PyTorch installation are recommended.

```bash
pip install -r requirements.txt
```

If a specific CUDA build is required, install `torch` and `torchvision` first
using the appropriate PyTorch command, then install the remaining dependencies.

Quick import check:

```bash
python -c "import knee_kl.model, knee_kl.losses, knee_kl.train; print('imports ok')"
```

## Dataset

Set `Config.data_root` in `knee_kl/config.py`. The expected structure is:

```text
kneeKL224/
├── train/
│   ├── 0/ ... 4/
├── val/
│   ├── 0/ ... 4/
└── test/
    ├── 0/ ... 4/
```

Images must be PNG files stored under their KL-grade directory. Filenames must
follow the pattern `7-digit patient ID + L/R + .png`, for example:

```text
9003175R.png
```

The official training and validation splits are combined into a development
set. Stratified group five-fold cross-validation keeps both knees from the same
patient in the same fold. The official test set is kept independent.

The dataset is not included in this repository. Users must obtain it from an
authorized source and comply with the original data-use agreement.

## Default Training Settings

Important settings are defined in `knee_kl/config.py`:

| Setting | Value |
|---|---:|
| Backbone | ConvNeXt-Tiny |
| Input size | 224 x 224 |
| Center crop ratio | 0.8 |
| Attention | Triplet |
| Classification loss | CDW-CE |
| Backbone learning rate | 1e-4 |
| Non-backbone learning rate | 1e-3 |
| Batch size | 128 |
| CDW exponent | 3 |
| Contrastive weight | 0.5 |
| Contrastive temperature | 0.1 |
| Projection dimension | 128 |

The formal experiment scheduler uses a maximum of 50 epochs, early-stopping
patience of 10, AdamW, and cosine learning-rate annealing.

## Running Experiments

Run commands from the repository root.

Core ablation study:

```bash
python main.py --block ablation
```

Other experiment groups:

```bash
python main.py --block loss
python main.py --block attention
python main.py --block backbone
python main.py --block sensitivity
```

Run every experiment group:

```bash
python main.py --block all
```

Run a small synthetic smoke test:

```bash
python main.py --smoke
```

Available comparisons include:

- Losses: CE, MSE, CORN, SORD, Gaussian soft labels and CDW-CE
- Attention: none, SE, CBAM, Triplet and Coordinate Attention
- Backbones: ResNet-50, DenseNet-121, EfficientNet-B0, Swin-Tiny and
  ConvNeXt-Tiny
- Sensitivity: CDW exponent, contrastive weight, ordinal exponent and learning
  rate multiplier

## Outputs

Each experiment writes:

```text
outputs/<timestamp>_<config>_summary.csv
outputs/<timestamp>_<config>_details.json
```

The JSON file contains the complete configuration, fold-level validation
metrics, seed-level test ensemble metrics, and calibration records.

Checkpoints are saved only when `Config.save_checkpoints=True`:

```text
outputs/checkpoints/<config>_seed<seed>_fold<fold>.pt
outputs/checkpoints/<config>_seed<seed>_fold<fold>.json
```

Checkpoints are not included in this repository because each model file is
approximately 111 MB. The `.pt` file and its matching `.json` sidecar must be
kept together when checkpoints are shared separately.

## Result Extraction

The following scripts read existing result files and do not start training:

```bash
python -m scripts.analysis.extract_runs
python -m scripts.analysis.extract_per_seed
```

Extracted tables are written under `outputs/extracted/`.


## Reproducibility Notes

- Python, NumPy and PyTorch random seeds are fixed.
- Splits are stratified by KL grade and grouped by patient.
- The independent test set is not used for training or early stopping.
- Test predictions are averaged across five fold models.
- `lr_head` applies to all non-backbone parameters, including attention,
  classification and projection modules.
- Per-experiment JSON and CSV records are included.
- Raw medical images, model checkpoints, caches and historical backups are
  excluded from Git.
