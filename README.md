# TMCP

Official implementation of Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs (WISE 2026).

TMCP addresses dynamic uncertain knowledge graph confidence prediction (DUKGCP), where newly emerging entities may be unavailable during training. It predicts the confidence of facts involving emerging entities by combining structural evidence and topology-aware reasoning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ad7411f9-0f45-48c0-a320-c9b5575637b2" alt="TMCP architecture" width="75%" />
</p>

## Overview

TMCP is a model for inductive confidence prediction in dynamic uncertain knowledge graphs. The method jointly considers:

- ETER: expected topology reasoning from the head entity and relation
- RTEA: structural evidence aggregation around the target entity
- FCE: confidence encoding via Fourier features
- GLA: long-range global structural modeling

## Environment Setup

We recommend using a Python virtual environment and installing dependencies from `requirement.txt`.

### Conda

```bash
conda create -n tmcp python=3.10 -y
conda activate tmcp
pip install -r requirement.txt
```

### Core dependencies

The repository uses the following packages:

- torch>=2.0.0
- torchvision
- pytorch-lightning>=2.0.0
- numpy
- matplotlib

> Note: The file in this repository is named `requirement.txt` (not `requirements.txt`).

---

## Dataset Preparation

This repository expects dataset folders containing:

- `train.txt`
- `support.txt`
- `valid.txt`
- `test.txt`

Each line is typically in the format:

```text
head<TAB>relation<TAB>tail<TAB>score
```

The `score` field is optional and defaults to `1.0` when absent.

The included archive `datasets.zip` can be extracted as follows:

```bash
unzip datasets.zip -d data_generated
```

Example dataset names include:

- `CN15k-10`
- `CN15k-100`
- `CN15k-1000`
- `NL27k-10`
- `NL27k-100`
- `NL27k-1000`

---

## Quick Start

This repository uses `main.py` as the actual training entry point.

### Train a model

```bash
python main.py \
  --data_path data_generated/cn15k-10 \
  --gpu 1 \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-4 \
  --save_dir checkpoints/cn15k-10
```

### CPU smoke test

```bash
python main.py \
  --data_path data_generated/cn15k-10 \
  --gpu 0 \
  --epochs 2 \
  --batch_size 64
```

---

## Important CLI Arguments

The main script supports the following key arguments:

```bash
--data_path      Path to the dataset folder (required)
--batch_size     Batch size (default: 256)
--num_workers    Number of DataLoader workers (default: 4)
--d_model        Hidden dimension (default: 128)
--n_layers       Number of model layers (default: 2)
--top_k          Top-k aggregation value (default: 10)
--epochs         Training epochs (default: 100)
--lr             Learning rate (default: 1e-4)
--gpu            Number of GPUs to use (set to 0 for CPU)
--seed           Random seed (default: 42)
--save_dir       Directory for checkpoints
--drop_rate      DropEdge rate for augmentation (default: 0.4)
--conf_mask_prob Confidence masking probability (default: 0.6)
--disable_conf   Disable confidence encoder
--disable_lre    Disable logic reasoning encoder
--disable_sfe    Disable structural feature encoder
--disable_former Disable global linear attention
```

---

## Repository Structure

```text
.
├── main.py             # training / evaluation entry point
├─��� dataloader.py       # dataset loading and graph construction
├── model.py            # TMCP model implementation
├── lightning.py        # PyTorch Lightning wrapper
├── metric.py           # evaluation metrics
├── requirement.txt     # dependencies
├── datasets.zip        # example datasets
├── README.md
└── ...
```

---

## Notes

- There is no `scripts/train.py` or `scripts/evaluate.py` in this repository; the actual training pipeline is implemented in `main.py`.
- The script can continue into testing automatically after training, and it will use the best checkpoint if available.
- The `--sweep` option exists, but the sweep-related list args are not defined in the current parser and therefore require additional implementation if you want to use sweep mode.

---

## Citation

If you use TMCP in your work, please cite our WISE-2026 paper:

```bibtex
@inproceedings{tmcp2026,
  title={Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs},
  author={Authors, Example and Another, Person},
  booktitle={Proceedings of the Web Information Symposium (WISE)},
  year={2026}
}
```

---

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the maintainers.
