# TMCP

Official implementation of **Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs** (WISE 2026).

TMCP addresses dynamic uncertain knowledge graph confidence prediction (DUKGCP), where newly emerging entities may be unavailable during training. It predicts the confidence of facts involving emerging entities using topology matching and structural evidence.

<p align="center">
<img src="https://github.com/user-attachments/assets/ad7411f9-0f45-48c0-a320-c9b5575637b2" alt="architecture" width="75%" />
</p>

## Requirements

We recommend creating a Python virtual environment (venv or conda) and installing the required packages.

Example (conda):

```bash
conda create -n tmcp python=3.10 -y
conda activate tmcp
pip install -r requirement.txt
```

Notes:
- The repository contains a `requirement.txt` (not `requirements.txt`). Use that file with pip.
- Core packages used in development (also listed in `requirement.txt`):
  - torch>=2.0.0
  - torchvision
  - pytorch-lightning>=2.0.0
  - numpy
  - matplotlib

If your environment does not use conda, install packages with pip manually (e.g. `pip install torch pytorch-lightning numpy matplotlib`).

## Quick setup

1. Unzip the included dataset archive (if you want to use the provided data):

```bash
unzip datasets.zip -d data_generated
```

2. Ensure each dataset folder contains the files (tab-separated):

- train.txt
- support.txt
- valid.txt
- test.txt

Each line format: `head<TAB>relation<TAB>tail[TAB>score]` (score is optional; default=1.0).

## Main components

- ETER: infers expected topology from the query head entity and relation.
- RTEA: aggregates structural evidence around the target entity.
- FCE: encodes confidence scores with Fourier features.
- GLA: captures long-range global structural information.

## Files

- `model.py` model implementation (KGReasoningModel and components).
- `dataloader.py` dataset loader (KnowledgeGraph, ReasoningDataset, collate_fn).
- `main.py` training / evaluation entry point (this repo uses `main.py` instead of `scripts/train.py`).
- `lightning.py` PyTorch Lightning wrapper module.
- `metric.py` metric utilities.
- `datasets.zip` example datasets archive (large, extract before use).
- `requirement.txt` required Python packages.

## Usage (actual commands)

Training and testing are handled by `main.py`. The script trains the model and then runs an inductive test automatically. If you interrupt training with Ctrl+C, the script will jump to the testing phase using the current weights (or the best checkpoint if available).

Example: single-run training (use a GPU if available):

```bash
python main.py --data_path data_generated/cn15k-10 --gpu 1 --epochs 100 --batch_size 256 --lr 1e-4 --save_dir checkpoints/cn15k-10
```

Quick run on CPU (small number of epochs for a smoke test):

```bash
python main.py --data_path data_generated/cn15k-10 --gpu 0 --epochs 2 --batch_size 64
```

Notes about flags (available in `main.py`):
- --data_path: path to dataset folder containing train.txt, support.txt, valid.txt, test.txt (required)
- --batch_size: batch size (default 256)
- --num_workers: DataLoader workers (default 4)
- --d_model: model hidden dim (default 128)
- --n_layers: number of GNN layers (default 2)
- --top_k: top-K evidence for aggregation (default 10)
- --epochs: number of training epochs (default 100)
- --lr: learning rate (default 1e-4)
- --gpu: number of GPUs to use (0 for CPU, default 1)
- --seed: random seed (default 42)
- --save_dir: directory to store checkpoints (default `checkpoints`)
- --drop_rate: DropEdge rate for dynamic data augmentation (default 0.4)
- --conf_mask_prob: Keep probability for confidence masking (default 0.6)
- Ablation flags: `--disable_conf`, `--disable_lre`, `--disable_sfe`, `--disable_former`

Important implementation notes:
- There is no `scripts/train.py` or `scripts/evaluate.py` in this repository; use `main.py` as shown above.
- The `--sweep` flag exists but the helper list flags (`--d_model_list`, `--n_layers_list`, `--top_k_list`) are not defined in the argument parser in `main.py` — sweep mode is therefore experimental and will error unless those flags are added.

## Running with a pretrained checkpoint / Testing-only

`main.py` will attempt to use the best checkpoint found in the `--save_dir` after training. If you want to run only testing with a saved checkpoint, you can adjust the code or run a short training then interrupt (Ctrl+C) — the script will jump to testing and use the best checkpoint if present.

If you'd like, I can add a small helper script `evaluate.py` that loads a given checkpoint and runs only the test set. Tell me the desired interface and I'll create it.

## Datasets

We evaluate TMCP on CN15k and NL27k. Example dataset folders (constructed for dynamic evaluation):

- CN15k-10, CN15k-100, CN15k-1000
- NL27k-10, NL27k-100, NL27k-1000

Each folder should contain the four files listed above.

## Citation

If you use TMCP in your work, please cite our WISE-2026 paper:

```
@inproceedings{tmcp2026,
  title={Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs},
  author={Authors, Example and Another, Person},
  booktitle={Proceedings of the Web Information Symposium (WISE)},
  year={2026}
}
```

## Contact

For questions, issues, or contributions, please open an issue or contact the maintainers.
