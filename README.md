# TMCP

Official implementation of **Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs** (WISE 2026).

TMCP addresses dynamic uncertain knowledge graph confidence prediction (DUKGCP), where newly emerging entities may be unavailable during training. It predicts the confidence of facts involving emerging entities by matching expected topologies and aggregating structural evidence.

<p align="center">
<img src="https://github.com/user-attachments/assets/ad7411f9-0f45-48c0-a320-c9b5575637b2" alt="architecture" width="75%" />
</p>

## Requirements

We recommend creating a Python virtual environment (venv or conda) and installing the required packages.

Example (conda):

```bash
conda create -n tmcp python=3.10 -y
conda activate tmcp
pip install -r requirements.txt
```

Minimum packages used in development:

```
torch >= 2.1.1
transformers >= 4.38.2
wandb >= 0.16.1
numpy
scipy
```

(If this repository does not include a requirements.txt, please run pip install for the packages above or create one from your environment.)

## Main Components

- ETER: infers expected topology from the query head entity and relation.
- RTEA: aggregates structural evidence around the target entity.
- FCE: encodes confidence scores with Fourier features.
- GLA: captures long-range global structural information.

## Datasets

We evaluate TMCP on CN15k and NL27k.

| Dataset   | #Entities  | #Relations | #Train   | #Valid | #Test  |
| --------- | ---------- | ---------- | -------- | ------ | ------ |
| CN15k     | 15,000     | 36         | 204,984  | 16,881 | 19,293 |
| NL27k     | 27,221     | 404        | 149,001  | 12,278 | 14,034 |

For dynamic evaluation we construct settings with N ∈ {10, 100, 1000} emerging entities:

```
CN15k-10   CN15k-100   CN15k-1000
NL27k-10   NL27k-100   NL27k-1000
```

Each generated dataset contains the following files:

```
train.txt
support.txt
valid.txt
test.txt
```

The model is trained only on facts involving old entities. At inference time, the support graph provides structural information for emerging entities while model parameters remain fixed.

## Files

- `models/` contains model implementations for ETER, RTEA, FCE and GLA components.
- `data/` contains dataset loading and preprocessing utilities.
- `scripts/` contains training and evaluation scripts.
- `configs/` contains example configuration files.
- `README.md` this file.

(Adjust paths above if your local tree uses different names.)

## Usage

Training example (replace config and dataset paths as needed):

```bash
python scripts/train.py --config configs/cn15k.yaml
```

Evaluate or predict confidence scores:

```bash
python scripts/evaluate.py --config configs/cn15k.yaml --mode test
```

If the repository provides other entry points, consult the `scripts/` folder and adapt commands accordingly.

## Argument Descriptions

Common arguments used by training/evaluation scripts:

```bash
--data_path:      path to dataset folder
--config:         path to yaml/JSON config file
--batch_size:     batch size
--learning_rate:  learning rate
--num_epochs:     number of training epochs
--device:         cuda device id or cpu
--save_dir:       directory to store checkpoints
--seed:           random seed
```

Configs are set in the `configs/` files; modify them when necessary.

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
