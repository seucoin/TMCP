# TMCP — Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs

This repository contains the official implementation of "Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs" (WISE 2026).

Abstract

TMCP addresses the problem of confidence prediction in dynamic uncertain knowledge graphs (DUKGCP), where previously unseen (emerging) entities appear at inference time and are absent during training. TMCP estimates the confidence of candidate facts by matching predicted local topologies to the observed support graph of the emerging entities and by aggregating multi-scale structural evidence.

Key contributions

- A topology-matching framework that estimates how well a candidate fact fits the expected local topology of the query entity.
- A set of modular components: ETER (expected topology estimation), RTEA (local evidence aggregation), FCE (confidence encoding with Fourier features), and GLA (global long-range aggregation).
- An evaluation protocol for dynamic uncertain KGs that simulates emerging entities with a fixed-model inference procedure.

Method overview

- ETER: Given the head entity and relation, ETER predicts an expected subgraph/topology reflecting plausible neighbor patterns.
- RTEA: Aggregates structural evidence and supports across the neighborhood of the target node to provide robust local features.
- FCE: Encodes continuous confidence values using Fourier-feature mappings to enrich representation capacity.
- GLA: Captures global and long-range structural context complementary to local evidence.

Datasets and evaluation

We evaluate TMCP on the CN15k and NL27k datasets under dynamic settings. Dataset statistics used in our experiments:

| Dataset   | #Entities  | #Relations | #Train   | #Valid | #Test  |
| --------- | ---------- | ---------- | -------- | ------ | ------ |
| CN15k     | 15,000     | 36         | 204,984  | 16,881 | 19,293 |
| NL27k     | 27,221     | 404        | 149,001  | 12,278 | 14,034 |

Dynamic evaluation: for each dataset we construct settings with N ∈ {10, 100, 1000} emerging entities:

```
CN15k-10   CN15k-100   CN15k-1000
NL27k-10   NL27k-100   NL27k-1000
```

Each experimental split contains the following files:

```
train.txt
support.txt
valid.txt
test.txt
```

Training is performed only on facts that involve old (non-emerging) entities. At inference time, the model parameters remain fixed and the support graph for emerging entities is provided to supply structural information.

Installation and requirements

We recommend using a Python virtual environment. Example using conda:

```bash
conda create -n tmcp python=3.10 -y
conda activate tmcp
pip install -r requirements.txt
```

Minimum packages used in development (specify precise versions in requirements.txt as needed):

```
torch >= 2.1.1
transformers >= 4.38.2
wandb >= 0.16.1
numpy
scipy
PyYAML
```

If a requirements.txt is not present in this repository, create one from your environment or install the packages above.

Usage

Training example (adapt configuration paths as appropriate):

```bash
python scripts/train.py --config configs/cn15k.yaml
```

Evaluation / inference example:

```bash
python scripts/evaluate.py --config configs/cn15k.yaml --mode test
```

Exposed arguments (examples — consult scripts for full set):

```
--data_path:      path to dataset folder
--config:         path to yaml/JSON config file
--batch_size:     batch size
--learning_rate:  learning rate
--num_epochs:     number of training epochs
--device:         cuda device id or cpu
--save_dir:       directory to store checkpoints
--seed:           random seed
```

Repository structure

- models/        Implementation of ETER, RTEA, FCE, GLA and model wrappers
- data/          Dataset loading and preprocessing utilities
- scripts/       Training and evaluation scripts (train.py, evaluate.py, etc.)
- configs/       Example configuration files used for experiments
- experiments/   (optional) experiment logs and saved checkpoints
- README.md      This document

(Adjust the paths above if your local tree uses different names.)

Reproducibility and experimental protocol

- The dynamic evaluation protocol fixes model weights during inference for emerging entities; only the support graph is provided for those entities.
- Reported metrics and hyperparameters should be logged alongside the experiment configuration files in `configs/` and the saved checkpoint metadata.

Citation

If you use TMCP in your research, please cite our WISE-2026 paper:

```
@inproceedings{tmcp2026,
  title={Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs},
  author={Author, A. and Author, B. and Author, C.},
  booktitle={Proceedings of the Web Information Symposium (WISE)},
  year={2026}
}
```

Contact

For questions, issues, or contributions please open an issue in this repository or contact the maintainers.

---

This README was revised to adopt a formal, paper-style presentation suitable for archiving with the project release and linking from the WISE-2026 paper. If you would like, I can also:

- Insert a formal abstract and full author list taken directly from the paper PDF.
- Add an architecture diagram with a local image file and caption.
- Produce a compact Quick Start subsection for newcomers.
