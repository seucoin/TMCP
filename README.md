# TMCP

Official implementation of **Topology Matching for Confidence Prediction in Dynamic Uncertain Knowledge Graphs** (WISE 2026).

TMCP addresses **dynamic uncertain knowledge graph confidence prediction (DUKGCP)**, where newly emerging entities may be unavailable during training. It predicts the confidence of facts involving emerging entities **without retraining** by matching expected and observed topology representations.

## Main Components

* **ETER**: infers expected topology from the query head entity and relation.
* **RTEA**: aggregates structural evidence around the target entity.
* **FCE**: encodes confidence scores with Fourier features.
* **GLA**: captures long-range global structural information.

## Datasets

We evaluate TMCP on **CN15k** and **NL27k**.

For each dataset, dynamic settings with $N\in{10,100,1000}$ emerging entities are constructed:

```text
CN15k-10   CN15k-100   CN15k-1000
NL27k-10   NL27k-100   NL27k-1000
```

The generated dataset contains:

```text
train.txt
support.txt
valid.txt
test.txt
```

The model is trained only on facts involving old entities. At inference time, the support graph provides structural information for emerging entities, while model parameters remain fixed.
