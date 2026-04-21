# Analysis Data Preparation

This folder is for reproducible experiment helpers that do not change the
training pipeline.

## Goal

Use SlimPajama-6B as an open replacement for the private `dataset-100k` data.
The exporter writes JSONL files with a `text` field, so the existing
`JsonFolderDataset` can read them without code changes. It also preserves
`source` and `meta` fields for representation and cluster-quality diagnostics.

## Server setup

Install the extra data-loading dependency:

```bash
pip install datasets pyarrow
```

## Balanced subset

Balanced sampling gives each RedPajama source roughly the same number of
examples. This is useful for representation validation because source labels
are less dominated by CommonCrawl/C4.

```bash
python analysis/export_slimpajama_subset.py \
  --out-dir local_data/slimpajama_6b_balanced \
  --mode stratified \
  --train-size 100000 \
  --repr-size 10000 \
  --val-size 5000 \
  --seed 42
```

Outputs:

```text
local_data/slimpajama_6b_balanced/train/train.jsonl
local_data/slimpajama_6b_balanced/repr/repr_eval.jsonl
local_data/slimpajama_6b_balanced/valid/slimpajama_valid.jsonl
local_data/slimpajama_6b_balanced/manifest.json
```

## Natural subset

Natural sampling keeps the shuffled stream distribution. This is closer to the
training setting after the representation checks pass.

```bash
python analysis/export_slimpajama_subset.py \
  --out-dir local_data/slimpajama_6b_natural \
  --mode natural \
  --train-size 100000 \
  --repr-size 10000 \
  --val-size 5000 \
  --seed 42
```

## Smoke test

For a quick connectivity check on the server:

```bash
python analysis/export_slimpajama_subset.py \
  --out-dir local_data/slimpajama_smoke \
  --mode stratified \
  --train-size 70 \
  --repr-size 14 \
  --val-size 14 \
  --max-scan 50000 \
  --seed 42
```

If the smoke test cannot fill all quotas, increase `--max-scan` or use
`--mode natural`.

## Training override

Once a subset exists, point the current training pipeline at the exported train
directory:

```bash
torchrun --nproc_per_node=8 train.py --config configs/default.yaml \
  data.train_dir=local_data/slimpajama_6b_balanced/train \
  data.dev_dir=valid \
  training.save_dir=outputs/slimpajama_balanced_smoke
```
