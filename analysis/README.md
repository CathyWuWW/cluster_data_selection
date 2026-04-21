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

When running from a network that cannot reach Hugging Face directly, use the
mirror and keep caches inside the project workspace:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/dataset-assist-0/usr/lh/wyn/data_selection/.hf_cache
export HF_DATASETS_CACHE=/home/dataset-assist-0/usr/lh/wyn/data_selection/.hf_cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0
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
  --seed 42 \
  --shuffle-buffer 1 \
  --progress-every 10000 \
  --flush-every 1000 \
  --hard-exit
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
  --seed 42 \
  --shuffle-buffer 1 \
  --progress-every 10000 \
  --flush-every 1000 \
  --hard-exit
```

## Smoke test

For a quick connectivity check on the server:

```bash
python analysis/export_slimpajama_subset.py \
  --out-dir local_data/slimpajama_smoke \
  --mode natural \
  --train-size 20 \
  --repr-size 5 \
  --val-size 5 \
  --max-scan 2000 \
  --seed 42 \
  --shuffle-buffer 1 \
  --progress-every 100 \
  --flush-every 10 \
  --hard-exit \
  --overwrite
```

If the smoke test cannot fill all quotas, increase `--max-scan` or use
`--mode natural`. On some mirrors, pyarrow/datasets can crash during Python
shutdown even after files are written correctly; `--hard-exit` skips that
shutdown path after the manifest is safely written.

## Training override

Once a subset exists, point the current training pipeline at the exported train
directory:

```bash
torchrun --nproc_per_node=8 train.py --config configs/default.yaml \
  data.train_dir=local_data/slimpajama_6b_balanced/train \
  data.dev_dir=valid \
  training.save_dir=outputs/slimpajama_balanced_smoke
```

## Nearest-Neighbor Sanity Check

After exporting a `repr/repr_eval.jsonl` subset, run the first representation
sanity check. Start with a small model and a few thousand samples before using a
large training model.

```bash
python analysis/nearest_neighbor_sanity.py \
  --input-jsonl local_data/slimpajama_6b_balanced/repr/repr_eval.jsonl \
  --out-dir analysis_outputs/nn_sanity/qwen05b_balanced \
  --model Qwen/Qwen2.5-0.5B \
  --max-samples 0 \
  --sample-strategy stratified \
  --query-count 100 \
  --top-k 10 \
  --layers middle,last \
  --poolings mean \
  --normalize-options l2 \
  --batch-size 8 \
  --max-length 512 \
  --dtype bfloat16 \
  --attn-impl sdpa \
  --remove-top-pcs 0,1,3 \
  --trust-remote-code
```

Main outputs:

```text
analysis_outputs/nn_sanity/qwen05b_natural_1k/manifest.json
analysis_outputs/nn_sanity/qwen05b_natural_1k/metrics.json
analysis_outputs/nn_sanity/qwen05b_natural_1k/samples.jsonl
analysis_outputs/nn_sanity/qwen05b_natural_1k/features/*.npy
analysis_outputs/nn_sanity/qwen05b_natural_1k/nearest_neighbors/<representation>/neighbors.jsonl
analysis_outputs/nn_sanity/qwen05b_natural_1k/nearest_neighbors/<representation>/neighbors.csv
```

The CSV/JSONL neighbor tables include query text/source/length, neighbor
text/source/length, rank, cosine distance, and `same_source`. These are the
files to inspect manually for the first sanity-check pass.

For balanced exports, avoid `--sample-strategy first` when using a small
`--max-samples`; exported files may be source-blocked, so the first 1000 records
can all come from one source. Use `--sample-strategy stratified` or set
`--max-samples 0` to read the full `repr_eval.jsonl`.
