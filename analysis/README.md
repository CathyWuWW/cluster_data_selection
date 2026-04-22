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

By default, `--out-dir` is treated as an experiment root. The script creates a
parameterized run subdirectory such as:

```text
analysis_outputs/nn_sanity/qwen05b_balanced/20260421_153012_qwen2.5-0.5b_n1000_stratified_q100_k10_len512_layers-middle-last_pool-mean_norm-l2_pc-0-1-3_seed42/
```

Use `--run-name <name>` for a shorter explicit run directory, or
`--flat-output` to write directly into `--out-dir`.

The experiment root also gets an updated `runs_summary.csv`; this is the
cross-run comparison table to use when comparing settings like `n1000` versus
`nall`, random versus stratified sampling, or different PC-removal choices.

Main outputs:

```text
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/manifest.json
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/metrics.json
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/summary.csv
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/samples.jsonl
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/features/*.npy
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/nearest_neighbors/<representation>/neighbors.jsonl
analysis_outputs/nn_sanity/<experiment_root>/<run_name>/nearest_neighbors/<representation>/neighbors.csv
analysis_outputs/nn_sanity/<experiment_root>/runs_summary.csv
```

The CSV/JSONL neighbor tables include query text/source/length, neighbor
text/source/length, rank, cosine distance, and `same_source`. These are the
files to inspect manually for the first sanity-check pass.

Use `summary.csv` for quick horizontal comparison across representations. It
includes same-source neighbor rate, lift versus a random same-source baseline,
norm-length correlation, and top-PC variance ratios.

For balanced exports, avoid `--sample-strategy first` when using a small
`--max-samples`; exported files may be source-blocked, so the first 1000 records
can all come from one source. Use `--sample-strategy stratified` or set
`--max-samples 0` to read the full `repr_eval.jsonl`.

## Representation Comparison

The second validation layer compares hidden-state representations with
sentence-embedding, lexical, random, length, and source baselines. Keep this as
an independent analysis script before wiring anything into the training loop.

Recommended balanced run:

```bash
python analysis/representation_comparison.py \
  --input-jsonl local_data/slimpajama_6b_balanced/repr/repr_eval.jsonl \
  --out-dir analysis_outputs/representation_comparison/qwen05b_balanced \
  --hidden-model Qwen/Qwen2.5-0.5B \
  --max-samples 1000 \
  --sample-strategy stratified \
  --query-count 200 \
  --top-k 10 \
  --hidden-layers "first,1,2,3,middle,-4,-3,-2,last" \
  --hidden-layer-groups "early=first,1,2,3;middle=middle;late=-4,-3,-2;final=last" \
  --poolings mean,last \
  --normalize-options l2 \
  --remove-top-pcs 0,1,3 \
  --include-whitening \
  --sentence-models intfloat/e5-small-v2,BAAI/bge-small-en-v1.5 \
  --include-random \
  --include-lexical \
  --include-length-source \
  --kmeans-k 10,20,50 \
  --batch-size 8 \
  --max-length 512 \
  --dtype bfloat16 \
  --attn-impl sdpa \
  --trust-remote-code
```

Main outputs:

```text
analysis_outputs/representation_comparison/qwen05b_balanced/<run_name>/manifest.json
analysis_outputs/representation_comparison/qwen05b_balanced/<run_name>/samples.jsonl
analysis_outputs/representation_comparison/qwen05b_balanced/<run_name>/representation_summary.csv
analysis_outputs/representation_comparison/qwen05b_balanced/<run_name>/clustering_summary.csv
```

`representation_summary.csv` compares top-k source-neighbor rate, lift over
random source matching, length correlation, and PC variance ratios.
`clustering_summary.csv` compares MiniBatchKMeans inertia, cosine silhouette,
cluster size distribution, and source purity/entropy.

Plot the CSV summaries:

```bash
python analysis/plot_representation_comparison.py \
  --run-dir analysis_outputs/representation_comparison/qwen05b_balanced/<run_name>
```

Figures are written to:

```text
analysis_outputs/representation_comparison/qwen05b_balanced/<run_name>/figures/
```

Use these plots to judge whether hidden mean representations are close to
sentence embedding baselines, merely better than random/lexical/source
baselines, or mainly organized by source/length/template artifacts.

## Semantic Reference Compatibility

This validation uses E5/BGE as semantic reference spaces rather than
ground-truth labels. It asks whether hidden representations preserve the
nearest-neighbor sets and similarity rankings induced by sentence embedding
models. Run both all-candidate and cross-source candidate metrics; the
cross-source setting is important because earlier checks show strong source
structure.

```bash
python analysis/semantic_reference_eval.py \
  --input-jsonl local_data/slimpajama_6b_balanced/repr/repr_eval.jsonl \
  --out-dir analysis_outputs/semantic_reference/qwen05b_balanced \
  --hidden-model Qwen/Qwen2.5-0.5B \
  --max-samples 1000 \
  --sample-strategy stratified \
  --query-count 200 \
  --candidate-count 500 \
  --top-k 10,50 \
  --hidden-representations "layer12_mean_l2:middle:mean:l2:0;layer12_mean_l2_rm1pc:middle:mean:l2:1;layer23_mean_l2:last:mean:l2:0;layer23_mean_l2_rm1pc:last:mean:l2:1" \
  --reference-models intfloat/e5-small-v2,BAAI/bge-small-en-v1.5 \
  --include-tfidf \
  --include-random \
  --batch-size 8 \
  --max-length 512 \
  --dtype bfloat16 \
  --attn-impl sdpa \
  --trust-remote-code
```

Main output:

```text
analysis_outputs/semantic_reference/qwen05b_balanced/<run_name>/semantic_reference_metrics.csv
```

Key columns:

```text
overlap_at_10 / overlap_at_50
jaccard_at_10 / jaccard_at_50
spearman_mean
mode = all or cross_source
```

Plot the results:

```bash
python analysis/plot_semantic_reference_eval.py \
  --run-dir analysis_outputs/semantic_reference/qwen05b_balanced/<run_name> \
  --top-k 10
```

Interpretation:

- Hidden close to E5/BGE and above TF-IDF/random, especially in
  `cross_source`, supports semantic compatibility.
- Hidden close only in `all` but much weaker in `cross_source` suggests source
  or format shortcuts.
- Hidden far below E5/BGE but above random/TF-IDF means it may be useful for
  domain structure, but should not be called a semantic representation.
