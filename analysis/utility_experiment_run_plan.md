# Utility Experiments Run Plan

## Goal

Move the project from "is hidden mean similar to sentence embedding" to:

- whether hidden-based clusters encode model-relevant utility structure
- whether clusters with similar utility vectors can be regrouped into stronger meta-clusters for data selection

This plan assumes `l2_rm1pc` hidden clustering is the default base representation.

## Outputs We Need

Each run should produce:

- `cluster_ids_initial.npy` or reused fixed ids
- `cluster_assignments_initial.json`
- `cluster_weight_history.jsonl`
- eval logs with per-domain metrics
- final checkpoint or final metrics summary

The core artifact for Experiment 2 is a cluster-level utility matrix:

`U[k, a] = utility of cluster k for ability a`

where ability `a` is one of `math / logic / reading / code / ...`.

## One-Time Setup

### 1. Freeze a base clustering

Run clustering once on the chosen train snapshot and save:

- `cluster_ids_initial.npy`
- `cluster_assignments_initial.json`

Recommended base:

- representation: `intermediate`
- normalization: `l2`
- `remove_top_pcs=1`
- cluster size: `1000`

After this point, all Experiment 2 runs should set:

- `clustering.precomputed_ids_path=<saved cluster_ids_initial.npy>`
- `clustering.recluster_interval=-1`

Do not recluster during utility-vector collection, otherwise cluster indices are no longer aligned across abilities.

Command template for a one-time hidden-base freeze:

```bash
python train.py --config configs/default.yaml \
  model.path=Qwen/Qwen2.5-0.5B \
  model.attn_impl=sdpa \
  model.max_length=256 \
  model.gradient_checkpointing=false \
  data.train_dir=local_data/slimpajama_6b_balanced/train \
  data.dev_dir=data/dev/math \
  data.text_field=text \
  data.dev_num=32 \
  data.dev_seed=42 \
  data.eval_format=text \
  training.total_iters=0 \
  training.no_eval_at_start=true \
  training.save_interval=999999 \
  training.save_dir=outputs/base_hidden_l2rm1pc \
  clustering.method=minibatch \
  clustering.cluster_size=1000 \
  clustering.kmeans.feature=intermediate \
  clustering.kmeans.embed_layer=-1 \
  clustering.kmeans.normalize=l2 \
  clustering.kmeans.remove_top_pcs=1 \
  clustering.kmeans.feature_batch_size=32 \
  clustering.embedding_model.enabled=true \
  clustering.embedding_model.path=Qwen/Qwen2.5-0.5B \
  clustering.embedding_model.dtype=bfloat16 \
  clustering.embedding_model.attn_impl=sdpa \
  pmp.drop_bad_clusters=false \
  deepspeed.enabled=false
```

Expected fixed-cluster artifacts:

```bash
outputs/base_hidden_l2rm1pc/cluster_ids_initial.npy
outputs/base_hidden_l2rm1pc/cluster_assignments_initial.json
```

### 2. Prepare dev domains

Create separate dev directories for each ability:

- `data/dev/math`
- `data/dev/logic`
- `data/dev/reading`
- `data/dev/code`

Each directory should be fixed once and reused across all runs.

Recommended first wave:

- `math`
- `reading`
- `code`

Add `logic` in the second wave if the first three runs are stable.

### 3. Keep one embedding baseline

Do not run a large baseline grid.
Keep one embedding-cluster baseline for comparison:

- `bge` or `e5`

Use the same train snapshot, same dev domains, and same training budget.

Optional one-time embedding-base freeze:

```bash
python train.py --config configs/default.yaml \
  model.path=Qwen/Qwen2.5-0.5B \
  model.attn_impl=sdpa \
  model.max_length=256 \
  model.gradient_checkpointing=false \
  data.train_dir=local_data/slimpajama_6b_balanced/train \
  data.dev_dir=data/dev/math \
  data.text_field=text \
  data.dev_num=32 \
  data.dev_seed=42 \
  data.eval_format=text \
  training.total_iters=0 \
  training.no_eval_at_start=true \
  training.save_interval=999999 \
  training.save_dir=outputs/base_embed_e5 \
  clustering.method=minibatch \
  clustering.cluster_size=1000 \
  clustering.kmeans.feature=embedding \
  clustering.kmeans.normalize=l2 \
  clustering.kmeans.remove_top_pcs=0 \
  clustering.kmeans.feature_batch_size=32 \
  clustering.embedding_model.enabled=true \
  clustering.embedding_model.path=intfloat/e5-small-v2 \
  clustering.embedding_model.dtype=float16 \
  clustering.embedding_model.attn_impl=sdpa \
  pmp.drop_bad_clusters=false \
  deepspeed.enabled=false
```

## Experiment 2: Utility Vector

### Design

For each ability, run the same training pipeline but change the PMP objective so it is driven by only that domain.

That means one run per ability:

- `math-only`
- `reading-only`
- `code-only`

Each run uses:

- same train data
- same fixed cluster ids
- same model
- same training budget
- same optimizer and PMP settings

Only the dev target differs.

### Config rule

Use `data.dev_domains` with exactly one active domain per run, for example:

```yaml
data:
  dev_domains:
    - name: "math"
      dir: "data/dev/math"
      weight: 1.0
```

This lets PMP compute a single-domain utility signal while keeping the pipeline unchanged.

### Recommended training budget

Use a two-stage budget:

1. Smoke test
   - `total_iters=50`
   - `eval_interval=25`
   - verify logs, output files, and weight updates
2. Formal utility run
   - `total_iters=300` for the first full pass
   - increase to `500-1000` only if utility rankings are too noisy

### Recommended batched launch command

Run all hidden-cluster utility jobs in one shot:

```bash
PRECOMPUTED_CLUSTER_IDS=outputs/base_hidden_l2rm1pc/cluster_ids_initial.npy \
CLUSTER_LABEL=hidden_l2rm1pc \
DEV_ROOT=data/dev \
ABILITIES=math,reading,code \
NPROC_PER_NODE=4 \
MASTER_PORT=29501 \
MODEL_PATH=Qwen/Qwen2.5-0.5B \
TRAIN_DIR=local_data/slimpajama_6b_balanced/train \
TOTAL_ITERS=300 \
EVAL_INTERVAL=100 \
PMP_UPDATE_INTERVAL=20 \
DEV_NUM=256 \
DEV_SEED=42 \
MAX_LENGTH=256 \
TRAIN_BATCH_SIZE=1 \
GRAD_ACCUM=1 \
DEV_BATCH_SIZE=2 \
EVAL_BATCH_SIZE=16 \
LR=1e-5 \
WARMUP_ITERS=20 \
SEED=42 \
bash scripts/run_utility_vector_experiments.sh
```

This will create:

```bash
outputs/utility_vector_hidden_l2rm1pc_<timestamp>/math
outputs/utility_vector_hidden_l2rm1pc_<timestamp>/reading
outputs/utility_vector_hidden_l2rm1pc_<timestamp>/code
```

### Example command template

```bash
torchrun --nproc_per_node=4 train.py --config configs/default.yaml \
  model.path=Qwen/Qwen2.5-0.5B \
  model.attn_impl=sdpa \
  model.max_length=256 \
  model.gradient_checkpointing=false \
  data.train_dir=local_data/slimpajama_6b_balanced/train \
  data.text_field=text \
  data.dev_domains="[{name: math, dir: data/dev/math, weight: 1.0}]" \
  data.eval_format=text \
  data.dev_num=256 \
  training.total_iters=300 \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.eval_batch_size=16 \
  training.eval_interval=100 \
  training.save_interval=999999 \
  training.lr=1e-5 \
  training.warmup_iters=20 \
  training.save_dir=outputs/utility_math_hidden \
  clustering.precomputed_ids_path=outputs/base_hidden/cluster_ids_initial.npy \
  clustering.recluster_interval=-1 \
  pmp.update_interval=20 \
  pmp.dev_batch_size=2 \
  pmp.drop_bad_clusters=false \
  deepspeed.enabled=false
```

Change only:

- `data.dev_domains`
- `training.save_dir`

for each ability run.

Equivalent single-run command for one ability:

```bash
python train.py --config configs/default.yaml \
  model.path=Qwen/Qwen2.5-0.5B \
  model.attn_impl=sdpa \
  model.max_length=256 \
  model.gradient_checkpointing=false \
  data.train_dir=local_data/slimpajama_6b_balanced/train \
  data.dev_dir=data/dev/math \
  data.dev_domains=[] \
  data.text_field=text \
  data.dev_num=256 \
  data.dev_seed=42 \
  data.eval_format=text \
  data.num_workers=0 \
  training.total_iters=300 \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.eval_batch_size=16 \
  training.eval_interval=100 \
  training.save_interval=999999 \
  training.no_eval_at_start=false \
  training.save_dir=outputs/utility_math_hidden \
  training.lr=1e-5 \
  training.warmup_iters=20 \
  training.seed=42 \
  clustering.precomputed_ids_path=outputs/base_hidden_l2rm1pc/cluster_ids_initial.npy \
  clustering.recluster_interval=-1 \
  pmp.update_interval=20 \
  pmp.dev_batch_size=2 \
  pmp.drop_bad_clusters=false \
  deepspeed.enabled=false
```

### How to derive utility vectors

For each run, extract from `cluster_weight_history.jsonl`:

- final `grad_gamma`
- cumulative `grad_gamma_delta`
- final sampler weights

Recommended primary utility statistic:

- `u_a(k) = sum_t grad_gamma_delta_t[k]`

Recommended secondary statistics:

- final weight rank of cluster `k`
- average weight over the last 20 percent of updates
- sign consistency across updates

The final utility vector is:

`u(k) = [u_math(k), u_reading(k), u_code(k), ...]`

Command to build the utility matrix after the runs finish:

```bash
python analysis/build_utility_matrix.py \
  --runs-root outputs/utility_vector_hidden_l2rm1pc_<timestamp> \
  --out-dir outputs/utility_vector_hidden_l2rm1pc_<timestamp>/utility_matrix \
  --abilities math,reading,code
```

Main outputs:

```bash
utility_matrix/utility_matrix_sum_delta.csv
utility_matrix/utility_matrix_final_weight.csv
utility_matrix/utility_matrix_avg_tail_weight.csv
utility_matrix/utility_matrix_positive_fraction.csv
utility_matrix/utility_long.csv
utility_matrix/cluster_summary.csv
```

### Stability checks

Before trusting the matrix:

- rerun one ability with a second seed
- compare top-k utility clusters across seeds
- check whether high-utility clusters remain high-utility

If rankings are unstable, increase training budget before moving to Experiment 3.

## Embedding Baseline for Experiment 2

Run the same per-ability protocol once with an embedding-based clustering.

Keep everything else fixed:

- same train data
- same dev domains
- same training budget
- same cluster size

Then compare:

- utility-vector separability
- cross-seed stability
- downstream dev gains

This is the key comparison supporting the claim that hidden clustering is closer to model utility space than standard semantic embeddings.

Embedding utility-run command:

```bash
PRECOMPUTED_CLUSTER_IDS=outputs/base_embed_e5/cluster_ids_initial.npy \
CLUSTER_LABEL=embed_e5 \
DEV_ROOT=data/dev \
ABILITIES=math,reading,code \
NPROC_PER_NODE=4 \
MASTER_PORT=29511 \
MODEL_PATH=Qwen/Qwen2.5-0.5B \
TRAIN_DIR=local_data/slimpajama_6b_balanced/train \
TOTAL_ITERS=300 \
EVAL_INTERVAL=100 \
PMP_UPDATE_INTERVAL=20 \
DEV_NUM=256 \
DEV_SEED=42 \
MAX_LENGTH=256 \
TRAIN_BATCH_SIZE=1 \
GRAD_ACCUM=1 \
DEV_BATCH_SIZE=2 \
EVAL_BATCH_SIZE=16 \
LR=1e-5 \
WARMUP_ITERS=20 \
SEED=42 \
bash scripts/run_utility_vector_experiments.sh
```

## Experiment 3: Utility-Driven Meta-Clustering

### Input

Use the utility matrix from Experiment 2:

- rows: base clusters
- columns: abilities

Normalize each ability dimension before clustering, for example with z-score normalization.

### Step 1. Cluster the clusters

Run a second clustering on utility vectors:

- algorithm: KMeans or MiniBatchKMeans
- input objects: original clusters
- features: utility vectors

Start with a small number of meta-clusters, such as:

- `M in {8, 16, 32}`

### Step 2. Build regrouped assignments

Map each sample:

- sample -> base cluster
- base cluster -> meta-cluster
- therefore sample -> meta-cluster

Save:

- `meta_cluster_ids.npy`
- `meta_cluster_assignments.json`

### Step 3. Run the same training pipeline on meta-clusters

Use:

- `clustering.precomputed_ids_path=<meta_cluster_ids.npy>`

and compare against:

- base hidden clusters
- embedding clusters
- random regroup of base clusters into the same number of meta-clusters

### Core baselines for Experiment 3

Keep only these:

- `hidden base cluster`
- `hidden utility meta-cluster`
- `hidden random meta-cluster`
- `embedding base cluster`

This is enough to isolate whether the gain comes from utility-aware regrouping rather than just a second clustering step.

## Decision Rules

Move from Experiment 2 to Experiment 3 only if at least one of these is true:

- utility vectors show visible ability specialization
- hidden clusters are more stable than embedding clusters in utility space
- hidden clusters yield better ability-specific downstream gains

If none of these holds, do not invest in meta-clustering yet.

## Suggested Execution Order

1. Freeze one hidden base clustering.
2. Run `math / reading / code` utility runs on that fixed clustering.
3. Run one embedding baseline with the same three ability runs.
4. Build the hidden utility matrix:

```bash
python analysis/build_utility_matrix.py \
  --runs-root outputs/utility_vector_hidden_l2rm1pc_<timestamp> \
  --out-dir outputs/utility_vector_hidden_l2rm1pc_<timestamp>/utility_matrix \
  --abilities math,reading,code
```

5. Build the embedding utility matrix:

```bash
python analysis/build_utility_matrix.py \
  --runs-root outputs/utility_vector_embed_e5_<timestamp> \
  --out-dir outputs/utility_vector_embed_e5_<timestamp>/utility_matrix \
  --abilities math,reading,code
```

6. Inspect separability and stability.
7. If structure exists, generate meta-clusters.
8. Run the four Experiment 3 comparisons.
