# Joint Clustering Loss — Level 1 实现设计

> 本文档是 `docs/joint_loss_level1_idea.md` 的具体实现方案，锁定本轮实验的所有
> 设计决策，后续代码与脚本都以本文为单一事实源。

## 1. 背景与目标（一页版）

- **任务性质**（AGENTS.md）：小规模 CPT 数据选择研究。训练规模 ~0.15M tokens /
  600 步；LM dev loss 是唯一有意义的评估指标，task-level accuracy 在本规模
  下不用作主判据。
- **Level 1 目标**：在现有 1.5B Qwen2.5 + 24 meta-cluster + math+code dev
  pilot setting 上，于标准 LM loss 之外引入一个 **cluster-pull 正则项**，把
  每个样本的 hidden representation 拉向其所属 cluster 的可学 prototype
  μ_k，从而把"功能性"注入到 representation 空间；不引入在线重分簇、不引入
  per-sample gradient、不引入 Sinkhorn 等防 collapse 技巧。
- **成功判据**（二者均需出现）：
  1. math/code dev loss 相对最强 baseline（`hidden_utility_meta_s42`）在多个
     checkpoint 上一致出现可读负 Δ（≥ 0.005 量级）；
  2. 训练结束后 meta-cluster 内部 hidden state 到 μ 的平均距离显著下降，
     且 μ 两两 pairwise cosine 未塌陷。
- **失败分支**：上面任何一项见不到信号 → 说明光靠"拉 representation"不够，
  跳过 Level 2，直接设计 Level 3 的 gradient-level utility 方案。

## 2. 决策记录（2026-05-06，已与用户对齐）

| 决策 | 选项 | 理由 |
|---|---|---|
| μ_k 初始化 | **训练模型自身 layer L\* 的 per-cluster hidden-mean** | 保证 μ 与 h 在同一坐标系，pull loss 从合理位置开始 |
| utility 注入通道 | **作为 per-cluster pull 权重 w_k**，不进入 μ_k 初始化 | μ 严格保持在 hidden 空间；功能性信号收束到 w_k 这一个通道 |
| cluster 粒度 | **24 meta-cluster**（= `hidden_utility_meta_s42` 所用） | 与当前最强 baseline 做直接对照；24 个 prototype 稀疏性好 |
| 本轮跑哪几组 | **J0 (λ=0 sanity) + J1 (λ=0.05) + J2 (λ=0.2) + J3 (λ=0.2 + utility weight)** | 三卡并行 ~2 轮；先看方向信号再决定是否扩 |

## 3. 损失函数

batch 中样本 i 属于 meta-cluster k_i ∈ {0,…,M−1}（M=24）。总损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda \cdot \mathcal{L}_{\text{pull}}
$$

$$
\mathcal{L}_{\text{pull}} = \frac{1}{B}\sum_{i=1}^{B} w_{k_i}\cdot \operatorname{dist}\bigl(h(x_i), \mu_{k_i}\bigr)
$$

- `h(x_i) ∈ R^H`：样本在 **layer L\*** 的 hidden states 经 **masked mean-pool**
  得到（mask 来自 `model_batch["attention_mask"]`，pad 位不参与平均）。
- `μ_k ∈ R^H`：`nn.Parameter`，形状 `[M, H]`。
- `dist`：默认 `cosine` → `1 − cos(h, μ)`；备选 `squared_l2 = ‖h − μ‖² / H`。
  优先 cosine：对 hidden norm 尺度无关，λ 的有效区间经验上比较稳。
- `w_k ≥ 0`：per-cluster 权重，两种取值
  - `uniform`: 全 1（J0/J1/J2）；
  - `utility`: 由 `utility_matrix_mc_sum_delta.csv`（math + code）经
    `analysis/utility_prior.load_utility_prior` 的 aggregator / normalize /
    sign 管线产出；最终对整体做 `w ← w / mean(w)` 使 `E[w_k] = 1`，以保持
    pull loss 平均强度与 uniform 方案可比。

- λ：全局强度，本轮只扫 {0, 0.05, 0.2}，J3 固定 λ=0.2。

## 4. 注入点（最小侵入）

当前 `_compute_lm_loss` 改写为 `_compute_joint_loss`，行为在
`joint_loss.enabled = false` 时与原实现**逐行等价**。伪代码：

```python
# forward (single call, no extra forward pass)
outputs = self.model(**model_batch, use_cache=False,
                     output_hidden_states=True)      # 新增 output_hidden_states
logits = outputs.logits
hidden_all = outputs.hidden_states                   # tuple 长度 = num_layers + 1
h_layer = hidden_all[self.joint_cfg.layer_idx + 1]   # +1 因为 index 0 是 embedding 输出

# LM loss: 与原逻辑完全一致
lm_loss = <原有 CE + loss_mask 的 mean>

if not self.joint_cfg.enabled:
    return lm_loss, {"lm_loss": lm_loss.detach(), "pull_loss": None}

# masked mean-pool
mask = model_batch["attention_mask"].unsqueeze(-1).float()
pooled = (h_layer * mask).sum(1) / mask.sum(1).clamp(min=1)    # [B, H]

mu = self.cluster_prototypes(cluster_ids_batch)                 # [B, H]
pull = _pairwise_dist(pooled, mu, self.joint_cfg.distance)      # [B]
w = self.joint_cluster_weights[cluster_ids_batch]               # [B]
pull_loss = (pull * w).mean()

total = lm_loss + self.joint_cfg.lam * pull_loss
return total, {"lm_loss": lm_loss.detach(),
               "pull_loss": pull_loss.detach()}
```

**两处关键重构**：
1. 当前代码在 forward **之后**才解析 `cluster_ids_batch`。必须把"从
   `__indices__` 查 `self.train_dataset.cluster_ids`"的动作移到 forward
   **之前**（直接放在 `__indices__` pop 之后）；这对基线路径零影响，但 joint
   路径必须用它。
2. `self.model(**model_batch, ...)` 传入 `output_hidden_states=True`。这会略
   增 memory（需要保留所有 layer 的 hidden），但因为只有 1 次 forward，不会
   有算力翻倍。为避免保存过多层，后续可以在 `_raw_model.config.output_hidden_states`
   只在 joint 分支临时打开，但一来 1.5B/1024 场景显存预算足够，二来会使代
   码更脆，本轮不做这个微优化。

## 5. Prototype 模块（新文件 `trainer/cluster_prototypes.py`）

```python
class ClusterPrototypes(nn.Module):
    """K × H learnable prototypes。

    初始化策略:
      - feature_mean:   外部传入 [K, H] 张量 (hidden-mean)，直接作为 μ^{(0)}。
      - random:         N(0, σ=1/sqrt(H))
      - zero:           全零（仅供单元测试或 sanity, 不推荐用于实验）

    其它:
      - distance: 对外提供 pairwise_distance(h, ids)，实现 cosine / squared_l2。
      - pairwise_metrics(): 返回 μ 的 frobenius norm、pairwise cosine 的
        {min,max,mean}、相邻两次调用的 ‖Δμ‖，用于 prototype_history.jsonl。
    """
```

**初始化来源**：`extract_prototype_init(raw_model, train_dataset, cluster_ids,
layer_idx, batch_size, device)`：

- 对所有训练样本按 `cluster_ids` 分组；
- 每组走一次 `model.forward(..., output_hidden_states=True)` → 取 layer L\*
  → masked mean-pool → 累积到 `sum_k, count_k`（防 OOM 走 mini-batch 累加，
  不一次性 stack 所有 hidden）；
- 返回 `[M, H]` 的 μ 初始值 + `M` 的样本计数，供日志写 `prototype_init.npy`
  和 `prototype_init_meta.json`。
- 用 `bf16` 前向但以 `fp32` 累加，避免数值漂移。
- 对空簇（理论上不会出现，但防御性处理）用全 0 + warning。

## 6. Optimizer 设计

- 主模型 param group 不变。
- 新增 prototype param group:
  - `lr = training.lr * joint_loss.proto_lr_multiplier`（默认 10×）；
  - `weight_decay = 0`；
  - betas / eps 与主模型一致。
- lr scheduler：prototype 组与主模型走同一个 scheduler（因为 `_build_lr_scheduler`
  是绑在 optimizer 上的，对所有 group 生效），这是可接受的：warmup 阶段
  prototype lr 也是 warmup 后的，不会一上来就把 μ 拉飞。
- DeepSpeed 分支：本轮只在 DDP（`deepspeed.enabled=false`）下跑（与 pilot
  一致）；DeepSpeed 分支临时要求 `joint_loss.enabled = false`，并在 trainer
  init 里断言。Level 2 以后再补 DS 支持。

## 7. 层选择与距离函数

- 默认 `layer_idx = 14`（Qwen2.5-1.5B 共 28 层，中层；与现有 base clustering
  的 `embed_layer=-1` 口径一致）。
- 默认 `distance = cosine`。本轮 J0..J3 都用 cosine；L2 作为后续 ablation。

## 8. 实验表（本轮跑的 4 个 run）

共享项：Qwen2.5-1.5B，max_length=1024，batch=1，gacc=4（≈ 4K tokens/step），
1000 steps，dev = math + code（each weight=1），seed=42，三卡并行（GPU 0/1/2）。
Baseline 沿用 `exp_mc_1p5b_pilot_m24_t1000_s42_20260506_162512` 的 B0/B1/B2，
**不重跑**。

| Run | `joint_loss.enabled` | λ | dist | w_k | 备注 |
|---|---|---|---|---|---|
| J0 `joint_l1_m24_lam0` | **true** | 0.0 | cosine | uniform | sanity：加模块但无 pull 信号，dev loss 应与 B1 基本重合 |
| J1 `joint_l1_m24_lam0p05` | true | 0.05 | cosine | uniform | 弱 pull |
| J2 `joint_l1_m24_lam0p2` | true | 0.2 | cosine | uniform | 标准 pull |
| J3 `joint_l1_m24_lam0p2_util` | true | 0.2 | cosine | utility (math+code, zscore, positive, mean-renorm) | 注入 utility |

Output 根目录：`outputs/joint_l1_pilot_m24_t1000_s42_<TAG>/`，内部沿用
`runs/<cfg>_s42/` 结构，写入 `launch_manifest.json`、`meta_clusters/`
（软链接复用 pilot 目录）、`logs/`。

## 9. 新增 / 修改产物与日志

对每个 run：
- `prototype_init.npy` (`[M, H]`, float32)；伴随
  `prototype_init_meta.json`（layer_idx / sample_count_per_cluster / 生成耗时）。
- `prototype_final.npy`（训练结束时 μ）。
- `prototype_history.jsonl`：每次 `pmp.update_interval` 步写一条
  `{step, proto_norm, pairwise_cos_min, pairwise_cos_mean, pairwise_cos_max,
  step_delta_norm}`。
- 原有 `train.log` 的 step log 额外加字段：`lm_loss=… pull_loss=… pull_eff=λ·pull_loss`。

## 10. 代码改动清单（顺序实现）

1. `trainer/cluster_prototypes.py`（新）：`ClusterPrototypes`、
   `build_prototype_init_from_model`、`build_utility_pull_weights`。
2. `configs/default.yaml`（改）：追加 `joint_loss` 段，全部默认关闭。
3. `trainer/integrated_trainer.py`（改）：
   - 在 `__init__` 末尾（LR scheduler 之后、训练循环之前）按 `joint_loss.enabled`
     条件加载 prototype 与 utility pull weights；
   - 主 loop 中 cluster-id 解析提前到 forward 前；
   - `_compute_lm_loss` → `_compute_joint_loss`（保留别名 `_compute_lm_loss`
     指向同一个 method 以便之后不改 `_evaluate_*` 里的调用处）；
   - step 日志 + `_log_prototype_stats`；
   - 训练结束保存 `prototype_final.npy`。
4. `scripts/run_joint_l1_pilot_launch.sh`（新）：仿
   `scripts/run_exp_mc_1p5b_launch.sh`，并行拉起 J0..J3。
5. 【可选 / 本轮不做】`analysis/analyze_joint_l1.py`：四组 run 结束后再加。

## 11. 风险 / 监控清单

| 风险 | 监控 | 阈值 / 响应 |
|---|---|---|
| prototype collapse | `pairwise_cos_max` | >0.95 立即 stop 该 run，减半 λ 或 `proto_lr_multiplier` |
| pull loss 反向炸 LM | 训练 log 中 `lm_loss` 趋势 | 若 100 步内 `lm_loss` 单调上升 > 0.3，立即减半 λ |
| 初始化 μ 偏差 | `prototype_init_meta.json` 里 `sample_count_per_cluster` | 任一 cluster count < 10 给 warning，仍继续 |
| J0 (λ=0) 与 B1 差异 | J0 vs B1 每个 eval 点差值 | 若任一 \|Δdev_loss\| > 0.02 说明引入 bug（多余 forward 参数 / 顺序变更），回滚查因 |

## 12. 非目标（明确不在本轮实现）

- 在线重分簇（Level 2+）。
- Sinkhorn / OT-based 防 collapse。
- gradient-level utility（Level 3）。
- DeepSpeed ZeRO 分支；FSDP 分支。
- prototype EMA / detached pull（经典 prototype learning 的变体）——本轮 μ 直
  接可训。
- task-level accuracy 评估扩展（AGENTS.md 明令）。
