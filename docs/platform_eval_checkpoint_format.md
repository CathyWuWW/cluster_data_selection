# 平台伴生评估 — Checkpoint 目录格式与避坑清单

本文档记录我们在接入内网平台 `hf-flow-eval` 伴生评估任务时踩过的所有坑，以及最终验证可用的 checkpoint 目录结构、tokenizer 修复步骤、平台 `start_cmd` 参数。**之后每次给平台交 ckpt，都按本文件走，不再重新踩坑。**

---

## 1. 目录结构（平台期望）

平台脚本硬性要求：

```
<CKPT_BASE_PATH>/
└── global_step<N>/          ← N 是任意正整数；只有一个 ckpt 就随便起
    └── hf/
        ├── config.json                (必需)
        ├── tokenizer.json              (必需)
        ├── tokenizer_config.json       (必需)
        ├── generation_config.json      (推荐，vLLM 会读)
        ├── model.safetensors           (单文件) —— 或 ——
        └── model-00001-of-*.safetensors + model.safetensors.index.json  (分片)
```

**关键点**：
- `global_step<N>` 这一级**不能省**，即使只有一个 ckpt。脚本是按 `global_step*/hf` 模式匹配的。
- `hf/` 这一级也不能省。
- 文件必须是**标准 HF 格式**（`safetensors` 单文件或分片都可以，不要 `.bin`）。

---

## 2. 平台可见路径（存放位置）

**`/jizhicfs/wuyanning/...` 平台容器不可见**。必须放到平台挂载的共享盘。

我们当前用的路径（师兄协助挂载）：

```
/apdcephfs_jn4/share_304380933/rongyiyu/wuyanning/platform_eval_ready/<EXP_NAME>/global_step<N>/hf/
```

从训练机器 `/jizhicfs/wuyanning/` 到 apdcephfs 的复制用：

```bash
rsync -a --info=progress2 \
    outputs/platform_eval_ready/<EXP_NAME>/ \
    /apdcephfs_jn4/share_304380933/rongyiyu/wuyanning/platform_eval_ready/<EXP_NAME>/
```

本机 staging 命令（已封装）：

```bash
SRC_HF=outputs/.../final \
DST_ROOT=outputs/platform_eval_ready \
EXP_NAME=<exp_name> \
STEP=<step> \
COPY_MODE=copy \
bash scripts/stage_for_platform_eval.sh
```

`stage_for_platform_eval.sh` 做了三件事：目录结构整理、**tokenizer_config.json 自动修复**（见第 3 节）、生成 `eval_config_snippet.sh`。之后还需手动 rsync 到 apdcephfs。

---

## 3. 必踩坑：`tokenizer_config.json` 的 `extra_special_tokens`

**现象**：平台日志反复打

```
starting eval:
Regularly KILL Potential Procs.
starting eval:
...
```

但 vLLM 服务从未起来。

**根因**：新版 transformers（训练时用的）把 `extra_special_tokens` 写成 **list**，平台老版 tokenizers 期望 **dict**，直接报错导致子进程退出。

**修复**：删掉这个字段（这些 token 在 `tokenizer.json` 里已完整定义，config 里的 list 是冗余声明）。

自动脚本：

```bash
python scripts/fix_tokenizer_config.py <path_or_dir> [<path> ...]
# 或全仓扫描：
python scripts/fix_tokenizer_config.py --glob 'outputs/**/tokenizer_config.json'
```

- 会对每个文件先备份 `.bak`（只在无 `.bak` 时备份），再删除字段
- 支持 `--dry-run`
- `stage_for_platform_eval.sh` 在 copy 完成后会自动调用它，**新 staging 的目录不用手动修**

---

## 4. 平台 `start_cmd` 参数（验证可用版）

把 `CHIEF Node` 分支里的参数区替换成：

```bash
BEGIN_STEP=0
END_STEP=99999999
EVAL_ITER=<你的 step>            # 例：600、1000、12000。EVAL_ITER=1 有时会被误判，设成实际 step 最稳
MAX_EVAL=1

EXP_NAME_BASE="<exp_name>"

CKPT_BASE_PATH="/apdcephfs_jn4/share_304380933/rongyiyu/wuyanning/platform_eval_ready/<exp_name>"
EVAL_RESULT_PATH="/apdcephfs_jn4/share_304380933/rongyiyu/wuyanning/platform_eval_ready/<exp_name>/eval_result"

TASK_SET="all"        # 首次调通建议 "all"；core 在这套脚本上有过空集合问题
BACKEND="vllm"
TP_SIZE=1             # 0.5B / 1.5B 小模型单卡够，设 1
API_WORKER_NUM=8
GPU_PER_NODE=8        # 必须和平台申请的 GPU 数量一致（师兄默认 H800 8 卡，就写 8）
```

**字段踩坑清单**：

| 字段 | 错误设置 | 正确设置 | 症状 |
|---|---|---|---|
| `CKPT_BASE_PATH` | `/jizhicfs/...` | `/apdcephfs_jn4/...` | 无限 `waiting ckpt` |
| `BEGIN_STEP/END_STEP` | `500/2000` | `0/99999999` | step=600 被过滤 |
| `EVAL_ITER` | `1` | 实际 step 值（600/1000/...） | 有时被判为无有效 step |
| `GPU_PER_NODE` | 和资源申请不一致 | 申请几卡写几 | 启动脚本分配异常 |
| `TASK_SET` | `core` | `all` | core 任务集合可能未实装 |
| tokenizer_config | list | dict（建议删除该字段） | `starting eval` 重复 |

---

## 5. 仓库与镜像配置（基本不动）

平台任务配置页面：

- **主仓库**：`rongyiyu/hf-flow-eval`
- **分支**：`hf-stream-eval-pipline`
- **Commit**：用模板默认，不要乱换
- **拉取 submodule**：开启
- **是否拉取 git 代码**：选"是"

`init_cmd` 基本保持模板原样（vLLM 安装 + swanlab 登录 + /opt/venv 路径优先）。`API_KEY` 建议换成平台 secret 环境变量，不要长期明文。

---

## 6. 一次完整的评估流程（照做即可）

```bash
# ① 在训练机本地 staging（自动做目录整理 + tokenizer 修复）
cd /jizhicfs/wuyanning/cluster_data_selection
SRC_HF=outputs/<exp>/runs/<run>/final \
DST_ROOT=outputs/platform_eval_ready \
EXP_NAME=<my_exp_name> \
STEP=<N> \
COPY_MODE=copy \
bash scripts/stage_for_platform_eval.sh

# ② 同步到 apdcephfs（平台可见）
rsync -a --info=progress2 \
  outputs/platform_eval_ready/<my_exp_name>/ \
  /apdcephfs_jn4/share_304380933/rongyiyu/wuyanning/platform_eval_ready/<my_exp_name>/

# ③ 去平台创建任务（或修改已有任务）
#    - 主仓库: rongyiyu/hf-flow-eval (branch: hf-stream-eval-pipline)
#    - init_cmd: 模板不动
#    - start_cmd: 把 CHIEF Node 参数区替换成第 4 节的版本，exp_name / CKPT_BASE_PATH / EVAL_ITER 改成自己的
#    - GPU: 申请 H800 x 8（或其他；和 GPU_PER_NODE 一致）

# ④ 启动任务，在日志里找：
#    "[HF Mode] Start evaluating: <my_exp_name>"
#    "Executing: CUDA_VISIBLE_DEVICES=0 vllm serve .../global_step<N>/hf ..."
#    "vLLM service started successfully."
#    看到这三行就说明评测真的开始了。
```

---

## 7. 调试快速检查清单

平台卡住不报错 / 反复 `starting eval` 时，按此顺序查：

1. **CKPT 路径可见**：`ls` 一下 `CKPT_BASE_PATH`，确认容器能读
2. **目录结构**：`ls CKPT_BASE_PATH/global_step*/hf/` 里必须有 config/tokenizer/model.safetensors
3. **tokenizer_config.json**：`grep extra_special_tokens tokenizer_config.json` 不应该返回任何东西
4. **step 范围**：`EVAL_ITER` ≤ 文件夹 step 编号，`BEGIN_STEP`/`END_STEP` 范围要包住
5. **GPU 数量一致**：`GPU_PER_NODE` == 平台申请卡数
6. **`TASK_SET`**：不确定先用 `all`
7. **真正的报错在子进程日志**：容器里 `/root/err.*.8000` 和 `/root/log.*.8000`

---

## 8. 重要产物索引

| 用途 | 路径 |
|---|---|
| Staging 脚本 | `scripts/stage_for_platform_eval.sh` |
| Tokenizer 修复脚本 | `scripts/fix_tokenizer_config.py` |
| 已验证 0.5B ckpt | `/apdcephfs_jn4/share_304380933/rongyiyu/wuyanning/platform_eval_ready/cluster_pretrain_exp_mc125237_utility_m24_s43/` |
| 本地原始 final/（含 .bak） | `outputs/exp_mc_m24_t600_20260430_125237/runs/<run>/final/` |
