# Joint Clustering Loss — Level 1 方案思路

## 一段话概述

针对当前"先算能力再二次聚类"的两步走方案过于人为、聚类空间与训练目标彼此独立这一结构性缺陷，Level 1 提出在原有预训练 LM 损失之外引入一个轻量级 cluster-pull 正则项，把每个样本的模型 hidden representation 拉向其所属 cluster 的可学 prototype $\mu_k$，并以 Exp2 得到的 utility 向量作为 $\mu_k$ 的初始化。这样做的核心价值是：cluster assignment 保持冻结（沿用现有 87-cluster 或 24-meta-cluster 划分，避免在线重分簇带来的工程复杂度与稳定性问题），但 prototype 向量会在训练过程中随 LM 损失和 cluster-pull 损失一起被梯度更新，使模型学到的 representation 空间逐步被 task-aware 的 prototype 拉扯成"功能性"结构，从而让 PMP 仍能在稳定 cluster 结构上工作，同时让 representation 在训练过程中"偏向能力贡献"——本质上用最小改动闭环验证"一边训练一边把功能性注入表示空间"这件事在我们 setting 下是否带来 dev loss 方向性改善。该级别不引入 per-sample gradient 计算、不引入 online re-assignment、不引入 Sinkhorn 等防 collapse 技巧，仅靠 $\lambda$ 控制 cluster-pull 强度，属于风险最低的 proof-of-concept：若它在 1.5B pilot 上能观察到 utility-meta cluster 内部 hidden state 更聚集、且 math/code dev loss 对比 baseline 出现可读方向（哪怕仅 Δ≈0.005 量级），就说明 joint objective 路线可行，可以推进到 Level 2 的 online re-assignment；若看不到信号，则说明光靠"拉 representation"不够，需要直接引入 gradient-level utility 项，直接跳到 Level 3 的完整端到端方案。
