# M18 / exp2 — ppRAG 检索精度实验（真实 MS MARCO 数据）

> 负责人: data-retrieval ｜ 代码: `experiments/exp2_retrieval.py` ｜ 门禁: `results/gate_M18_retrieval.json`
> 产物: `results/exp2_retrieval.json`（全量数字，含逐配置明细）、`results/exp2_retrieval.csv`、`figures/fig_retrieval.png`
> 复跑: `env -u ALL_PROXY .venv/bin/python experiments/exp2_retrieval.py`（全量 45 s；`--n-queries 30` 可快速跑）
> 论文位置: §V Experimental Evaluation（论文声称 "maintains high retrieval accuracy"，但没有给出检索精度表/图；本实验是 MODULE_PLAN M18 定义的补充实验）

## 1. 实验设置

| 项 | 值 |
|---|---|
| 数据库 | `data/`：m = **50,000** 真实 MS MARCO passage（1053 qrels-relevant + 18,544 官方 BM25 候选硬负例 + 30,403 均匀随机填充，见 `notes/data_manifest.md`） |
| 查询 | `data/eval_qids.json` 的 **1000** 条 dev query（qrels 覆盖 100%）；gtr-t5-base 768-d L2 归一化 |
| 参数 | s=3, β=0.2（论文隐私实验设置）, k=10, 重排 = 客户端 DecDB 后用**明文精确 L2 距离**重排（SPEC.md §3 约定，隔离"加密检索损失"与"重排模型损失"） |
| 指标 | ① Recall@10（加密 top-10 与明文 top-10 的集合重叠率，越接近 1 越好）；② qrels MRR@10 / Recall@10 / nDCG@10（相对官方 qrels）；③ 平均 k'；④ 单 query 端到端耗时（EncQ + 云端 top-k' 检索 + DecDB(k') + 重排 + AES 解密） |
| 硬件 | Apple M5 CPU（无 GPU）；加密/检索由 numpy/Python 实现；嵌入预计算（MPS fp16） |
| 公平性说明 | 明文与密文检索均用同一个 **float32 Gram-trick top-k 内核**计时（`EncryptedVectorDB` 存 float32；`PlaintextDB` 固定 float64，故另测 float32 内核 2.45 ms，float64 精确参考 4.26 ms） |

## 2. 结果（真实测量，m=50,000，1000 queries，k=10，β=0.2）

| 配置 | Recall@10 vs 明文 | qrels MRR@10 | qrels Recall@10 | qrels nDCG@10 | 平均 k' | 耗时 ms/query | 备注 |
|---|---|---|---|---|---|---|---|
| **plaintext**（明文基线） | 1.0000 | 0.7230 | 0.970 | 0.7836 | 10 | 2.45 (float32 内核) / 4.26 (float64) | 检索上界 |
| CAPRISE k'=k（严格 Alg.2，无 DP） | **0.9699** | 0.7230 | 0.970 | 0.7836 | 10 | 2.35（server search 2.17 + 客户端 ~0.2） | 加密噪声造成 top-10 边界翻转 |
| CAPRISE k'=15（oversampling） | **0.9999** | 0.7230 | 0.970 | 0.7836 | 15 | 2.39 | 几乎无损 |
| CAPRISE k'=2k=20（oversampling） | **1.0000** | 0.7230 | 0.970 | 0.7836 | 20 | 2.56 | 完全恢复明文 top-10 |
| CAPRISE + DistanceDP r=0.02（k'=Theorem1） | **1.0000** | 0.7230 | 0.970 | 0.7836 | **72** | 3.11 | 定理保证的扩张集 |
| CAPRISE + DistanceDP r=0.05（k'=Theorem1） | **1.0000** | 0.7230 | 0.970 | 0.7836 | **785** | 11.81 | 隐私更强，成本 ∝ k' 解密 |
| ADCPE (SAP 基线) k'=k | 0.9724 | 0.7232 | 0.970 | 0.7842 | 10 | 2.34 | 论文对照方案 |
| ADCPE k'=2k | 1.0000 | 0.7230 | 0.970 | 0.7836 | 20 | 2.44 | |

**β 敏感性**（CAPRISE，k'=k；β 越大 = 噪声半径 3sβ/8 越大 = 隐私越强/精度越低）

| β | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|
| Recall@10 vs 明文 | **0.9910** | 0.9854 | 0.9699 | **0.9401** |
| qrels MRR@10 | 0.7230 | 0.7231 | 0.7230 | 0.7232 |

**Phase-1 上传（含 EncDB + AES-GCM，m=50,000）**：0.71 s，≈ 70,140 vec/s（本机 CPU 单进程，一次性）。

## 3. 结论与分析

1. **CAPRISE 严格 k'=k 有轻微精度损失**：Recall@10（相对明文）0.9699，即加密噪声使约 3% 的 top-10 文档在"噪声边界"处翻转。这正是 Lead 合成数据 smoke test 中观察到的现象（k'=k ≈0.87）；真实 50k 语料 + 硬负例下损失更小（0.97）。
2. **少量 oversampling 即可完全恢复**：k'=15 → 0.9999，k'=20=2k → 1.0000，额外成本只有 ~0.04–0.2 ms/query。**实用建议：k'=2k**。
3. **DistanceDP 场景由 Theorem 1 给出 k'，Recall@10 达到 1.0000**：r=0.02 → k'=72（3.11 ms/query）；r=0.05 → k'=785（11.81 ms/query）。成本主要来自**客户端解密 k' 个候选向量**（785 × DecDB ≈ 9.5 ms），云端检索本身几乎不随 k' 增加。
4. **端到端 qrels 质量几乎不变**：所有加密配置的 qrels MRR@10 都在明文 0.7230 的 ±0.0002 内。原因（诚实的协议性质）：客户端对候选集做**明文精确距离重排**，只要相关文档进了候选集，排序就能恢复；因此差异体现在"候选集是否含明文 top-10 文档"（即 Recall@10 vs 明文），而不是 MRR。论文未指定 ReRank 模型，这是上限设置（SPEC.md §3）。
5. **β 隐私-精度权衡单调**：β 0.05→0.4 时 Recall@10 0.9910→0.9401，即 β 越小（噪声半径越小、隐私越弱）精度越高，与论文 Discussion 的 "stronger privacy requires broader retrieval, may reduce precision" 定性一致。
6. **CAPRISE vs ADCPE**：同 k'=k 下 ADCPE recall 0.9724，略高于 CAPRISE 0.9699。这与两者噪声预算分配一致——ADCPE 对所有向量统一用 sβ/4=0.15，而 CAPRISE 把数据库噪声放宽到 3sβ/8=0.225、查询侧收紧到 sβ/8=0.075（换取"库内距离不保留"的隐私性）；查询-库距离的扰动总量略大，所以检索保真度略低。**这是为隐私付出的真实精度代价（小 ~0.25%）**。
7. **与论文对照**：论文没有检索精度表，无法逐数字对照；本实验支持论文 "high retrieval accuracy" 的定性主张（严格 k'=k 时 top-10 保真度 97%，2k oversampling 后 100%）。论文 Table III 用 m=100,000、r∈{0.033,0.02}；本实验 m=50,000、r∈{0.02,0.05}，且 SPEC.md §5 已指出论文表值对应 Δα≈1.2r，而本实验按 RemoteRAG/SPEC 约定用 Δα=r（k' 更保守）。

## 4. 复现说明

- 加密每次使用新的 128-bit nonce（`os.urandom`），因此逐次运行数字有 ~1e-3 量级抖动；key 生成用固定种子（`--seed 20260101`）。
- 全部数字由 `experiments/exp2_retrieval.py` 单次运行产生（45.3 s，1000 queries × 11 配置），JSON 中保留每个配置的 `latency_ms` 均值/中位/P95 与 `k_prime_set`。
- 中间 smoke 结果：m=8000 子集上 CAPRISE k'=k 的 Recall@10=0.959（M15 门禁），与全量 50k 的 0.9699 接近。
