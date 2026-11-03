# 复现状态（会话恢复用）

更新时间: 2026-11-03 15:40 (+08:00)
论文: *Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption* (arXiv:2601.12331v1, 2026-01-18, NTU/DTC)
论文源: `doc/References/CAPRISE_2026_Privacy_Preserving_RAG_DPE.pdf`（LaTeX 源 `output/pdf/source/caprise/ppRAG_FLLM.tex`）；代码源: 论文未公开，本仓库独立复现
当前阶段: Phase 5（除 M11 外全部模块 PASS；M11 Vec2Text 环境已打通，正在跑；交叉验收进行中）
当前目标: 完成 M13（Theorem 1/Table III）与 M10/M11（隐私实验），并行推进 M0 数据 → M14–M18
本轮门禁: P0 skeleton — PASS（见 results/gate_P0_skeleton.json）
证据: results/gate_summary.json, results/gate_*.json

| 模块 | 依赖 | 状态 | 门禁证据 | 备注 |
|---|---|---|---|---|
| P0 | — | PASS | results/gate_P0_skeleton.json | 骨架门禁（含 fail-fast 自检） |
| M0 数据 | — | PASS | results/gate_M0_data.json | m=50,000（1053 qrels + 18,544 BM25 硬负 + 随机填充）；1000 query；qrels 覆盖 100%；MRR@10=0.722 |
| M1 PRF | — | PASS | results/gate_M1_prf.json | KS p=0.35/0.74 |
| M2 KeyGen | M1 | PASS | results/gate_M2_keygen.json | s∈[1,2^16], K=128bit |
| M3 噪声 | M1 | PASS | results/gate_M3_noise.json | 模长<R 严格；球内均匀 KS p=0.63 |
| M4 EncDB | M2,M3 | PASS | results/gate_M4_encdb.json | 往返 2.8e-17；距离比 3.008 |
| M5 DecDB | M4 | PASS | results/gate_M5_decdb.json | 精确逆；错误 nonce 不可恢复 |
| M6 EncQ | M2,M3 | PASS | results/gate_M6_encq.json | 往返 1.4e-17；‖η‖<sβ/8 |
| M7 Eq.(2) | M4,M6 | PASS | results/gate_M7_eq2.json | 50k 前提三元组 0 违反 |
| M8 Eq.(3) | M4 | PASS | results/gate_M8_eq3.json | 对抗可实现；实测 0/20000（finding F2） |
| M9 ADCPE | M2,M3 | PASS | results/gate_M9_adcpe.json | 20k β-DCP 0 违反 |
| M10 向量分析 | M4,M9 | IN_PROGRESS | — | privacy-attacks；exp4 初版已出（需真实 GTR + 修 dist_err NaN） |
| M11 Vec2Text | M4,M0 | PENDING | — | 依赖 .venv-vec2text（transformers 4.x，已验证导入） |
| M12 DistanceDP | — | PASS | results/gate_M12_distance_dp.json | Gamma 均值 0.099989（期望 0.1） |
| M13 Theorem1 | M12 | PASS | results/gate_M13_theorem1.json | 20/20；Δα/r=1.213±0.044（A10）；MC 偏差<2.4%；最小 k'≈k（F9） |
| M14 Phase1 | M4,M9 | PASS | results/gate_M14_phase1.json | 12/12；服务器无密钥/无明文；AES 8000/8000；DecDB 9.4e-9 |
| M15 Phase2 | M6,M12,M13,M14 | PASS | results/gate_M15_phase2.json | 11/11；只收 e'_q；k'=Theorem1=55；recall@10=0.959(k'=k)/1.000(k'=20,DP) |
| M16 Phase3 | M15 | PASS | results/gate_M16_phase3.json | 11/11；prompt 精确、无未检索文档泄漏 |
| M15 Phase2 | M6,M12,M13,M14 | PENDING | — | 等待上游 |
| M16 Phase3 | M15 | PENDING | — | 等待上游 |
| M17 效率 | M4,M6,M1 | PASS | results/gate_M17_efficiency.json | 18/18；emb/s 101.6k→33.9k；overhead 0.23–0.69%；78.6k vec/s vs Paillier 2.19/0.30 vec/s |
| M18 检索精度 | M15,M0 | PASS | results/gate_M18_retrieval.json | 19/19；CAPRISE k'=k recall@10=0.9699；k'=15→0.9999；DP r=0.02 recall=1.0，k'=72；β 单调 |
| M19 报告 | 全部 | PENDING | — | Lead 收尾 |

下一动作:
1. 运行 P0 门禁：`.venv/bin/python -m gates.run_gates --from P0 --to P0`
2. 用 .venv-vec2text 验证 gtr-base corrector 加载 + 2-step 反演（进行中），通过后通知 privacy-attacks 用该环境跑 M11
3. 等 data-retrieval 的 M0 门禁；等 theory-efficiency 的 M13 门禁；等 privacy-attacks 的 M10 门禁
4. 收到各模块 PASS 后：Lead 独立重跑门禁 + 逐行读 diff + 手算抽查，然后提交 git

阻塞项:
- MS MARCO 语料/嵌入尚未生成完（M0）
- vec2text 与 transformers 5.x 不兼容 → 已建 .venv-vec2text（transformers 4.57.6 + sentence-transformers 4.1.0 + pyarrow 16.1 + numpy 2.2.6），待加载测试
- 论文 Table III 与正文约定不一致（A10/F1）→ 由 M13 量化，属 finding 不阻塞
