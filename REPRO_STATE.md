# 复现状态（会话恢复用）

更新时间: 2026-11-03 23:00 (+08:00)
论文: *Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption*
(arXiv:2601.12331v1, 2026-01-18, NTU/DTC)
论文源: `doc/References/CAPRISE_2026_Privacy_Preserving_RAG_DPE.pdf`（LaTeX 源 `output/pdf/source/caprise/ppRAG_FLLM.tex`）；
代码源: 论文未公开，本仓库独立复现。
当前阶段: **R3 — 全要素对齐收尾**（论文逐行/逐式/逐行算法/逐表图覆盖矩阵见 `docs/PAPER_COVERAGE.md`）
当前目标: 完成 M11 敏感性 + 全量门禁回归 + 文档数字同步，然后交付。

## 1. 模块门禁

| 模块 | 依赖 | 状态 | 门禁证据 | 备注 |
|---|---|---|---|---|
| P0 骨架 | — | PASS | results/gate_P0_skeleton.json | fail-fast 自检 |
| M0 数据 | — | PASS | results/gate_M0_data.json | m=50,000；1000 query；qrels 覆盖 100%；MRR@10=0.722 |
| M1 PRF | — | PASS | results/gate_M1_prf.json | 确定性/敏感性/KS |
| M2 KeyGen | M1 | PASS | results/gate_M2_keygen.json | s∈[1,2^16]、K=128 bit |
| M3 噪声 | M1 | PASS | results/gate_M3_noise.json | 模长<R 严格；球内均匀 |
| M4 EncDB | M2,M3 | PASS | results/gate_M4_encdb.json | 往返 2.8e-17；距离尺度比 3.008 |
| M5 DecDB | M4 | PASS | results/gate_M5_decdb.json | 精确逆；错误 nonce 不可恢复 |
| M6 EncQ | M2,M3 | PASS | results/gate_M6_encq.json | 往返 1.4e-17；‖η‖<sβ/8 |
| M7 Eq.(2) | M4,M6 | PASS | results/gate_M7_eq2.json | 50k 前提三元组 0 违反 |
| M8 Eq.(3) | M4 | PASS | results/gate_M8_eq3.json | 对抗可实现；随机采样 0/20000（F2） |
| M9 ADCPE | M2,M3 | PASS | results/gate_M9_adcpe.json | 20k β-DCP 0 违反 |
| M10 向量分析 | M4,M9 | PASS | results/gate_M10_vector_analysis.json | 29 检查；论文 Fig.5 数字化；27/27 cell |
| M11 Vec2Text | M4,M0 | PASS（敏感性收尾中） | results/gate_M11_vec2text.json | 50 条 + 95% CI；敏感性 50/10 见 §3 |
| M12 DistanceDP | — | PASS | results/gate_M12_distance_dp.json | Gamma 均值 0.099989；KS p=0.996 |
| M13 Theorem1 | M12 | PASS | results/gate_M13_theorem1.json | 36 检查；四约定 + 舍入细分 + RemoteRAG 交叉验证（N=10^5 已确认） |
| M14 Phase1 | M4,M9 | PASS | results/gate_M14_phase1.json | 服务器无密钥/无明文；AES 8000/8000 |
| M15 Phase2 | M6,M12,M13,M14 | PASS | results/gate_M15_phase2.json | 只收 e'_q；k'=Theorem1；top-10 一致 |
| M16 Phase3 | M15 | PASS | results/gate_M16_phase3.json | prompt 精确、0 泄漏 |
| M17 效率 | M4,M6,M1 | PASS | results/gate_M17_efficiency.json | 36 检查；论文 15 维网格；paper-style 探针 |
| M18 检索精度 | M15,M0 | PASS | results/gate_M18_retrieval.json | recall 0.9699；DP k'=72/785 |
| M19 报告 | 全部 | PASS | REPRODUCTION_REPORT.md | 见 §4 |

**独立证明链审计**: `scripts/check_proof_chain.py` → `results/proof_chain_check.json`
（Eq.(4)–(20) 逐式，4000+4000 组；Eq.(12) 打印式符号笔误 F16）。

## 2. 本轮新增/变更（2026-11-03 晚）

1. **Fig.4 对齐**：效率实验维度从 6 点改为论文 Fig.4 的 **15 点**（64…3584），新增
   "论文实现风格"（逐元素 Python 标量）探针、论文图像素数字化对照、负载校准与 `previous_runs`。
   结论：paper-style 复现论文量级/形状（中位 1.41×，drop 42.1× vs 44.5×），向量化实现快
   一个量级以上且曲线更平（A17/F5）；d=256 凹陷未复现，已登记。
2. **Table II 对齐**：Vec2Text 样本 10 → **50 条真实 passage**，bootstrap 95% CI；
   CAPRISE raw BLEU **13.696 [9.72,17.48]** vs 论文 **12.425**（CI 覆盖论文值）；
   明文 **69.875 [61.99,78.17]** vs 论文 83.021（0.84×）；rescaled 最强攻击 34.584。
3. **Table III 对齐**：四种 Δα 约定系统枚举 + 舍入感知细分（r=0.02 列 8/8 兼容、r=0.033 列 0/8）
   + RemoteRAG 参照实现交叉验证（附录 B.4 文档默认 N=10^5，r\*=0.03392→0.03，与 Δα=r 自洽）。
   结论：论文 r=0.033 列只能由未声明的 `Δα≈1.2r` 复现（1.2129±0.0435）；被引实现用 `Δα≈r`（F1/A10）。
4. **证明逐式复核**：Eq.(12) 符号笔误（F16），其余 Eq.(5)–(20) 0 违反。
5. **文档修正**：F1 的出处（把误归于 CAPRISE 的 `Δα≈r` 改回 RemoteRAG §3.2.2）、A10/A13/A16 更新、
   新增 `docs/PAPER_COVERAGE.md`（论文要素覆盖矩阵）、C8 声明、F16 发现。

## 3. 进行中 / 待确认

- **M11 敏感性**：`--phase sensitivity --sensitivity-n 5 --sensitivity-steps 50 --sensitivity-beam 10
  --sensitivity-batch-size 1`（5 条 × 3 模式；共享机/16GB 内存限制，~108 s/text）。
  完成后脚本自动重写 JSON/CSV/图并追加 `step_beam_sensitivity` finding。
- **M11 notes**（`notes/exp3_vec2text.md`）仍是 10 条样本版本，需在敏感性完成后重写为 50 条 + CI + 50/10。
- **全量门禁回归**：M11 敏感性完成后重跑 `python -m gates.run_gates --from P0 --to M18`。

## 4. 关键证据

- 覆盖矩阵：`docs/PAPER_COVERAGE.md`（公式逐式、Alg.1/2 逐行、表/图/章节声明）
- 声明判定：`docs/CLAIMS_CHECKLIST.md`（VERIFIED 20 / FALSIFIED 2 / BLOCKED 1 / UNKNOWN 0）
- 论文问题：`docs/FINDINGS.md`（F1–F16）
- 歧义决策：`docs/PAPER_ANALYSIS.md` §12、`SPEC.md` §2.2
- 结果汇总：`results/SUMMARY.md`（`python scripts/collect_results.py` 重新生成）

## 5. 复跑命令

```bash
cd code/CAPRISE复现
.venv/bin/python -m gates.run_gates --from P0 --to M18   # 全部模块门禁
.venv/bin/python scripts/check_proof_chain.py            # 证明逐式审计（Lead 侧）
.venv/bin/python experiments/exp1_efficiency.py          # Fig.4（~3 min）
.venv/bin/python experiments/exp5_query_dp.py            # Table III / Theorem 1
.venv/bin/python experiments/exp2_retrieval.py           # 检索精度（补充）
.venv/bin/python experiments/exp4_vector_analysis.py     # Fig.5
.venv-vec2text/bin/python experiments/exp3_vec2text.py --metrics-only   # Table II 指标复算
# 若需重跑 Vec2Text 攻击（主实验 ~40 min + 敏感性）：
.venv-vec2text/bin/python experiments/exp3_vec2text.py --phase all
```
