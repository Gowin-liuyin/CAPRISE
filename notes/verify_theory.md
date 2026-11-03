# 交叉验收：theory-efficiency 的 M12 / M13 / M17（独立复核）

> 验收人: data-retrieval ｜ 任务: task-6（Lead 消息里误写为 task-5；task-5 在任务板上是"验收 data-retrieval 自己"的反向任务，我不应自验，故 claim 了 task-6）
> 被验收: theory-efficiency（`experiments/exp5_query_dp.py`、`experiments/exp1_efficiency.py`、`gates/gate_m12_distance_dp.py`、`gates/gate_m13_theorem1.py`、`gates/gate_m17_efficiency.py` 及其 results/notes）
> 原则: 不采信自述；自己重跑门禁 + 从零实现独立复算（不 import 对方实验脚本，独立用 scipy betainc/betaincinv/brentq + 自写 Monte Carlo）
> 产物: `results/verify_theory.json`（机器可读，含全部复算明细）、本文件
> 结论: **PASS** —— 3 个门禁独立重跑均 PASS，所有关键数字独立复算一致；发现 6 条 minor issue（V1–V5 为表述/一致性问题，V6 为 JSON 舍入说明；均不影响测量值与结论，不改被验收方文件，只建议修）

---

## 1. 独立重跑门禁（自己观察到的输出）

```bash
.venv/bin/python -m gates.run_gates --from M12 --to M13    # M12 PASS, M13 PASS, OVERALL: PASS
.venv/bin/python -m gates.run_gates --from M17 --to M17    # M17 PASS, OVERALL: PASS
```

| 门禁 | 结果文件 | 检查数 | pass | 时长 | 本次重跑时间戳 |
|---|---|---|---|---|---|
| M12 DistanceDP | `results/gate_M12_distance_dp.json` | 18 | ✅ | 0.042 s | 2026-11-03T16:19:31 |
| M13 Theorem 1 / Table III | `results/gate_M13_theorem1.json` | 19 | ✅ | 2.674 s | 2026-11-03T16:19:34 |
| M17 Efficiency | `results/gate_M17_efficiency.json` | 18 | ✅ | 0.034 s | 2026-11-03T16:19:35 |

重跑中观察到的关键输出（原文摘要）：
- M12：`gamma_mean_equals_n_over_eps`（mean=0.099989 vs 0.1）、`gamma_ks_test_p_gt_0.01`（p=0.99567）、`norm_shift_equals_realised_radius_fixed`（误差 6.9e-18）、`directions_unit_norm`、`seeded_reproducible`、非法参数拒绝 5/5 —— 检查真实（非硬编码），种子固定（20260118）。
- M13：`formula_table_recomputed_consistently` max diff = **0.000e+00**；`mc_widened_cap_matches_formula_within_5pct` max rel err = **0.0238**；`mc_coverage_topk_sufficiency` = 1.0（200 query × 16 格）；`reverse_dalpha_converges_to_stable_value` mean=**1.21285**, std=0.0435, range=[1.15373, 1.29769]。
- M17：`independent_recheck_d768` = **87,233 emb/s** vs stored **86,492**（ratio 1.01）；`phase1_caprise_encryption_measured` EncQ n=128 median=**1.471 ms**（87,035 vec/s）；overheads=[0.691, 0.233, 1.85]%；`phase2_caprise_measured` **78,566.8 vec/s**（12.728 µs/vector）；Paillier 1024/2048-bit = 2.1852 / 0.2966 vec/s。

## 2. 独立复算（从零实现，未 import 被验收脚本）

### 2.1 Table III 公式格（Δα = r）— 用 scipy 独立实现
独立公式：`F(α) = 1 − I_{(1+cosα)/2}((n−1)/2, (n−1)/2)`（`scipy.special.betainc`），`α_k = F^{-1}(k/m)`（`betaincinv`），`k' = k + m·(F(α_k+r) − F(α_k))`。

| 指定格 | 我的独立值 | JSON 值 | 差 |
|---|---|---|---|
| n=768, r=0.033, k=5 | **147.250961377344** | 147.2509613773437 | **0.0** |
| n=1536, r=0.02, k=20 | **293.140863326560** | 293.140863326560 | **0.0** |

全部 16 格最大绝对差 = **0.0**（JSON 存全精度浮点，独立实现逐位一致）。

### 2.2 反推论文表值对应的 Δα/r（brentq）— 与 JSON 对照
对每格解 `m·(F(α_k+Δα) − F(α_k)) = k'_paper − k`（`scipy.optimize.brentq`, xtol=1e-14）。

| 指定格 | 论文 k' | 我的 Δα (rad) | 我的 Δα/r | JSON Δα/r | Δα 差 | Δα/r 差 |
|---|---|---|---|---|---|---|
| n=768, r=0.033, k=5 | 258 | 0.0393595190 | **1.19271270** | 1.19271 | 1.03e-9 | 2.70e-6 |
| n=1536, r=0.02, k=20 | 501 | 0.0245952864 | **1.22976432** | 1.22976 | 3.55e-9 | 4.32e-6 |

全部 16 格：Δα/r **mean=1.212852, std=0.043497, range=[1.153734, 1.297690]**，与 JSON（1.21285 / 0.0435 / [1.15373, 1.29769]）一致；Δα 最大差 **4.94e-9 rad**。
> 关于"容差 1e-6"：JSON 把 Δα/rad 四舍五入到 8 位、Δα/r 到 5 位，因此 **Δα** 层面差 <5e-9（远优于 1e-6），**比值**层面差 4.4e-6 纯粹是 JSON 存的 5 位小数舍入，不是数值分歧（见 issue V6）。

结论：**finding A10 独立成立** —— 论文 Table III 隐含 Δα ≈ 1.21 r（约 1.15–1.30 r），与正文/RemoteRAG 的 Δα≈r 系统性不符；Δα=1.2r 约定下论文/公式比均值 1.042（notes 表格与 JSON 一致）。

### 2.3 独立 Monte-Carlo（m=20000, nq=20, n=768, r=0.033, k=5, seed=424242）
自写 MC：随机单位 DB + 随机单位 query，`q̃ = q + r·u`（DistanceDP，不归一化），统计扰动 query 的 `α_k+r` 角帽内 DB 点数，与 `m·F(α_k+r)` 比较。

| 量 | 值 |
|---|---|
| 公式 k' = m·F(α_k+r) | 103.3615 |
| MC 角帽点数均值（population α_k） | 101.30 → 相对偏差 **−1.99%** |
| MC 角帽点数均值（empirical α_k） | 97.30 → 相对偏差 **−5.86%** |
| 覆盖率（k'=ceil(103.36)=104，取扰动距离 top-104 是否含明文 top-5） | **100%** |
| 真实检索所需最小 k' | mean 5.25 / median 5 / p95 6.05（远小于公式 103） |

偏差 **<10%**（要求），并且独立复现了"Eq.21 是充分但极宽松上界（A10b）"的方向。

### 2.4 exp1 效率数字手算（从 JSON 原始 runs_s / elapsed 手算）
吞吐（`embeddings/s = batch / median(runs_s)`, `elements/s = emb/s × d`）：

| d | 手算 median (ms) | 手算 emb/s | JSON emb/s | 相对差 |
|---|---|---|---|---|
| 384 | 9.843 | 101,595.0 | 101,595.0 | +4.2e-7 |
| **768** | **11.562** | **86,490.2** | **86,491.5** | **−1.5e-5** |
| 1536 | 17.904 | 55,853.4 | 55,853.4 | +7.3e-7 |
| 2048 | 20.270 | 49,334.0 | 49,333.4 | +1.2e-5 |
| 3072 | 25.237 | 39,624.4 | 39,624.4 | −1.0e-6 |
| 4096 | 29.529 | 33,865.0 | 33,864.9 | +3.4e-6 |

→ d=768 手算 **86,490.2 emb/s**（JSON 86,491.5 ≈ Lead 说的 86,492；差异仅来自 runs_s 存 6 位小数的舍入）。元素/s 也逐行一致。

Phase-1 overhead（EncQ n=128 手算 median = **1.471 ms**，与 JSON 一致）：
- MPS fp16 嵌入 212.8 ms → 1.471/212.8 = **0.6913%**（JSON 0.691%，差 2.6e-4）
- CPU fp32 嵌入 632.105 ms → **0.2327%**（JSON 0.233%）
- "论文 A100 嵌入 + 本实现加密"混合参照 79.52 ms → **1.8498%**（JSON 1.85%，且 JSON 明确标注 "hardware-mixed reference"）
- 检查 `1.47/212.8 = 0.69%` ✓。嵌入来源 JSON 明确 `measured_local_gtr_t5_base`（真测，非引用论文值）。

Phase-2：CAPRISE 2000 vec / 25.46 ms → 78,554.6 vec/s（JSON 78,566.8，差 0.016%，elapsed_ms 舍入）；Paillier 1024-bit：3840 个元素实测 2.288 s → 595.83 µs/element → **2.1853 vec/s**（JSON 2.1852）；2048-bit → 0.2966 vec/s，与 JSON 一致。

**Paillier 数字来源**：真实 python-phe + gmpy2 密钥，实际加密 5 个向量 × 768 = 3840 个元素后按"每元素耗时 × 768"外推到 vectors/s；JSON 的 `note` 写明 "extrapolated from n_vectors x 768 element encryptions; CPU only, no batching"，`notes/exp1_efficiency.md` 表格也写明"5 个向量外推"——**声明充分，不算把外推当实测**。

## 3. 静态审查（逐行读脚本，查硬编码/阈值/反推/样本量/种子）

| 检查项 | 结论 |
|---|---|
| 硬编码期望值 | exp5 的 `PAPER_TABLE`、exp1 的 `PAPER_*` 常量只作"论文参照/画图对照"，不参与任何"实测"输出；**我另用论文 LaTeX 源（`output/pdf/source/caprise/ppRAG_FLLM.tex` 559–562 行）逐格核对 Table III 转录，完全一致**。公式表 / MC / 吞吐数字均由代码计算，未见写死的结果值。 |
| 阈值过松 | M13 MC 容差 5%（实测 max 误差 2.38%）；M17 recheck 容差 [0.33,3]（实测 ratio 1.01）；M12 均匀性 var 容差 0.02（估计标准差 ~7e-4）。均有"松但实测远在窗口内"的特征，未掩盖坏数据；建议收紧 recheck 窗口（V5）。 |
| 用论文值反推当"实测" | 反推 Δα/r 被明确标注为 **reverse-engineered / finding A10（分析）**，没有当作实测；Paillier 是真实测量后外推且已声明；Phase-1 嵌入是本地真测。**未发现把论文数字冒充复现实测**。 |
| 样本量/种子 | exp5 MC：m=100,000、每维 nq=200、`DB_SEED=20260118` + MC `seed=7`，逐 query 的 `min_kprime_samples` 全部落盘；exp1：batch=1000、reps=3、种子固定（SEED=20260118 / rng 1000+d）；M12：5000 draws 固定种子。**未发现偷偷缩小样本或跳过种子**。 |
| 结果与 notes 一致性 | M13 notes 的反推表（1.1927/1.2298 等）、A10 的 0.444–0.737 倍、review 的 1.2129±0.0435 均与 JSON 一致。exp1 notes 的吞吐/Phase-1/Paillier 表述与 JSON 一致（除 V1 那条 findings 文本）。 |

## 4. 发现的问题（均列为"建议修"，不影响数据与结论；未修改对方任何文件）

| ID | 严重度 | 位置 | 问题 | 影响 |
|---|---|---|---|---|
| **V1** | minor（表述一致性） | `results/exp1_efficiency.json → findings[A17c]` | 文本写死 "~224 ms (MPS) / ~606 ms (CPU) / ~1.6 ms / overhead 0.26–0.71%"，与同一 JSON 的实测中位数（212.8 / 632.105 / 1.471 ms；overhead 0.233–0.691%）**不一致**（疑似早前一轮运行的旧数字，未用格式化从原始数据生成） | 结论（嵌入是瓶颈、加密 <1%）不受影响；但**建议修掉**，否则报告"独立验证"一节会引用到自相矛盾的数字 |
| **V2** | minor（指标口径） | `results/exp1_efficiency.json → findings[A17]` | 用 86,492 emb/s（批量 EncDB 吞吐）对比论文 2,339 vec/s（Phase-2 单向量 EncQ）说"~37×"；同 JSON 的同口径对比是 78,566.8/2,339 = **33.6×** | 两个数方向一致；报告里应用 33.6× 或注明 batch-vs-single 口径差异 |
| **V3** | minor（命名/推断） | `experiments/exp1_efficiency.py → style_probe()` | JSON key `per_element_scalar_loop_us`=98.334 实际是 **768 次标量 draw 的每向量耗时**（1e6/98.334=10,169 vec/s 说明按向量用）；且"~0.5–0.7 µs/element 复现论文量级"是推断（论文 428µs/768=0.56µs/element），而实测标量 draw 只有 ~0.128 µs/element | A17（论文吞吐是逐元素实现风格）**仍是合理假设而非测量**，建议改措辞并重命名字段 |
| **V4** | minor（透明性） | `experiments/exp1_efficiency.py → builtin_pow_paillier()` | 纯 CPython "Paillier" 下界基线用的是**合成模数**（无素数性要求，仅测模幂成本），只写在代码 docstring，未写进 JSON/notes | 只是下界基线；主 Paillier 数字用真实 python-phe 密钥且已声明外推；建议 notes 补一句 |
| **V5** | minor（门禁设计） | `gate_m17_efficiency.py` / `gate_m13_theorem1.py` | M17 recheck 接受 [0.33,3] 很宽；M13 表转录检查允许 k'/k 四舍五入差 1；M13 的 "independent reduced MC" 复用 exp5 模块（非实现独立） | 实测值远在窗口内（recheck ratio 1.01、转录与 LaTeX 完全一致、我的独立 MC −1.99%），**没有掩盖坏数据**；建议收紧/补充 |
| **V6** | note | `results/exp5_query_dp.json → reverse_delta_alpha` | Δα 存 8 位、Δα/r 存 5 位小数；对 Δα/r 用 1e-6 容差低于 JSON 舍入粒度 | 独立复算差 ≤4.94e-9 rad / ≤4.36e-6（=舍入），无数值分歧 |

## 5. 总评

**PASS**：M12 / M13 / M17 门禁独立重跑全部 PASS；Table III 公式表 16 格独立复算逐位一致；反推 Δα/r（1.21285±0.0435）与 JSON 一致（finding A10 成立）；独立 MC（m=20000, nq=20）验证角帽点数与公式偏差 −1.99%（<10% 要求）；exp1 吞吐 d=768 手算 86,490.2 emb/s（JSON 86,491.5）、Phase-1 overhead 0.6913%（JSON 0.691%）均可从原始 runs 复算；Paillier 为真实测量 3840 元素后外推且已声明。

- **影响测量值/结论的问题：0 个**；
- 发现 5 个表述/一致性问题（V1–V5）+ 1 个舍入说明（V6），建议在最终报告前修 V1/V2（最容易被报告引用到）。

复现命令（本次验收）：
```bash
cd "code/CAPRISE复现"
.venv/bin/python -m gates.run_gates --from M12 --to M13
.venv/bin/python -m gates.run_gates --from M17 --to M17
# 独立复算见 results/verify_theory.json 的 independent_* 段（scipy betainc/betaincinv/brentq + 自写 MC，
# seed=424242；未 import 被验收脚本）
```
