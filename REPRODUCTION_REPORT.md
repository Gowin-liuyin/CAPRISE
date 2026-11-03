# CAPRISE / ppRAG 复现报告

> 论文：*Efficient Privacy-Preserving Retrieval Augmented Generation with
> Distance-Preserving Encryption*（arXiv:2601.12331v1, 2026-01-18，
> Huanyi Ye, Jiale Guo, Ziyao Liu, Kwok-Yan Lam，NTU / DTC Singapore）
>
> 复现方式：**逐行分析论文 → 逐协议、逐模块实现 → 每个模块门禁 PASS 才进入下一模块**
> （门禁定义见 `docs/MODULE_PLAN.md`，逐行分析见 `docs/PAPER_ANALYSIS.md`）。
>
> 复现环境：Apple M5（10 核，16 GB，无 GPU），Python 3.12，torch 2.15 CPU。
> 论文环境：NVIDIA A100 GPU。效率数字跨硬件不可直接比较，报告中逐项标注。

---

## 0. 结论摘要（TL;DR）

（待实验完成后填写：复现了哪些声明、哪些成立、哪些有偏差、关键发现）

---

## 1. 复现范围与方法

### 1.1 复现范围

| 论文内容 | 是否复现 | 说明 |
|---|---|---|
| Algorithm 1: CAPRISE (KeyGen/EncDB/DecDB/EncQ) | ✅ 完整 | 逐行实现 + M1–M6 门禁 |
| 性质 Eq.(2) 距离比较保留 | ✅ | M7 门禁（构造式前提，50k 三元组） |
| 性质 Eq.(3) 库内不保留 | ✅（含修正） | M8 门禁 + finding A19 |
| ADCPE (SAP) 基线 | ✅ | M9 门禁 |
| DistanceDP（RemoteRAG 机制） | ✅ | M12 门禁 |
| Theorem 1 (Eq.21) | ✅ | M13 + Monte Carlo |
| Algorithm 2: ppRAG Phase 1/2/3 | ✅（Phase 3 仅 prompt 组装） | M14–M16 门禁 |
| Fig.4 效率 | ✅（CPU 重测） | M17 |
| Table II Vec2Text | ✅（小规模 CPU） | M11 |
| Fig.5 向量分析 ASR | ✅（定义补全） | M10 |
| Table III top-k' 扩张 | ✅（发现内部不一致） | M13 |
| 检索精度（论文未给数字） | ✅ 补充实验 | M18 |
| 安全性形式化证明（IND 类） | ➖ 超出范围 | 需密码学证明，复现只做经验泄漏检查 |
| LLM 生成质量 | ➖ 超出范围 | 论文未指定 LLM，Phase 3 只验证 prompt 组装 |

### 1.2 方法：逐模块门禁（gate）

每个模块有可执行的验收门禁（`gates/gate_<id>_*.py`），产物 `results/gate_<id>_*.json`
记录每项检查的 expected/actual/是否通过/硬件/时间戳。运行器
`python -m gates.run_gates --from M1 --to M9` 按序执行并在第一个失败处停止。
**任何模块门禁不通过，不进入下游模块。**

### 1.3 论文逐行分析

见 `docs/PAPER_ANALYSIS.md`：包含 Algorithm 1 的 31 行逐行表、证明 Eq.(5)–(20) 逐式核对、
Theorem 1 逐项、Algorithm 2 三阶段逐行、实验设置逐项，以及 19 项歧义/不一致决策
（编号 A1–A19）。

---

## 2. 核心协议复现（M1–M9，全部 PASS）

### 2.1 CAPRISE 实现要点

（公式逐行对应、实例化约定：PRF=HMAC-SHA256 计数器模式、k=ℓ=128bit、
Gaussian 方向由 PCG64(128bit PRF seed) 生成、u=(128bit+1)/2^128、
keyspace 整数 s、实验固定 s=3、β=0.2、嵌入 L2 归一化）

### 2.2 门禁结果表

| 模块 | 检查 | 结果 |
|---|---|---|
| M1 PRF | 确定性/密钥敏感性/输出长度/高斯与均匀 KS | ✅ PASS（KS p=0.35/0.74） |
| M2 KeyGen | s/K 合法性、参数校验、密钥唯一性 | ✅ PASS |
| M3 噪声 | 模长严格 <R、球内均匀 KS、方向均匀 | ✅ PASS（max/R=1.0000, KS p=0.63） |
| M4 EncDB | 往返 2.8e-17、噪声界严格、批量=单条 | ✅ PASS |
| M5 DecDB | 精确逆、错误 nonce 不可恢复 | ✅ PASS |
| M6 EncQ | 往返 1.4e-17、噪声界 sβ/8 | ✅ PASS |
| M7 Eq.(2) | 50,000 组前提三元组 0 反转（含边界 gap=β+1e-6） | ✅ PASS |
| M8 Eq.(3) | 合法对抗噪声可反转（命题真）；实测 0/20000（见 finding） | ✅ PASS |
| M9 ADCPE | 往返、sβ/4 界、20,000 组 β-DCP 0 违反 | ✅ PASS |

### 2.3 复现发现

（A19 等，详见各节与 `docs/CLAIMS_CHECKLIST.md`）

---

## 3. 检索流程复现（M14–M16）与检索精度（M18）

（待完成：Phase1/2/3 门禁结果、MS MARCO 上的 Recall@10/MRR 表、β 与 oversampling 敏感性、
耗时对照）

---

## 4. 效率实验（Fig.4 / M17）

（待完成：吞吐 vs 维度、Phase-1 overhead、Phase-2 CAPRISE vs Paillier PHE）

---

## 5. 隐私实验（M10、M11）

### 5.1 Vec2Text 重建攻击（Table II / M11）

（待完成：与论文数值对照表、攻击适配说明、CPU 规模说明）

### 5.2 向量分析攻击 ASR（Fig.5 / M10）

（待完成：ASR@t、Kendall τ、距离重建误差；finding A19 的量化）

---

## 6. Theorem 1 与查询 DP（M12、M13）

（待完成：Table III 复现与内部不一致发现、Monte Carlo 验证）

---

## 7. 论文声明核对总表

见 `docs/CLAIMS_CHECKLIST.md`（C1–C10、E1–E4、P1–P4、R1–R2、F1–F8）。

---

## 8. 复现过程中发现的论文问题

| # | 问题 | 类型 | 影响 |
|---|---|---|---|
| F1 | Table III 的 top-k' 与正文 `Δα≈r` 约定不一致（需 ≈1.2r） | 数值不一致 | 表值不可由公式复现 |
| F2 | Eq.(3) 仅存在性；高维实际反转率 ≈0（概率 e^{-Ω(d)}） | 声明过强/解释问题 | 向量分析防护效果需重新表述 |
| F3 | Vec2Text 攻击如何适配密文未说明 | 可复现性缺口 | 指标依赖攻击者假设 |
| F4 | 无检索精度数字 | 可复现性缺口 | 本复现补测 |
| F5 | 效率数字实现风格敏感 | 可复现性缺口 | 跨实现不可比 |
| F6 | Table II 缺样本量/步数/归一化细节 | 可复现性缺口 | 仅趋势可验证 |
| F7 | nonce 未说明随密文返回（DecDB 必需） | 规范缺口 | 实现补充 |
| F8 | 高维各向同性向量无法满足 gap>β 前提 | 实验设计注意 | 性质实验需构造式三元组 |

---

## 9. 局限与偏差

- 硬件：CPU vs A100 → 效率绝对值不可比，只比较同环境相对量级
- 规模：MS MARCO 子集（m=5万~10万）而非全量；Vec2Text 用 10–20 条 passage
- Vec2Text：CPU 上降低 num_steps；攻击者适配方式为复现所加假设
- 检索精度：论文未给数字，本复现的补充实验不能与论文数值对齐
- 安全证明：未重做 IND 类形式化证明，只做经验泄漏检查

---

## 10. 复跑方式

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
python -m gates.run_gates --all          # 全部模块门禁（失败即停）
bash scripts/run_all.sh                  # 数据 + 门禁 + 全部实验
```

产物：`results/*.json`（机器可读）、`figures/*.png`、`notes/*.md`、各实验日志。
