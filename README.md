# CAPRISE / ppRAG 复现

> 论文: **Efficient Privacy-Preserving Retrieval Augmented Generation with
> Distance-Preserving Encryption** — arXiv:2601.12331v1 (2026-01-18)
> Huanyi Ye, Jiale Guo, Ziyao Liu, Kwok-Yan Lam (NTU / DTC Singapore)

本仓库是对该论文的**独立复现**（论文未公开代码）。复现遵循严格的
**"逐行分析论文 → 逐协议、逐模块实现 → 门禁通过才进入下一模块"** 流程：

- 论文逐行分析: [`docs/PAPER_ANALYSIS.md`](docs/PAPER_ANALYSIS.md)
- 模块/门禁计划: [`docs/MODULE_PLAN.md`](docs/MODULE_PLAN.md)
- 实现规范（公式、约定、歧义决策）: [`SPEC.md`](SPEC.md)
- 复现报告（含论文对照、偏差与发现）: `REPRODUCTION_REPORT.md`（收尾阶段生成）

## 复现内容概览

| 组件 | 论文对应 | 状态 |
|---|---|---|
| CAPRISE 加密方案 (KeyGen/EncDB/DecDB/EncQ) | Algorithm 1 | ✅ M1–M6 PASS |
| ADCPE (SAP) 基线 | Fuchsbauer et al. 2022 | ✅ M9 PASS |
| 距离比较性质 (Eq.2 / Eq.3) 验证 | §III-B1 | ✅ M7–M8 PASS（M8 含 finding F2） |
| DistanceDP + Theorem 1 | §III-B2, Eq.21 | ✅ M12–M13 PASS（含 finding F1/F9） |
| ppRAG 三阶段流程 | Algorithm 2 | ✅ M14–M16 PASS |
| 数据: MS MARCO 子集 + gtr-t5-base | §V setup | ✅ M0 PASS（m=50,000，1000 queries） |
| 效率实验 (Fig.4) | §V-1 | ✅ M17 PASS |
| 检索精度（补充） | §V（论文未给数字） | ✅ M18 PASS（CAPRISE k'=k recall@10=0.9699） |
| Vec2Text 攻击 (Table II, 50 条 + 95% CI) | §V-2 | ✅ M11 PASS |
| 向量分析 ASR (Fig.5) | §V-3 | ✅ M10 PASS（论文图数字化对照） |
| top-k' 扩张表 (Table III) | §V-4 | ✅ M13 PASS（论文表值与公式不一致） |

## 复现结果速览（全部模块门禁 PASS：`python -m gates.run_gates --all`）

| 实验 | 论文 | 本复现 | 判定 |
|---|---|---|---|
| CAPRISE 加解密正确性 | Alg.1 | 往返 2.8e-17；噪声界严格；Eq.(2) 50,000 组 0 违反 | ✅ |
| Eq.(3) 库内不保留 | §III-B1 | 存在性成立（构造反例）；真实采样 0/20,000 反转 → finding F2 | ⚠️ |
| 检索精度（论文未给） | §V | CAPRISE k'=k Recall@10=0.9699；k'=15→0.9999；qrels MRR 与明文相同 0.7230 | ✅ 补充 |
| Table III top-k' | Table III | 四约定枚举：论文/公式(Δα=r)=1.36–2.25；只有未声明的逐格反解 Δα/r=1.213±0.043 能复现；RemoteRAG 参照实现自洽于 Δα=r → 论文内部不一致 F1 | ❌ FALSIFIED |
| Theorem 1 | Eq.21 | 公式 vs MC 偏差 ≤2.38%；实测最小 k'≈k → 定理为宽松上界 F9 | ❌ FALSIFIED |
| Vec2Text (Table II) | 83.02→12.43 BLEU | 50 条 + 95% CI：明文 **69.88** [61.99,78.17] → CAPRISE raw **13.70** [9.72,17.48]（CI 覆盖论文 12.43）；rescaled 强攻击 34.58 | ✅ 加密后水平在 CI 内复现 |
| 向量分析 ASR (Fig.5) | CAPRISE 更低 | 27/27 cell 更低；真实数据 −2.3pt；距离重建误差 2.16× | ✅ 趋势成立 |
| 效率 (Fig.4) | 15 维曲线、2339 vec/s、<19% overhead | 论文 15 维全部实测；paper-style 探针 1.18–2.38×（中位 1.41×）、drop 42.1× vs 论文 44.5×；向量化 43,468 vec/s（18.6×）；overhead 0.19–0.27% | ✅ 趋势+量级（A17/F5） |
| ppRAG 三阶段 | Alg.2 | 服务器无密钥/无明文；只收 e'_q；prompt 0 泄漏 | ✅ |

**论文问题清单（F1–F16，详见 docs/FINDINGS.md）**：两个 FALSIFIED（Table III 内部不一致、
Theorem 1 过强）、一个证明符号笔误（F16）、若干实验设计/攻击模型缺失（Vec2Text 适配、β 校准、
Fig.5 维度配置、Fig.4 实现风格）、其余为实现细节缺失（PRF/keyspace/nonce/ReRank/AES/LLM 等 19 项歧义）。
逐要素对齐见 [`docs/PAPER_COVERAGE.md`](docs/PAPER_COVERAGE.md)。

## 环境

- Python 3.12，依赖见 [`requirements.txt`](requirements.txt)
- 无 GPU，全部实验在 Apple M5 CPU（10 核 / 16 GB）上完成；与论文 A100 的差异在每个实验中记录
- 首次运行会下载 MS MARCO 子集与 gtr-t5-base（约 1–2 GB），缓存放在仓库内 `.cache/`

```bash
# 1) 环境
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2) 各模块门禁（按顺序；失败即停）
.venv/bin/python gates/run_gates.py --from M1 --to M9

# 3) 数据准备
env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/prepare_data.py --m 50000

# 4) 实验
.venv/bin/python experiments/exp1_efficiency.py
.venv/bin/python experiments/exp2_retrieval.py
.venv/bin/python experiments/exp3_vec2text.py
.venv/bin/python experiments/exp4_vector_analysis.py
.venv/bin/python experiments/exp5_query_dp.py

# 5) 一键全流程（自动检查门禁顺序）
bash scripts/run_all.sh
```

## 目录结构

```
src/caprise/     核心实现（PRF、CAPRISE、ADCPE、DistanceDP、Theorem 1、ppRAG、AES、指标）
gates/           逐模块门禁脚本 + 运行器（M0…M18）
experiments/     论文实验脚本（exp1…exp5）
scripts/         数据准备、全流程脚本
docs/            论文逐行分析、模块计划
results/         门禁与实验结果 JSON（小文件，入库）
figures/         实验图
notes/           每个实验的人读说明
data/            数据产物（不入库，脚本重建）
```

## 重要复现发现

1. **Table III 内部不一致**：论文 top-k' 数值无法用被引用来源 RemoteRAG 的
   `Δα_k ≈ r` 约定复现；四种约定枚举后，只有未声明的 `Δα_k ≈ 1.2 r`（逐格反解
  1.1537–1.2977）能精确复现，而 RemoteRAG 自身高亮例却与 `Δα≈r` 自洽（+6.46%）。
  详见 `notes/exp5_query_dp.md` §4–5。
2. 论文未公开：PRF 实例化、keyspace、nonce 传输、ReRank 模型、AES 模式、
   LLM、ASR 定义、Vec2Text 对密文的攻击适配等 18 项歧义——每项的实现决策
   记录在 `SPEC.md` §2.2 与 `docs/PAPER_ANALYSIS.md` §12。
3. 论文**没有给检索精度数字**（只声称 high retrieval accuracy），本复现补充
   exp2 在 MS MARCO 上的检索精度评估。

## 引用

```bibtex
@article{ye2026caprise,
  title  = {Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption},
  author = {Ye, Huanyi and Guo, Jiale and Liu, Ziyao and Lam, Kwok-Yan},
  year   = {2026},
  eprint = {arXiv:2601.12331}
}
```
