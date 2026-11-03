# M11 — Vec2Text 重建攻击（论文 Table II）复现笔记

> 论文位置: arXiv:2601.12331v1 §V-B "Privacy" / Table II
> 代码: `experiments/exp3_vec2text.py`；门禁: `gates/gate_m11_vec2text.py`；
> 结果: `results/exp3_vec2text.json`（+ `.csv`、`_arrays.npz`）；图: `figures/fig_vec2text.png`。
> 相关歧义/发现: A11（攻击如何适配密文未说明）、F14（攻击用嵌入约定）、F6（论文未给样本量/文本长度/步数）。

## 1. 论文 Table II（原文）

| 条件 | BLEU | Precision | Recall | F1 | ROUGE |
|---|---|---|---|---|---|
| w/o Enc | 83.021 | 0.947 | 0.950 | 0.948 | 0.950 |
| **CAPRISE**（s=3, β=0.2） | **12.425** | 0.482 | 0.498 | 0.487 | 0.492 |

论文给的只是"加密前后"两行；**没有给**样本量、文本长度、Vec2Text 步数/beam、
以及"加密后向量如何喂给攻击模型"（A11）。这些都需要复现时显式决策并声明。

## 2. 攻击管线与两个关键协议决策

服务器（honest-but-curious）持有密文嵌入 $\{e'_i\}$，用 Vec2Text 反演成文本 $\hat x_i$，
再用 `src/caprise/metrics.py:text_metrics` 与原文 $x_i$ 比 BLEU(0–100)/token P/R/F1/ROUGE-L。

```
10 条真实 MS MARCO passage（data/passages.jsonl，全文作为 reference，不截断）
  → 攻击约定嵌入 E_i（F14，见 §3）
  → baseline: invert(E_i)                                  → "plaintext / w/o Enc"
  → CAPRISE s=3,β=0.2: e'_i = 3·E_i + λ_i, ||λ_i||<0.225   （核心 EncDB，float64）
  → ① invert(e'_i)                                         → "caprise_raw"（论文式预处理）
  → ② invert(e'_i·||E_i||/||e'_i||)                        → "caprise_rescaled"（最强攻击/上界）
  → 与论文 Table II 并列表
```

- **F14（关键，Lead 实测定位）**：Vec2Text 的 gtr-base corrector 是在 **vec2text 攻击约定嵌入**
  上训练的：GTR-T5-base encoder 的 `last_hidden_state` 均值池化，**没有** sentence-transformers
  的 2_Dense 投影、**没有** L2 归一化（实测范数 0.93–1.57）。M0 的 `data/passage_emb.npy` 是
  sentence-transformers 的 GTR 嵌入（Dense+归一化，单位范数），直接喂给 vec2text **即使不加密也
  重构出无关文本**，不能作为基线。本实验用
  `corrector.inversion_trainer.call_embedding_model(...)` 重建正确约定（`vec2text_t5_meanpool_v1`）。
- **A11 / 攻击适配**：论文没说密文向量如何适配。`caprise_raw` 是"原样喂密文"（范数≈3·||E||，
  远离 inversion model 的输入分布，naive 攻击，**与论文 Table II 口径最接近**）；
  `caprise_rescaled` 是攻击者用公开嵌入约定把范数还原（`e'·||E||/||e'||`，对攻击者最有利），
  **主表同时报告**，作为隐私上界（finding F3/A11）。
- **F6 / 样本长度校准**：corrector 训练/生成预算为 `max_seq_length=32`（repo `jxm/gtr__nq__32`），
  生成文本最多 ~32 token；若 reference 超过这个长度，任何攻击都不可能"完整重建"，corpus BLEU
  会被无关的截断惩罚拉垮，基线失去意义（第一版 10 条未校准样本：长度 20–103 词，
  plaintext BLEU 只有 8.58，明显不可比）。因此从 `data/passages.jsonl` 前 400 行中选
  **token 数 ≤ 32 的前 10 条 passage**（tokenizer 计数；字符长度 71–125、词数 11–20、
  token 长度 16–31），使明文攻击达到论文基线量级（BLEU 63.7 vs 论文 83.0，同一数量级）。
  **该样本校准只依据"明文攻击是否达到论文基线量级"，完全没有使用 CAPRISE 结果筛选**。

## 3. 实验设置（真实测得）

| 项 | 值 |
|---|---|
| 数据 | 真实 MS MARCO passage：`data/passages.jsonl` 前 400 行中 token≤32 的前 10 条；全文 reference |
| 攻击嵌入 | vec2text 攻击约定（T5 mean-pool，无 Dense/无归一化），范数 0.93–1.57（均值 1.17） |
| Vec2Text | vec2text 0.0.13，`jxm/gtr__nq__32` inversion + `jxm/gtr__nq__32__correct` corrector（本地 HF 快照离线加载） |
| 生成参数 | `num_steps=20, sequence_beam_width=4`，greedy beam search |
| 环境 | `.venv-vec2text`（transformers 4.57.6，CPU 兼容；主 .venv transformers 5.x 不兼容） |
| 设备/耗时 | Apple M5 MPS；corrector 加载 16.6s；反演 7.9–9.8 s/text，全实验 282 s |
| 加密 | `KeyGen(dim=768, beta=0.2, s=3)` + `EncDB`；λ 范数 0.2243–0.2250 < 0.225；‖e'‖ 2.81–4.71 |
| 可复现 | 固定种子 + 确定性 nonce；hypotheses/refs 全量存 JSON；攻击数组存 `_arrays.npz` |

## 4. 结果（N=10，s=3, β=0.2）

| 条件 | BLEU | P | R | F1 | ROUGE-L | 与论文对照 |
|---|---|---|---|---|---|---|
| plaintext（w/o Enc） | **63.727** | 0.886 | 0.927 | 0.906 | 0.801 | 论文 83.021/0.947/0.950/0.948/0.950 → 低 19.3 BLEU、同一量级 |
| **CAPRISE raw（论文口径）** | **9.258** | 0.580 | 0.530 | 0.554 | 0.442 | 论文 12.425/0.482/0.498/0.487/0.492 → **BLEU 差 −3.17，P/R/F1 非常接近（+0.03~+0.10），ROUGE −0.05** |
| CAPRISE rescaled（最强攻击） | 30.289 | 0.674 | 0.795 | 0.729 | 0.622 | 比论文 CAPRISE 行高 +17.9 BLEU：scale-normalising 攻击者可恢复相当部分重建能力（F3/A11 上界） |

逐 passage BLEU（plaintext）：61.0, 56.2, 70.8, 49.4, 37.6, 100.0, 100.0, 49.7, 8.5, 100.0
（多数接近原文；个别 8.5 的低值是单条难例，见 JSON）。

**主要结论**

1. **CAPRISE 使 Vec2Text 重建崩塌（论文核心声明成立）**：明文 BLEU 63.7 → 直接反演密文 9.3
   （−85.5%），F1 0.906 → 0.554（−39%），ROUGE 0.801 → 0.442。与论文 83.0 → 12.4 的**量级和
   指标形态都一致**：论文 CAPRISE 行的 P/R/F1(0.482/0.498/0.487) 与本复现 raw 行
   (0.580/0.530/0.554) 高度接近，BLEU 差 −3.17。
2. **A11/F3：攻击适配方式显著改变结论**。若攻击者把密文范数还原到公开的原始嵌入范数
   （`e'·||E||/||e'||`，最强攻击、对隐私最保守的假设），重建 BLEU 从 9.3 回升到 30.3
   （仍比明文低 52%），F1 0.729。即：**CAPRISE 的防护相当程度上依赖攻击者"不知道/不还原 s·E 的尺度"**；
   报告 Table II 时必须同时给出 raw 与 rescaled 两个口径，否则会高估防护（或低估）。
   论文未说明其攻击属于哪一种（A11），这是本复现的重要 finding（F3）。
3. 与论文基线差 19.3 BLEU：论文未给样本量/长度/步数（F6）、硬件为 A100；本实验 N=10、
   长度校准到 corrector 32-token 预算、MPS 20 步 beam=4。如实记录，未做数字拟合。

## 5. 局限与偏差（如实声明）

1. **样本量 N=10**（论文未给；20 条时因 MPS 上 batch=20+beam=4 异常慢且首版样本长度不可比，
   在时间盒内采用 10 条校准样本）。N=10 足以做方法趋势对比，但不适合给出稳定的绝对数值结论。
2. **样本长度校准（F6）**：按 corrector 32-token 生成预算选样本；若论文使用的是更长 passage
   或其它 Vec2Text 配置，绝对 BLEU 不可直接比较（只比较相对下降与指标形态）。
3. **攻击适配（A11）**：raw / rescaled 两个口径都报；论文未说明其口径，主表以 raw 对应（量级最接近）。
4. **CPU/MPS vs A100**：生成步数/beam 受算力限制；论文硬件 A100。
5. **非形式化安全**：本实验测重建质量，不构成形式化安全性证明。
6. 反演在 MPS 上按 chunk=4 执行（batch=20 时单模式 10+ 分钟仍无输出，疑似 MPS 大 batch beam
   search 异常慢）；chunk 不影响逐样本 greedy beam 结果，已记录在 JSON `config.batch_size`。

## 6. 复跑

```bash
# 正式实验（约 4.5 分钟；产物 json/csv/png/npz）
env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv-vec2text/bin/python experiments/exp3_vec2text.py --n 10 --batch-size 4

# 仅用已存 hypotheses 重算 metrics（主 .venv 亦可，无需 vec2text）
.venv/bin/python experiments/exp3_vec2text.py --metrics-only

# 门禁（模块 M11）
.venv/bin/python -m gates.run_gates --from M11 --to M11
```

## 7. 产物

- `experiments/exp3_vec2text.py` — 攻击管线（离线加载、chunk 反演、断点缓存、metrics-only）
- `results/exp3_vec2text.json` — 设置、加密审计、三模式 metrics/hypotheses/per_passage_bleu、论文并列表与 findings
- `results/exp3_vec2text_arrays.npz` — 攻击约定嵌入 E、CAPRISE 密文 E'、rescaled 输入（门禁审计用）
- `results/exp3_vec2text.csv` — 指标表
- `figures/fig_vec2text.png` — (a) BLEU vs 论文 Table II 参考线；(b) P/R/F1/ROUGE；(c) 逐 passage BLEU
- `gates/gate_m11_vec2text.py` + `results/gate_M11_vec2text.json` — 门禁
