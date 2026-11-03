# exp3 — Vec2Text 重建攻击（论文 Table II / M11）

> 论文: *Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving
> Encryption* (arXiv:2601.12331v1) §V-2 Privacy（Table II）。
> 实现: `experiments/exp3_vec2text.py`；结果: `results/exp3_vec2text.json` / `.csv` /
> `.npz`；图: `figures/fig_vec2text.png`；门禁: `gates/gate_m11_vec2text.py`
> → `results/gate_M11_vec2text.json`。
> 运行环境: `.venv-vec2text`（transformers 4.x；主 `.venv` 的 transformers 5.x 与 vec2text 0.0.13 不兼容），
> 设备 MPS，本地离线模型快照（`models/vec2text`、`~/.cache/huggingface`）。
> 复跑: `.venv-vec2text/bin/python experiments/exp3_vec2text.py --phase all`（主实验约 40 min +
> 敏感性；`--metrics-only` 可只重算指标/CI 而不跑 vec2text）。

## 0. 结论速览

1. **50 条真实 MS MARCO passage × 3 种攻击模式**（论文未给样本量，A11/F6）：
   明文 BLEU **69.875** [61.99, 78.17]；CAPRISE raw（论文口径）BLEU **13.696** [9.72, 17.48]；
   尺度还原最强攻击 BLEU **34.584** [27.91, 40.91]（95% bootstrap CI，1000 次重采样）。
2. **论文 Table II 的加密后水平在 CI 内复现**：raw 13.70 vs 论文 12.425（+1.27，1.10×），
   P/R/F1（0.534/0.529/0.531 vs 0.482/0.498/0.487）同一水平；CI 覆盖论文值。
3. **明文基线 69.9 vs 论文 83.0（0.84×）**：同量级；差距主要来自 32-token 参考长度与论文
   未给的 Vec2Text 设置（F6/F15），不主张绝对值完全对齐。
4. **攻击口径很重要（A11/F3）**：若攻击者做尺度还原（把密文范数还原到明文范数），
   BLEU 从 13.7 回升到 34.6（仍比明文低 51%）——CAPRISE 的 Vec2Text 防护相当程度上依赖
   "攻击者不还原尺度"这一未声明假设；本复现同时报告两种口径。
5. **步数/beam 敏感性（F6）**：同一 5 条 passage 用 num_steps=50、beam=10 重跑，
   与 20/4 的差值见 §7（用于界定"论文未给设置"能解释多少偏差）。

---

## 1. 论文原文与复现映射

> "Given an input document $x_i$, its embedding $e_i$ can be inverted to reconstruct a text
> approximation $\hat{x}_i$ using Vec2Text. We measure the semantic similarity … using metrics:
> BLEU, Precision, Recall, F1, and ROUGE. … under the secure parameter setting of $s=3$, $\beta=0.2$,
> CAPRISE significantly reduces the reconstructability of encrypted embeddings. Notably, the BLEU
> score drops from over 83.021 on plain embeddings to 12.425 on encrypted ones…"

论文 Table II：w/o Enc `83.021 / 0.947 / 0.950 / 0.948 / 0.950`；CAPRISE `12.425 / 0.482 / 0.498 / 0.487 / 0.492`。
论文**未给**样本量、样本长度、Vec2Text 的 steps/beam、以及攻击者如何把"加密后的嵌入"喂给
反演模型（F6/A11）。

复现设置（全部写入 JSON `setup` / `passages` / `encryption`）：
- 数据：`data/passages.jsonl`（真实 MS MARCO passage）；取 token 数 ≤32 的前 **50** 条
  （扫描 3207 行，pool 自动扩展）；**全文**作为 BLEU/P/R/F1/ROUGE 的 reference（不截断）。
  32-token 限制来自 corrector 的生成预算（`jxm/gtr__nq__32`）。
- 嵌入：`gtr-t5-base`；攻击输入使用 vec2text 训练约定（T5 encoder 均值池化、无 Dense、无 L2 归一化；
  范数 0.929–1.568，均值 1.189）——见 §2 F14。
- 加密：CAPRISE Alg.1 EncDB，`s=3, β=0.2, d=768`；库侧噪声半径 0.225，实测 ‖λ‖ ∈ [0.2238, 0.2250]；
  密文范数 2.792–4.717（≈ s‖E‖）。
- 反演：vec2text 0.0.13 gtr-base inversion + corrector，`num_steps=20, sequence_beam_width=4`，
  chunk=4；主实验 50×3 共 150 次反演，MPS 上 16.35 s/text（共享机负载下 13–21 s/text）。
- 指标：BLEU（sacrebleu，0–100，effective_order）、token 级 micro P/R/F1（0–1）、
  ROUGE-L F1（0–1）；passage 级 bootstrap 95% CI（1000 次，seed=20260101，三种模式共享重采样索引）。

## 2. 攻击嵌入约定（F14，关键）

Vec2Text 的 gtr-base 反演器是在 **T5 encoder 均值池化**的嵌入上训练的（无 sentence-transformers
Dense 层、无 L2 归一化）。若直接把 sentence-transformers 的 GTR 嵌入（含 Dense + 归一化）喂进去，
**即使不加密也会重建出完全无关的文本**——这样得到的"基线"没有意义。因此本实验的攻击输入用
`corrector.inversion_trainer.call_embedding_model`（vec2text 自身约定），M11 门禁用 npz 数组对
嵌入约定、加密链（E' = 3E + λ）与 rescaled 攻击输入做了字节级审计（sha256 对账）。

## 3. 主表（50 条 passage，95% bootstrap CI）与论文对照

| 条件 | BLEU [95% CI] | Precision | Recall | F1 | ROUGE-L | vs 论文 |
|---|---|---|---|---|---|---|
| 论文 w/o Enc | 83.021 | 0.947 | 0.950 | 0.948 | 0.950 | — |
| 论文 CAPRISE | 12.425 | 0.482 | 0.498 | 0.487 | 0.492 | — |
| **本复现 明文（w/o Enc）** | **69.875** [61.99, 78.17] | 0.924 | 0.928 | 0.926 | 0.860 | BLEU 0.84×（−13.15）；P/R/F1/Rouge 0.90–0.98× |
| **本复现 CAPRISE raw（论文口径）** | **13.696** [9.72, 17.48] | 0.534 | 0.529 | 0.531 | 0.409 | BLEU **1.10×（+1.27，CI 覆盖论文值）**；P/R/F1 1.06–1.11×；ROUGE 0.83× |
| 本复现 CAPRISE rescaled（最强攻击） | 34.584 [27.91, 40.91] | 0.723 | 0.760 | 0.741 | 0.651 | 比论文 CAPRISE 行 +22.16 BLEU |

逐条 passage 结果与 per-passage BLEU 见 `results/exp3_vec2text.csv`；图 `figures/fig_vec2text.png`
含 (a) 三模式 BLEU + CI vs 论文参考线、(b) P/R/F1/ROUGE、(c) 逐 passage BLEU 分布、
(d) 20/4 vs 50/10 敏感性。

## 4. 与论文的偏差分析

- **CAPRISE 行**：+1.27 BLEU（1.10×），CI 覆盖 12.425，P/R/F1 同水平；10 条小样本时曾得到 9.26，
  说明小样本方差约 ±4 BLEU（50 条的 CI 半宽 3.9）。**论文 Table II 的加密后水平可复现**。
- **明文行**：69.9 vs 83.0（−13.15，0.84×）。可能来源：(a) 参考长度为 ≤32 token（corrector 预算，
  论文未说明其长度协议）；(b) 论文未给 num_steps/beam；(c) A100 vs M5 MPS 的数值差异可忽略（贪心/beam 搜索
  是确定性的）。因此只主张**量级与形态一致**（F6/F15）。
- **ROUGE**：论文未说 ROUGE 变体；本复现用 ROUGE-L F1（stemming），明文 0.860 vs 论文 0.950，
  CAPRISE 0.409 vs 0.492——同方向的下降，绝对值受变体影响（A11/F6）。

## 5. 攻击者假设的量化（A11/F3，重要安全限定）

三种攻击模式：

| 模式 | 攻击输入 | 语义 | BLEU |
|---|---|---|---|
| plaintext | E（vec2text 约定嵌入） | 论文 w/o Enc 行 | 69.875 |
| caprise_raw | E' = sE + λ（直接反演） | 论文最可能的口径（攻击者不知道 s） | 13.696 |
| caprise_rescaled | E'·‖E‖/‖E'‖（还原范数） | 攻击者用某种方式恢复原始尺度（最强上界） | 34.584 |

rescaled 攻击把 BLEU 从 13.7 拉回 34.6（仍比明文低 51%）：**CAPRISE 对 Vec2Text 的防护有相当部分
来自"密文范数 ≈ s‖E‖ 落在反演器输入分布之外"**，论文没有说明其 Table II 采用哪种攻击口径（A11）。
本复现把 raw 作为与论文对照的主行，rescaled 作为隐私上界同时报告（F3）。

## 6. 复现性与审计

- 每个模式的 hypotheses/references 与逐 passage 指标都保存在 JSON/CSV；`--metrics-only` 可在**不跑
  vec2text** 的环境下重算全部指标与 CI（M11 门禁即用它做独立复算，max diff = 0）。
- `results/exp3_vec2text_arrays.npz` 保存 E、E'、E_rescaled、references 与 sha256；门禁验证
  `‖E' − sE‖ < 3sβ/8`、rescaled 与 raw 的关系、以及 JSON 与 npz 的一致性。
- nonce 采用确定性派生（记录 sha256），保证加密链可复现；vec2text 反演本身是确定性的。

## 7. 步数 / beam 敏感性（F6/A11）

同一 **5 条** passage（主实验前 5 条的子集）在三种模式下用 `num_steps=50, sequence_beam_width=10`
重跑（batch=1；共享机/16 GB 内存限制，约 108 s/text）。20/4 基线在同一 5 条上从主实验重新计算。

| 模式 | 20/4（同 5 条基线） | 50/10（敏感性） | Δ BLEU | Δ F1 |
|---|---|---|---|---|
| plaintext | 55.724 | **77.411** | **+21.687** | +0.023 |
| caprise_raw（论文口径） | 7.210 | **7.156** | **−0.054** | +0.051 |
| caprise_rescaled（最强攻击） | 29.728 | 33.014 | +3.286 | +0.014 |

- 5 条 passage：主实验前 5 条（indices 0–4；pids 85696, 3489272, 7071501, 7075317, 7079676）。
- 计算预算：50/10 + batch=1 在共享机上约 129 s/text，15 次反演共 1937.6 s（内存限制见 §8）。
- **判读**：
  1. **明文行对 steps/beam 高度敏感**（+21.7 BLEU：55.7→77.4）——这解释了论文明文 83.0 与本复现
     50 条 20/4 的 69.9 之间的部分差距：论文很可能用了更大的 steps/beam（F6 未声明）。
  2. **CAPRISE raw 行几乎不动**（−0.05 BLEU：7.21→7.16）——加密后的重建质量已经塌到底，
     提高反演预算不能恢复，说明"CAPRISE 显著降低可重建性"的结论**不依赖**未声明的 steps/beam。
  3. rescaled 攻击 +3.29（29.7→33.0）：最强攻击者从更大预算中获益更多，但仍比明文低约 57%。
- 因此 Table II 的**趋势与加密后水平**是稳健的；明文基线的绝对值应视为"给定反演预算"下的值。

## 8. 限制与口径

1. **样本量**：50 条（论文未给）；样本为 token≤32 的真实 passage，可能比论文的样本短。
2. **参考长度**：32-token 限制是 corrector 的生成预算；论文未说明是否同样限制（F6）。
3. **攻击口径**：论文未说明如何适配密文（A11）；raw 与 rescaled 两口径都报告。
4. **硬件**：M5 MPS vs 论文 A100；贪心/beam 反演对硬件不敏感，但吞吐差 ~2–5×。
5. **敏感性样本仅 5 条**：只用于方向/量级，不用于绝对对齐；原因（机器内存/时间）已记录。
6. **ROUGE 变体**：论文未指定，本复现用 ROUGE-L F1。

## 9. 产物

- `results/exp3_vec2text.json`（setup/passages/embeddings/encryption/modes+bootstrap CI/
  sensitivity/paper_table_ii/comparison/findings/summary/runtime）
- `results/exp3_vec2text.csv`（逐 passage 行 + 带 CI 的 aggregate 行 + 敏感性行）
- `results/exp3_vec2text_arrays.npz`（E、E'、E_rescaled、references；sha256 入库）
- `figures/fig_vec2text.png`（四面板，见 §3）
- `gates/gate_m11_vec2text.py` → `results/gate_M11_vec2text.json`
