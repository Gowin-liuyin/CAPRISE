# 论文声明 vs 复现证据 核对清单

> 逐条核对论文中每一个可检验的声明（claim），记录复现证据、是否成立、偏差原因。
> 状态: ✅ 成立 / ⚠️ 部分成立（有偏差） / ❌ 不成立 / ⏳ 待实验 / ➖ 无法验证（超出范围）
>
> 证据字段指向 `results/` 下的具体文件或 `gates/` 门禁。

## A. 方案正确性声明

| # | 论文声明 | 位置 | 证据 | 状态 |
|---|---|---|---|---|
| C1 | `DecDB(EncDB(e)) = e`（可逆） | Alg.1 | gates/gate_M4_encdb.json, gate_M5_decdb.json; 最大误差 2.8e-17 | ✅ |
| C2 | `‖η_eq‖ < sβ/8`，`‖λ_ei‖ < 3sβ/8`（噪声有界） | Eq.(7) | gate_M3_noise.json, M4, M6：严格小于，实测 max/R = 1.00000 | ✅ |
| C3 | Eq.(2) 距离比较保留：gap > β ⇒ 密文顺序保留 | Eq.(2) | gate_M7_eq2.json：50,000 组前提构造三元组 **0 违反**；边界 gap=β+1e-6 也 0 违反 | ✅ |
| C4 | Eq.(3) 库内距离比较**不**保留 | Eq.(3) | gate_M8_eq3.json：合法对抗噪声可实现反转（命题为真）；但真实采样下 **0/20000** 反转（→ finding A19） | ⚠️ 数学成立、实际不可达 |
| C5 | ADCPE 基线满足 β-DCP | ADCPE Sec.5.1 | gate_M9_adcpe.json：20,000 组 0 违反 | ✅ |
| C6 | CAPRISE 噪声使库内距离顺序被打乱 ⇒ 防向量分析 | §IV-1 | gate_M8_eq3.json：距离重建误差 CAPRISE 0.0046 vs ADCPE 0.0025（1.8×） | ⚠️ 有差异但幅度有限（详见 exp4） |
| C7 | Theorem 1 公式正确（检索集扩张） | Eq.(21) | exp5: 公式 vs Monte Carlo | ⏳ M13 |
| C8 | 加密嵌入抗 Vec2Text 反演 | §IV-1, Table II | exp3: BLEU 基线 vs CAPRISE | ⏳ M11 |
| C9 | 方案对 honest-but-curious 服务器保密（继承 ADCPE 安全性） | §IV | 无法形式化验证；仅做泄漏经验检查（smoke test: 服务器无明文） | ➖ 声明性 |
| C10 | DistanceDP 满足 (n,ε)-DP | §IV-2 | M12: 机制实现核对（Gamma 半径/球面方向/ε↔r̄） | ⏳ M12 |

## B. 效率声明 (Fig.4, §V-1)

| # | 论文声明 | 证据 | 状态 |
|---|---|---|---|
| E1 | 加密吞吐（embeddings/s）随嵌入维度增大而下降 | exp1 吞吐 vs d | ⏳ M17 |
| E2 | 元素级吞吐在 1536–3072 维保持高位 | exp1 element/s | ⏳ M17 |
| E3 | Phase-1: 128 query 嵌入 79.52ms + CAPRISE 加密 15ms，开销 <19% | exp1 overhead（CPU 重测） | ⏳ M17 |
| E4 | Phase-2: d=768 时 CAPRISE 2339 vec/s，RemoteRAG PHE 250–300 vec/s，9× | exp1 CAPRISE vs Paillier 微基准 | ⏳ M17 |

## C. 隐私实验声明 (Table II, Fig.5)

| # | 论文声明 | 论文数值 | 证据 | 状态 |
|---|---|---|---|---|
| P1 | 明文嵌入可被 Vec2Text 高保真重建 | BLEU 83.021 / P .947 / R .950 / F1 .948 / ROUGE .950 | exp3 | ⏳ M11 |
| P2 | CAPRISE 加密后重建质量显著下降 | BLEU 12.425 / P .482 / R .498 / F1 .487 / ROUGE .492 | exp3 | ⏳ M11 |
| P3 | 向量分析 ASR：CAPRISE 一致低于 ADCPE | Fig.5（无数值） | exp4: ASR@t / Kendall τ / 距离误差 | ⏳ M10（注意 finding A19 的预期修正） |
| P4 | 隐私-精度 trade-off：r 越大 k' 越大、隐私越强精度越低 | Table III + exp2 | exp5 + exp2 | ⏳ M13/M18 |

## D. 检索质量声明（论文未给数字）

| # | 论文声明 | 证据 | 状态 |
|---|---|---|---|
| R1 | ppRAG 保持 "high retrieval accuracy" | exp2: plaintext vs CAPRISE vs CAPRISE+DP vs oversampling 的 Recall@10/MRR | ⏳ M18 |
| R2 | 检索可行（top-k' 密文搜索 + 本地解密重排） | smoke test + gate M15; 无 DP 时 k'=k 下 recall@10 ≈0.87（加密噪声导致），oversampling 后 ≈1.0 | ⏳ M15/M18 |

## E. 复现过程中发现的论文内部问题（findings）

| # | 问题 | 证据 | 影响 |
|---|---|---|---|
| F1 (A10) | Table III 的 top-k' 无法用正文/RemoteRAG 的 `Δα≈r` 复现；反推 `Δα≈1.2r` | exp5 notes | 论文表值与公式/约定不一致 |
| F2 (A19) | Eq.(3) 只具"存在性"：高维下噪声方向需跨向量对齐（概率 e^{-Ω(d)}），实测 0 反转；论文 Fig.5 的 CAPRISE 优势需用距离重建误差/ASR 的实际差值衡量 | gate_M8_eq3.json, exp4 | 向量分析防护的实际效果可能弱于论文表述 |
| F3 (A11) | 论文未说明 Vec2Text 攻击如何适配"加密后"嵌入（尺度/归一化） | exp3 notes | 指标依赖攻击者假设，需报告多种攻击方式 |
| F4 (A18) | 论文未给任何检索精度数字 | exp2 | "high retrieval accuracy" 不可核验，本复现补测 |
| F5 (A17) | 效率数字的实现风格敏感（2339 vec/s 对应 Python 逐元素实现；numpy 向量化快数十倍） | exp1 notes | 跨实现/硬件不可直接比较 |
| F6 | 论文 Table II 未给样本量、攻击步数、攻击者是否归一化等细节 | exp3 notes | 绝对值不可完全对齐，仅趋势可验证 |
| F7 | Algorithm 2 未说明 nonce 如何随密文返回（DecDB 必需） | SPEC A9 | 实现补充：nonce 与密文同存同传 |
| F8 | 「高维各向同性随机向量无法满足 gap>β 前提」（pair-distance std≈0.025 vs β=0.2） | gate M7/M8 构建说明 | 实验必须用真实（聚类）嵌入或构造式三元组，否则会误判性质 |

## 统计口径说明

- Recall@10：检索结果与**明文 top-10** 的重合比例（衡量加密/DP 带来的检索损失，不依赖 qrels）
- qrels Recall/MRR：相对 MS MARCO dev qrels（衡量端到端质量，含嵌入模型本身的上限）
- ASR@t：密文 top-t 与明文 top-t 的重合比例；Kendall τ / Spearman ρ 作为全序一致性辅助
- Vec2Text：BLEU 0–100（sacrebleu），token P/R/F1，ROUGE-L F1；均与论文口径对齐
