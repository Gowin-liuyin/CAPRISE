# 论文声明核对表（CLAIMS_CHECKLIST）

> 判定只用: **VERIFIED / FALSIFIED / BLOCKED / UNKNOWN**。
> UNKNOWN = 复现尚未完成；BLOCKED = 超出复现范围（需形式化证明/不可获得资源），不得算作已验证。
> 每条声明必须有可复跑证据路径；论文值逐项对照，偏差必须量化。
> 细节分析: `docs/PAPER_ANALYSIS.md`；论文问题: `docs/FINDINGS.md`。

## A. 方案正确性声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C1 | Alg.1 L6–22 | `DecDB(EncDB(e)) = e`（密文+nonce 可逆） | 最大绝对误差 2.8e-17（float64）；错误 nonce 无法恢复（误差 ≈0.11 >> 噪声/s=0.075；每次运行 nonce 随机，数值随运行略变） | VERIFIED | results/gate_M4_encdb.json, results/gate_M5_decdb.json |
| C2 | Alg.1 L11/L28, Eq.(7) | `‖η_eq‖ < sβ/8`, `‖λ_ei‖ < 3sβ/8`（严格上界） | 1000/4000 次采样均严格小于；max‖λ‖/R = 1.00000；球内均匀 KS p>0.8（每次重跑 KS p 随随机种子略变，均 ≫0.01）；方向均匀 max\|z\|=3.12 | VERIFIED | results/gate_M3_noise.json, gate_M4, gate_M6 |
| C3 | Eq.(2) | `‖eq−e1‖<‖eq−e2‖−β ⇒ ‖e'q−e'1‖<‖e'q−e'2‖` | 50,000 组前提构造三元组 **0 违反**；边界 gap=β+1e-6 亦 0 违反 | VERIFIED | results/gate_M7_eq2.json |
| C4 | Eq.(3) | `‖e1−e2‖<‖e1−e3‖−β ⇏ ‖e'1−e'2‖<‖e'1−e'3‖`（存在性） | 构造出合法噪声实现使顺序反转（lhs 2.4594 > rhs 2.3283）；命题为真 | VERIFIED | results/gate_M8_eq3.json |
| C5 | ADCPE §5.1 | ADCPE/SAP 是 β-DCP（所有向量同一噪声半径 sβ/4） | 20,000 组 0 违反；噪声界 0.15 严格成立 | VERIFIED | results/gate_M9_adcpe.json |
| C6 | §III-B1, §IV-1, Fig.5 | CAPRISE 打乱库内距离结构，向量分析攻击 ASR 低于 ADCPE | 27/27 个 cell CAPRISE ASR 更低：真实 GTR d=768 ASR@10 0.928 vs 0.951（−2.3pt）；合成 d=768/1536/3072 −6.1/−6.8/−4.0pt；距离重建误差 2.16×（真实）/1.77–2.08×（合成）。方向成立但真实数据幅度小（见 F2/F11） | VERIFIED（定性；幅度小于论文暗示） | results/exp4_vector_analysis.json, results/gate_M10_vector_analysis.json |
| C7 | §IV-1 | CAPRISE 继承 ADCPE 的 IND 类安全保证 | 未重做密码学证明；仅做经验泄漏检查 | BLOCKED（超出复现范围，见 F10） | docs/FINDINGS.md F10 |
| C8 | §III-B1 Eq.(4)–(20) | 证明的每一中间不等式成立（Eq.(2) 的 Eq.(8)–(12)；Eq.(3) 的 Eq.(13)–(20)） | 4000 组满足前提的查询三元组 + 4000 组库三元组逐式复核：Eq.(5)–(11)、Eq.(13)–(20) 全部 0 违反；Eq.(12) 打印式 4000/4000 不成立（符号笔误，F16），改为 `+` 后 0/4000；结论 Eq.(2) 0/4000 违反。附加条件 δ+δ'<sβ/2 在随机噪声下 0/4000 触发（F2） | VERIFIED（Eq.(12) 需按修正符号；论文排版错误） | results/proof_chain_check.json, scripts/check_proof_chain.py |

## B. Theorem 1 / 查询 DP

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| T1 | Eq.(21) | `k'−k = m·(F(α_k+Δα)−F(α_k))`（角壳层期望点数） | 解析（Beta 形式，roundtrip 4.6e-12；直接积分对照 1.1e-16）+ MC（m=100k 固定 DB，nq=200）：16 格最大偏差 2.38% <5%；ceil(Eq.21) 覆盖率 100% | VERIFIED | results/gate_M13_theorem1.json, results/exp5_query_dp.json |
| T2 | Table III | m=100k, n=768/1536, r∈{0.033,0.02}, k∈{5,10,15,20} 的 top-k' 数值 | 四约定系统枚举（16 格）：(a) Δα=r → 论文值的 0.444–0.737×（相对误差 −26.3%…−55.6%）；(b) 弦长 2asin(r/2) 与 (a) 差 <0.02%；(c) 欧氏扩张 → 反向高估 +12.0%…+71.1%；(d) 逐格反解 Δα/r = 1.1537–1.2977（mean 1.2129±0.0435）才能精确复现（≤5.6e-14）。参照实现交叉验证：RemoteRAG 高亮例（k=5,r=0.03,n=768,k'=160）在 N=1e6 下 Δα=r 给 170.34（+6.46%），点态反解 Δα/r=0.9804 ⇒ 被引实现用 Δα=r，论文表值不是 | FALSIFIED（论文表值不能由论文公式 + 被引来源 Δα≈r 约定复现；需未声明的 Δα≈1.2r，见 F1/A10） | results/exp5_query_dp.json (`delta_alpha_conventions`, `remoterag_reference`), notes/exp5_query_dp.md §4–5 |
| T3 | §III-B2 声称 | Theorem 1 给出检索所需 k'（"must satisfy"） | 定理是充分但极宽松上界：实测最小 k' ≈ k（k=5→ mean 5.28–5.53，p99 8.01；公式/p99=18.4×；全表松弛度 6.7–85.6×（vs mean）/5.5–58.8×（vs p99））；论文 Table III 的 52× 既非最小扩张也非 Δα=r 公式值 | FALSIFIED（作为"必要/最小"解读；作为充分上界成立，见 F9/A10b） | results/exp5_query_dp.json |
| T4 | §III-B2, RemoteRAG | DistanceDP：r∼Gamma(n,1/ε)，球面均匀方向，‖ẽ−e‖=r | 5000 次均值 0.099989（期望 0.1，相对误差 1.1e-4）；KS p=0.996；方向 max|z|=2.93；‖ẽ−e‖=r 误差 ≤2e-17；ε↔r̄ 互转与论文区间 [10n,50n] 一致 | VERIFIED（机制/实现；DP 证明本身未重做，见 F10） | results/gate_M12_distance_dp.json |

## C. 效率声明 (Fig.4, §V-1)

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| E1 | Fig.4a, §V-1 | 加密吞吐（embeddings/s）随嵌入维度上升而下降 | 论文 Fig.4 的 15 维全部实测：向量化 65,417(d=64)→20,823(d=3584) emb/s，drop 3.14×（Spearman ρ=−1.00，无相邻上升；论文自身曲线 256→312 上升 +24.4%）；paper-style 探针 29,138→693 emb/s，drop 42.1× vs 论文 44.5× | VERIFIED（趋势+量级，A17/F5） | results/gate_M17_efficiency.json, results/exp1_efficiency.json |
| E2 | Fig.4b, §V-1 | 元素级吞吐在 1536–3072 维震荡/维持高位 | paper-style 探针 elements/s 为论文数字化值的 1.19–2.39×（中位 1.36×），1536–1792 高平台 2.47M vs 论文 2.04M（1.21×）；向量化实现 elements/s 随 d 上升 17.8×（4.19M→74.63M），不呈现论文的高位/震荡形状。论文 d=256 局部凹陷未复现（−18.1% vs +9.5%），已登记 | VERIFIED（量级/高位成立；局部震荡不可复现，见 F5/A17） | results/exp1_efficiency.json |
| E3 | §V-1 | 128 query：嵌入 79.52ms + CAPRISE 加密 15ms，开销 <19% | 本机 128 真实 query：MPS fp16 533.05ms + 加密 1.42ms → 0.266%；CPU fp32 756.12ms → 0.188%；论文嵌入混合参照 1.786%；加密 1.42ms vs 论文 15ms（论文 15ms 对应逐元素实现风格） | VERIFIED（<19% 成立且更优；绝对值跨硬件不可比） | results/exp1_efficiency.json |
| E4 | §V-1 | d=768：CAPRISE 2339 vec/s vs RemoteRAG PHE 250–300 vec/s（9×） | CAPRISE 43,468 vec/s（单向量同口径，向量化，18.6× 论文）；Paillier 实测 1.1525 vec/s(1024-bit)、0.1747(2048-bit)；论文 9× 内部自洽（2339/250–300=7.8–9.4×）；本机比值 145–174× 属实现/硬件差异（A17/F12） | VERIFIED（方向+论文内部自洽；绝对倍数不可跨实现比） | results/exp1_efficiency.json |

## D. 隐私实验声明 (Table II, Fig.5)

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| P1 | Table II | 明文 GTR 嵌入可被 Vec2Text 重建：BLEU 83.021/P .947/R .950/F1 .948/ROUGE .950 | **50 条**真实 MS MARCO passage（32-token 预算，bootstrap 95% CI）：BLEU **69.875** [61.99, 78.17]/P .924/R .928/F1 .926/ROUGE-L .860 —— 同量级（0.84×，−13.1 BLEU）；10 条小样本时曾为 63.7；同 5 条上 20/4→50/10 明文 BLEU 55.7→77.4（+21.7）说明绝对基线依赖未声明的反演预算 | VERIFIED（量级/形态一致；绝对值受论文未给设置影响，F6/F15） | results/exp3_vec2text.json, results/gate_M11_vec2text.json |
| P2 | Table II | CAPRISE(s=3,β=0.2) 后：BLEU 12.425/P .482/R .498/F1 .487/ROUGE .492 | raw 攻击（论文口径，50 条 + 95% CI）：BLEU **13.696** [9.72, 17.48]/P .534/R .529/F1 .531/ROUGE-L .409 —— 与论文值差 +1.27（1.10×），CI 覆盖论文值；去缩放最强攻击：BLEU 34.584（安全限定，A11/F3）；同 5 条上 20/4→50/10 raw BLEU 7.21→7.16（−0.05，加密后水平对反演预算不敏感） | VERIFIED（加密后水平在 CI 内复现；强攻击口径见 F3） | results/exp3_vec2text.json, results/gate_M11_vec2text.json |
| P3 | Fig.5 | 向量分析 ASR：CAPRISE 在所有维度上低于 ADCPE | 27/27 cell 成立：真实 d=768 −2.3pt；合成 d=768/1536/3072 −6.1/−6.8/−4.0pt；β↑ 差距↑（−3.4/−4.2/−11.6pt）；论文 Fig.5 数字化对照见 F11 | VERIFIED（趋势成立；真实数据幅度小） | results/gate_M10_vector_analysis.json, results/exp4_vector_analysis.json |
| P4 | Table III, Fig.5, §VI | 隐私-精度 trade-off：r 越大 k' 越大、隐私越强但检索精度越低 | k' 随 r 上升：r=0.02→k'=72、r=0.05→k'=785（Theorem 1）；EXP2 中 k' 上升使 recall 保持 1.0 但耗时 3.11→11.81ms/query；β 越大噪声越大 recall 越低（0.9910→0.9401）；r 与隐私的量化见 exp5（角帽扩张）与 exp4（ASR） | VERIFIED | results/exp2_retrieval.json, results/exp5_query_dp.json |

## E. 检索质量声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| R1 | §V 摘要/正文 | ppRAG 保持 "high retrieval accuracy"（无数值） | 真实数据 m=50k/1000q：qrels MRR@10 0.7230（与明文完全相同）；Recall@10 相对明文：k'=k 0.9699，k'=15 0.9999，k'=2k 1.0000；CAPRISE+DP r=0.02 (k'=72) 1.0000；β 单调 0.9910→0.9401 | VERIFIED（需 small oversampling；严格 k'=k 损失 3%，见 F13） | results/exp2_retrieval.json, results/gate_M18_retrieval.json |
| R2 | Alg.2 L13–L19 | 密文 top-k' 检索 + 本地解密重排流程可行 | M14–M16 门禁全部 PASS：服务器仅收 e'_q（300 次调用全密文）；DP r=0.02 k'=55=Theorem1；解密+明文距离重排与客户端 top-10 完全一致；真实 recall@10 k'=k=0.964（m=8000，门禁）/0.9699（m=50k，exp2）/ k'=20=1.0000 / DP=1.0000；Phase3 prompt 无泄漏 | VERIFIED | results/gate_M14_phase1.json, results/gate_M15_phase2.json, results/gate_M16_phase3.json |

## F. 论文问题清单（详见 docs/FINDINGS.md）

| id | 问题 | 状态 |
|---|---|---|
| F1 | Table III 与正文 Δα≈r 约定不一致（需 ≈1.2r） | CONFIRMED（M13 最终确认数值） |
| F2 | Eq.(3) 仅存在性；高维实际反转率≈0 | CONFIRMED |
| F3 | Vec2Text 攻击如何适配密文未说明 | CONFIRMED |
| F4 | 没有任何检索精度数字 | CONFIRMED |
| F5 | 效率数字对实现风格敏感 | CONFIRMED |
| F6 | Table II 缺样本量/步数/归一化细节 | CONFIRMED |
| F7 | nonce 未说明随密文返回（DecDB 必需） | CONFIRMED |
| F8 | 高维各向同性向量无法满足 gap>β 前提 | CONFIRMED |
| F9 | Theorem 1 的 k' 是充分但极宽松上界 | CONFIRMED（M13：实测最小 k'≈k，松弛度 6.7–85.6×） |
| F10 | 形式化安全性未复现（范围限制） | CONFIRMED（声明） |
| F11 | Fig.5 的 12 个"尺寸"点与 GTR 768 维矛盾；趋势部分不可复现 | CONFIRMED |
| F12 | 效率倍数跨实现/跨硬件不可比 | CONFIRMED |
| F13 | 加密噪声对良性语料 top-10 影响极小（需硬负例才可测） | CONFIRMED |
| F14 | Vec2Text 攻击必须使用 vec2text 训练约定的嵌入 | CONFIRMED |
| F15 | Table II 绝对数值只能在相同攻击约定/数据规模下对齐 | CONFIRMED |
| F16 | 证明 Eq.(12) 符号笔误（`−` 应为 `+`，结论不变） | CONFIRMED（proof_chain_check 4000/4000） |

## G. 判定统计（收尾时更新）

| 判定 | 数量 |
|---|---|
| VERIFIED | **20**（C1–C6, C8, T1, T4, E1–E4, P1–P4, R1, R2） |
| FALSIFIED | 2（T2、T3——Table III 与定理"必要/最小"解读；证据见 FINDINGS F1/F9） |
| BLOCKED | 1（C7——形式化安全性，超出复现范围） |
| UNKNOWN | **0** |

> 收尾条件（skill §7）：CLAIMS_CHECKLIST 无 UNKNOWN 后才可结束复现；FALSIFIED 必须有可复跑证据。
