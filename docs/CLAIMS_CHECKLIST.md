# 论文声明核对表（CLAIMS_CHECKLIST）

> 判定只用: **VERIFIED / FALSIFIED / BLOCKED / UNKNOWN**。
> UNKNOWN = 复现尚未完成；BLOCKED = 超出复现范围（需形式化证明/不可获得资源），不得算作已验证。
> 每条声明必须有可复跑证据路径；论文值逐项对照，偏差必须量化。
> 细节分析: `docs/PAPER_ANALYSIS.md`；论文问题: `docs/FINDINGS.md`。

## A. 方案正确性声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| C1 | Alg.1 L6–22 | `DecDB(EncDB(e)) = e`（密文+nonce 可逆） | 最大绝对误差 2.8e-17（float64）；错误 nonce 无法恢复（误差 0.106 >> 噪声/s=0.075） | VERIFIED | results/gate_M4_encdb.json, results/gate_M5_decdb.json |
| C2 | Alg.1 L11/L28, Eq.(7) | `‖η_eq‖ < sβ/8`, `‖λ_ei‖ < 3sβ/8`（严格上界） | 1000/4000 次采样均严格小于；max‖λ‖/R = 1.00000；球内均匀 KS p=0.63/0.72 | VERIFIED | results/gate_M3_noise.json, gate_M4, gate_M6 |
| C3 | Eq.(2) | `‖eq−e1‖<‖eq−e2‖−β ⇒ ‖e'q−e'1‖<‖e'q−e'2‖` | 50,000 组前提构造三元组 **0 违反**；边界 gap=β+1e-6 亦 0 违反 | VERIFIED | results/gate_M7_eq2.json |
| C4 | Eq.(3) | `‖e1−e2‖<‖e1−e3‖−β ⇏ ‖e'1−e'2‖<‖e'1−e'3‖`（存在性） | 构造出合法噪声实现使顺序反转（lhs 2.4594 > rhs 2.3283）；命题为真 | VERIFIED | results/gate_M8_eq3.json |
| C5 | ADCPE §5.1 | ADCPE/SAP 是 β-DCP（所有向量同一噪声半径 sβ/4） | 20,000 组 0 违反；噪声界 0.15 严格成立 | VERIFIED | results/gate_M9_adcpe.json |
| C6 | §III-B1, §IV-1, Fig.5 | CAPRISE 打乱库内距离结构，向量分析攻击 ASR 低于 ADCPE | 27/27 个 cell CAPRISE ASR 更低：真实 GTR d=768 ASR@10 0.928 vs 0.951（−2.3pt）；合成 d=768/1536/3072 −6.1/−6.8/−4.0pt；距离重建误差 2.16×（真实）/1.77–2.08×（合成）。方向成立但真实数据幅度小（见 F2/F11） | VERIFIED（定性；幅度小于论文暗示） | results/exp4_vector_analysis.json, results/gate_M10_vector_analysis.json |
| C7 | §IV-1 | CAPRISE 继承 ADCPE 的 IND 类安全保证 | 未重做密码学证明；仅做经验泄漏检查 | BLOCKED（超出复现范围，见 F10） | docs/FINDINGS.md F10 |

## B. Theorem 1 / 查询 DP

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| T1 | Eq.(21) | `k'−k = m·(F(α_k+Δα)−F(α_k))`（角壳层期望点数） | 解析（Beta 形式，roundtrip 4.6e-12；直接积分对照 1.1e-16）+ MC（m=100k 固定 DB，nq=200）：16 格最大偏差 2.38% <5%；ceil(Eq.21) 覆盖率 100% | VERIFIED | results/gate_M13_theorem1.json, results/exp5_query_dp.json |
| T2 | Table III | m=100k, n=768/1536, r∈{0.033,0.02}, k∈{5,10,15,20} 的 top-k' 数值 | Δα=r 时 k' = 论文值的 0.444–0.737×；逐格反推 Δα/r = 1.213±0.044（1.154–1.298）；Δα=1.2r 最吻合（论文/预测均值 1.042）；距离空间约定 0.717 系统性偏低 | FALSIFIED（论文表值不能由论文公式+Δα≈r 约定复现，见 F1/A10） | results/exp5_query_dp.json, notes/exp5_query_dp.md |
| T3 | §III-B2 声称 | Theorem 1 给出检索所需 k'（"must satisfy"） | 定理是充分但极宽松上界：实测最小 k' ≈ k（k=5→5.53，p99 8.01；公式/p99=18.4×；全表松弛度 6.7–85.6×）；论文 Table III 的 52× 既非最小扩张也非 Δα=r 公式值 | FALSIFIED（作为"必要/最小"解读；作为充分上界成立，见 F9/A10b） | results/exp5_query_dp.json |
| T4 | §III-B2, RemoteRAG | DistanceDP：r∼Gamma(n,1/ε)，球面均匀方向，‖ẽ−e‖=r | 5000 次均值 0.099989（期望 0.1，相对误差 1.1e-4）；KS p=0.996；方向 max|z|=2.93；‖ẽ−e‖=r 误差 ≤2e-17；ε↔r̄ 互转与论文区间 [10n,50n] 一致 | VERIFIED（机制/实现；DP 证明本身未重做，见 F10） | results/gate_M12_distance_dp.json |

## C. 效率声明 (Fig.4, §V-1)

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| E1 | Fig.4a, §V-1 | 加密吞吐（embeddings/s）随嵌入维度上升而下降 | d=384→4096: 69,838→24,813 embeddings/s（单调下降 2.8×；负载敏感，空闲时 d=768 可达 86.5k） | VERIFIED | results/gate_M17_efficiency.json, results/exp1_efficiency.json |
| E2 | Fig.4b, §V-1 | 元素级吞吐在 1536–3072 维震荡/维持高位 | elements/s: 26.8M(d=384)→113.9M(d=3072)→101.6M(d=4096)，同数量级；1536–3072 为 73.3M→113.9M（"高位"成立；精确"震荡"形态受负载影响，A17/F12） | VERIFIED（趋势方向部分，见 F12） | results/exp1_efficiency.json |
| E3 | §V-1 | 128 query：嵌入 79.52ms + CAPRISE 加密 15ms，开销 <19% | 本机 128 真实 query：MPS fp16 224.68ms + 加密 2.281ms → 1.015%；CPU fp32 747.82ms → 0.305%；论文嵌入混合参照 2.87%；加密 2.281ms vs 论文 15ms | VERIFIED（<19% 成立且更优；绝对值跨硬件不可比） | results/exp1_efficiency.json |
| E4 | §V-1 | d=768：CAPRISE 2339 vec/s vs RemoteRAG PHE 250–300 vec/s（9×） | CAPRISE 67,996 vec/s（单向量同口径，向量化）；Paillier 实测 1.6254 vec/s(1024-bit)、0.2513(2048-bit)；论文 9× 内部自洽（2339/250–300=7.8–9.4×）；本机比值 227–272× 属实现/硬件差异（A17/F12） | VERIFIED（方向+论文内部自洽；绝对倍数不可跨实现比） | results/exp1_efficiency.json |

## D. 隐私实验声明 (Table II, Fig.5)

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| P1 | Table II | 明文 GTR 嵌入可被 Vec2Text 重建：BLEU 83.021/P .947/R .950/F1 .948/ROUGE .950 | 10 条校准 passage（32-token 预算）：BLEU **63.727**/P .886/R .927/F1 .906/ROUGE-L .801 —— 同量级（−19.3 BLEU），受样本量/长度校准/CPU 影响 | VERIFIED（量级/形态一致；绝对值不可完全对齐，F6/F15） | results/exp3_vec2text.json, results/gate_M11_vec2text.json |
| P2 | Table II | CAPRISE(s=3,β=0.2) 后：BLEU 12.425/P .482/R .498/F1 .487/ROUGE .492 | raw 攻击（论文口径）：BLEU **9.258**/P .580/R .530/F1 .554/ROUGE-L .442（PDF 高度接近）；去缩放最强攻击：BLEU 30.289（安全限定，A11/F3） | VERIFIED（加密后水平复现；强攻击口径见 F3） | results/exp3_vec2text.json, results/gate_M11_vec2text.json |
| P3 | Fig.5 | 向量分析 ASR：CAPRISE 在所有维度上低于 ADCPE | 27/27 cell 成立：真实 d=768 −2.3pt；合成 d=768/1536/3072 −6.1/−6.8/−4.0pt；β↑ 差距↑（−3.4/−4.2/−11.6pt）；论文 Fig.5 数字化对照见 F11 | VERIFIED（趋势成立；真实数据幅度小） | results/gate_M10_vector_analysis.json, results/exp4_vector_analysis.json |
| P4 | Table III, Fig.5, §VI | 隐私-精度 trade-off：r 越大 k' 越大、隐私越强但检索精度越低 | k' 随 r 上升：r=0.02→k'=72、r=0.05→k'=785（Theorem 1）；EXP2 中 k' 上升使 recall 保持 1.0 但耗时 3.11→11.81ms/query；β 越大噪声越大 recall 越低（0.9910→0.9401）；r 与隐私的量化见 exp5（角帽扩张）与 exp4（ASR） | VERIFIED | results/exp2_retrieval.json, results/exp5_query_dp.json |

## E. 检索质量声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| R1 | §V 摘要/正文 | ppRAG 保持 "high retrieval accuracy"（无数值） | 真实数据 m=50k/1000q：qrels MRR@10 0.7230（与明文完全相同）；Recall@10 相对明文：k'=k 0.9699，k'=15 0.9999，k'=2k 1.0000；CAPRISE+DP r=0.02 (k'=72) 1.0000；β 单调 0.9910→0.9401 | VERIFIED（需 small oversampling；严格 k'=k 损失 3%，见 F13） | results/exp2_retrieval.json, results/gate_M18_retrieval.json |
| R2 | Alg.2 L13–L19 | 密文 top-k' 检索 + 本地解密重排流程可行 | M14–M16 门禁全部 PASS：服务器仅收 e'_q（300 次调用全密文）；DP r=0.02 k'=55=Theorem1；解密+明文距离重排与客户端 top-10 完全一致；真实 recall@10 k'=k=0.9590 / k'=20=1.0000 / DP=1.0000；Phase3 prompt 无泄漏 | VERIFIED | results/gate_M14_phase1.json, results/gate_M15_phase2.json, results/gate_M16_phase3.json |

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
| F9 | Theorem 1 的 k' 是充分但极宽松上界 | PENDING（M13） |
| F10 | 形式化安全性未复现（范围限制） | CONFIRMED（声明） |

## G. 判定统计（收尾时更新）

| 判定 | 数量 |
|---|---|
| VERIFIED | 18（C1–C6, T1, T4, E1–E4, P1–P4, R1, R2） |
| FALSIFIED | 2（T2, T3——论文内部不一致/声明过强，证据见 FINDINGS F1/F9） |
| BLOCKED | 1（C7——形式化安全性，超出复现范围） |
| UNKNOWN | 0 |

> 收尾条件（skill §7）：CLAIMS_CHECKLIST 无 UNKNOWN 后才可结束复现；FALSIFIED 必须有可复跑证据。
