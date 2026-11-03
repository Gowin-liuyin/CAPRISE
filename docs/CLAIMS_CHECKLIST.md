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
| C6 | §III-B1, §IV-1, Fig.5 | CAPRISE 打乱库内距离结构，向量分析攻击 ASR 显著低于 ADCPE | 实测：真实采样下库内反转率 0/20000（见 F2）；距离重建误差 CAPRISE 0.0046 vs ADCPE 0.0025（1.8×）；ASR@10 初测 CAPRISE 低 4–6 个百分点 | UNKNOWN（待 M10 最终数据，趋势方向成立、幅度小于论文暗示） | results/exp4_vector_analysis.json（初版）, results/gate_M8_eq3.json |
| C7 | §IV-1 | CAPRISE 继承 ADCPE 的 IND 类安全保证 | 未重做密码学证明；仅做经验泄漏检查 | BLOCKED（超出复现范围，见 F10） | docs/FINDINGS.md F10 |

## B. Theorem 1 / 查询 DP

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| T1 | Eq.(21) | `k'−k = m·(F(α_k+Δα)−F(α_k))`（角壳层期望点数） | 解析实现（Beta 稳定）+ Monte Carlo 预期 <5% 偏差 | UNKNOWN（待 M13） | — |
| T2 | Table III | m=100k, n=768/1536, r∈{0.033,0.02}, k∈{5,10,15,20} 的 top-k' 数值 | Δα=r 时 k'≈论文的 0.55–0.75×；论文值对应 Δα≈1.2r（内部不一致，F1） | UNKNOWN（待 M13 最终量子化） | — |
| T3 | §III-B2 声称 | Theorem 1 给出检索所需 k' | 定理是充分但极宽松上界；实测最小 k'≈k（见 F9） | UNKNOWN（待 M13） | — |
| T4 | §III-B2, RemoteRAG | DistanceDP：r∼Gamma(n,1/ε)，球面均匀方向，‖ẽ−e‖=r | 5000 次均值 0.099989（期望 0.1，相对误差 1.1e-4）；KS p=0.996；‖ẽ−e‖=r 误差 <1e-17 | VERIFIED（机制/实现；DP 证明本身未重做，见 F10） | results/gate_M12_distance_dp.json |

## C. 效率声明 (Fig.4, §V-1)

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| E1 | Fig.4a, §V-1 | 加密吞吐（embeddings/s）随嵌入维度上升而下降 | 待测（M17） | UNKNOWN | — |
| E2 | Fig.4b, §V-1 | 元素级吞吐在 1536–3072 维维持高位 | 待测（M17） | UNKNOWN | — |
| E3 | §V-1 | 128 query：嵌入 79.52ms + CAPRISE 加密 15ms，开销 <19% | 待测（M17，CPU） | UNKNOWN | — |
| E4 | §V-1 | d=768：CAPRISE 2339 vec/s vs RemoteRAG PHE 250–300 vec/s（9×） | 待测（M17，CPU + Paillier 微基准） | UNKNOWN | — |

## D. 隐私实验声明 (Table II, Fig.5)

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| P1 | Table II | 明文 GTR 嵌入可被 Vec2Text 重建：BLEU 83.021/P .947/R .950/F1 .948/ROUGE .950 | 待测（M11） | UNKNOWN | — |
| P2 | Table II | CAPRISE(s=3,β=0.2) 后：BLEU 12.425/P .482/R .498/F1 .487/ROUGE .492 | 待测（M11） | UNKNOWN | — |
| P3 | Fig.5 | 向量分析 ASR：CAPRISE 在所有维度上低于 ADCPE | 初测：d=768 CAPRISE ASR@10=0.852 vs ADCPE 0.896；d=1536 0.844 vs 0.904（方向成立，幅度 4–6pt） | UNKNOWN（待 M10 最终版 + 真实 GTR） | results/exp4_vector_analysis.json |
| P4 | Table III, Fig.5, §VI | 隐私-精度 trade-off：r 越大 k' 越大、隐私越强但检索精度越低 | 待测（M13/M18） | UNKNOWN | — |

## E. 检索质量声明

| id | 论文出处 | 论文值 / 命题 | 复现值 | 判定 | 证据 |
|---|---|---|---|---|---|
| R1 | §V 摘要/正文 | ppRAG 保持 "high retrieval accuracy"（无数值） | 待测（M18，补充实验） | UNKNOWN | — |
| R2 | Alg.2 L13–L19 | 密文 top-k' 检索 + 本地解密重排流程可行 | 集成 smoke test：m=2000 合成聚类嵌入，CAPRISE 无 DP k'=k 时 Recall@10=0.88；k'=15/20 时 0.99/1.00；DP+定理 k' 时 1.00 | UNKNOWN（待 M14–M16 门禁 + M18） | tests/test_pprag_smoke.py（临时证据）, results/gate_M14..16（待） |

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
| VERIFIED | 6（C1–C5, T4） |
| FALSIFIED | 0 |
| BLOCKED | 1（C7） |
| UNKNOWN | 13 |

> 收尾条件（skill §7）：CLAIMS_CHECKLIST 无 UNKNOWN 后才可结束复现；FALSIFIED 必须有可复跑证据。
