# 论文要素覆盖矩阵（PAPER COVERAGE）

> 目的：把论文 **每一个可复现要素**（公式 / 算法行 / 表 / 图 / 章节声明）逐条映射到本仓库的实现、实验、门禁与证据，
> 使"所有部分与论文描述对齐"可被第三方逐行核对，而不依赖任何"已完成"的口头结论。
>
> 论文：*Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption*
> （arXiv:2601.12331v1, 2026-01-18；LaTeX 源 `output/pdf/source/caprise/ppRAG_FLLM.tex`）
>
> 判定口径（与 `docs/CLAIMS_CHECKLIST.md` 一致）：
> - **VERIFIED**：复现值/行为与论文命题一致（数值需给出偏差与容差依据）。
> - **FALSIFIED**：论文的数值/命题在论文自身公式与被引用来源的约定下无法复现（附可复跑证据）。
> - **PARTIAL**：方向/结构一致但绝对数值受未声明的实验设置或跨硬件不可比影响，已量化偏差。
> - **BLOCKED**：需要形式化证明或不可获得资源，超出复现范围（明确声明，不算已验证）。
> - **N/A**：示意图/文献综述等无数值主张的要素，仅做结构核对。

| 统计 | 内容 |
|---|---|
| 逐条覆盖 | §2–§12 按"公式逐式、算法逐行、表/图/声明逐项"拆分标注（合计 80+ 条判定单元） |
| VERIFIED / PARTIAL / N/A | 逐条见各表"判定"列（PARTIAL 均已量化偏差） |
| **FALSIFIED（2 项）** | ① Table III 数值（§7）；② Theorem 1 的"must satisfy/最小 k'"解读（§7，作为充分上界仍 VERIFIED） |
| **BLOCKED（3 项）** | ① §IV CAPRISE 的 IND 类安全证明；② DistanceDP 的 (ε,δ)-DP 证明；③ Alg.2 行 25 的真实 LLM 生成（§12） |
| N/A（8 项） | Fig.1/2/3 示意图、Table I 中非本文行的文献条目（§3/§4/§12） |

---

## 1. 论文章节总览

| 论文章节 | 复现状态 | 证据 |
|---|---|---|
| Abstract / I Introduction（3 条贡献） | VERIFIED | 见 §2；M1–M9 / M14–M16 / M7,M8,M13 / M10,M11,M17,M18 |
| II Related Work + Table I | PARTIAL（自方列 VERIFIED；对照文献列 N/A） | 见 §3 |
| III-A Threat Model + Fig.1–3 | VERIFIED（结构）；示意图 N/A | M14–M16 |
| III-B Algorithm 1（KeyGen/EncDB/DecDB/EncQ） | VERIFIED | M1–M6 |
| III-B1 Claim Eq.(2)(3) + 证明 Eq.(4)–(20) | VERIFIED（Eq.(12) 有符号笔误，F16） | M7, M8, `results/proof_chain_check.json` |
| III-B2 DistanceDP + Theorem 1 / Eq.(21) | 机制 VERIFIED；公式 VERIFIED；Table III **FALSIFIED** | M12, M13 |
| Algorithm 2（Phase 1/2/3） | Phase 1/2 VERIFIED；Phase 3 prompt VERIFIED、LLM 生成 BLOCKED | M14–M16 |
| IV Security Analysis | BLOCKED（形式化证明）；机制级经验验证 VERIFIED | F10；M11 |
| V Experimental Evaluation（Table II/III、Fig.4/5、Phase-1/2） | Table II PARTIAL；Table III FALSIFIED；Fig.4 PARTIAL；Fig.5 VERIFIED（趋势）/PARTIAL（幅度） | M10/M11/M13/M17 |
| VI Discussions（β 标定、隐私-精度 trade-off） | VERIFIED（定性 → 定量） | exp2 β 扫描、exp4 β 扫描、exp5 |
| VII Conclusion（高检索效用、高吞吐） | VERIFIED（检索效用）/PARTIAL（吞吐绝对值） | M18、M17 |

---

## 2. Abstract / Introduction 贡献

| # | 论文声明 | 复现实现 | 判定 | 证据 |
|---|---|---|---|---|
| P-1 | ppRAG：外包加密数据库上的高效隐私保护 RAG 框架（Phase 1/2/3） | `src/caprise/pprag.py`；真实 MS MARCO 8000 文档端到端 | VERIFIED | `results/gate_M14_phase1.json`, `gate_M15_phase2.json`, `gate_M16_phase3.json` |
| P-2 | CAPRISE：条件距离比较保留加密（仅保查询-库距离，混淆库内距离） | `src/caprise/caprise.py`（KeyGen/EncDB/DecDB/EncQ） | VERIFIED | `gate_M1`–`gate_M9` |
| P-3 | 理论证明 + 安全分析 + 实验评估 | Eq.(2)(3) 证明逐式复核；安全分析 BLOCKED；5 组实验 | VERIFIED / BLOCKED | `docs/CLAIMS_CHECKLIST.md` |

---

## 3. II Related Work + Table I

| 要素 | 论文内容 | 复现核对 | 判定 |
|---|---|---|---|
| Table I 自方列（ppRAG） | Cloud ✓, Local ✓, DP ✓, ADCPE ✓, HE ✗, OT ✗ | 实现：云服务器（EncryptedVectorDB）、本地嵌入/加密/解密、DistanceDP、ADCPE 基线；无 HE/OT | VERIFIED |
| Table I RemoteRAG 行 | Cloud ✓, Local ✓, DP ✓, HE ✓, OT ✓ | 阅读 RemoteRAG（arXiv:2412.12775）核对：PHE 检索、DistanceDP、k-out-of-k' OT | VERIFIED（对引用文献原文） |
| Table I DP-RAG 行 | Local ✓, DP ✓ | 引用文献 [20] 无原文副本，未核对 | N/A（文献条目） |
| §II 对 ADCPE 的描述 | "preserving all pairwise distances" | M9：ADCPE/SAP 在 β-DCP 意义下成对顺序保留 0/20000 违反；噪声半径 sβ/4 | VERIFIED |

---

## 4. III-A Threat Model 与设计直觉

| 要素 | 论文内容 | 复现核对 | 判定 | 证据 |
|---|---|---|---|---|
| 两参与方 | User 本地嵌入/加密；Cloud 存储+检索 | `PPRAGClient` / `PPRAGServer` | VERIFIED | M14 |
| honest-but-curious 服务器 | 服务器只应有密文 + nonce + AES 密文 | 服务器对象无 `CryptoKey`、无明文嵌入/文档 | VERIFIED | M14 check `02_server_holds_no_secret_key`, `04_no_plaintext_documents_on_server` |
| 风险 1：Vec2Text 反演 | 明文嵌入可反演 | Vec2Text 在明文/密文上实测 | VERIFIED | M11 |
| 风险 2：向量分析 | 库内两两距离泄漏语义结构 | ASR/距离重建误差实测 | VERIFIED | M10 |
| 风险 3：查询分析 | 重复查询泄漏访问模式 | DistanceDP + Theorem 1 扩张 | VERIFIED（机制） | M12/M13 |
| 设计目标 | 同时保护嵌入语义内容与关系信息 | 上述三个实验共同支撑 | VERIFIED | — |
| Fig.1 总览 | 三阶段流程 | 与 Algorithm 2 三阶段一一对应 | N/A（结构核对） | M14–M16 |
| Fig.2 ADCPE vs CAPRISE 示意 | 实线保序、虚线扰动 | M7（实线保序）/M8+M10（虚线被扰动） | N/A（示意）+ 实验支撑 | M7/M8/M10 |
| Fig.3 球面投影示意 | DP 扰动扩张检索范围 | M12（机制）/M13（扩张量） | N/A（示意）+ 实验支撑 | M12/M13 |

---

## 5. III-B Algorithm 1 逐行

| Alg.1 行 | 论文语句 | 实现 | 验证门禁 | 判定 |
|---|---|---|---|---|
| 1–5 KeyGen | `s ←$ S`（`|S| ≤ 2λ`）、`K ←$ {0,1}^k`、返回 `(s,K)` | `KeyGen`：s∈[1,2^16] 整数、K=128 bit | M2（10 项：keyspace/长度/唯一性/非法参数） | VERIFIED（A1/A2 已声明） |
| 6–14 EncDB | `r ←$ {0,1}^ℓ`；`randbits1‖randbits2 ← PRF(K,r)`；`n ← N(0,I_d)`；`u ← U_{0,1}`；`λ ← (3/8)(n sβ/‖n‖)u^{1/d}`；`e' ← s e + λ`；返回 `(e',r)` | `EncDB_one/EncDB`；`prf.coins` | M1（PRF 确定性/敏感性）、M3（球内均匀、模长严格<R）、M4（往返 2.78e-17、批=单、nonce 唯一） | VERIFIED |
| 8 | 一次 PRF 输出切两段 | `prf.coins`：HMAC-SHA256 32B 切 16B+16B | M1 `coins_shape/deterministic` | VERIFIED |
| 9 | `n ← N(0,I_d)(randbits1)` | PCG64(seed=randbits1).standard_normal(d) | M1 KS p=0.345；M3 方向均匀 max\|z\|=3.12 | VERIFIED（A3 已声明） |
| 10 | `u ← U_{0,1}(randbits2)` | `(int+1)/2^128 ∈ (0,1)` 严格内点 | M1 `u_strictly_interior`、KS p=0.742 | VERIFIED |
| 11 | `λ ← (3/8)(n sβ/‖n‖)u^{1/d}` | 半径 `3sβ/8`，方向 n/‖n‖，模长 `R·u^{1/d}` | M3：max/R=1.00000、球内均匀 KS p=0.8705、R_db/R_q=3 | VERIFIED |
| 15–22 DecDB | 同 8–10 重建 λ；`e ← (e'−λ)/s` | `DecDB_one/DecDB` | M5：确定性、精确逆 2.78e-17、错误 nonce 不可恢复（误差 0.109） | VERIFIED |
| 23–31 EncQ | 同 EncDB 但半径 `sβ/8` | `EncQ/DecQ` | M6：往返 1.39e-17、max‖η‖=0.075、新鲜 nonce 改变密文 | VERIFIED |
| 返回 `(e',r)` | nonce 随密文返回（论文未写，A9） | `EncDB` 返回 nonces；服务器存储 | M4/M14 | VERIFIED（A9 已声明） |

---

## 6. III-B1 Claim 与证明逐式（Eq.(1)–(20)）

> 逐式数值复核脚本：`scripts/check_proof_chain.py` → `results/proof_chain_check.json`
> （4000 组满足前提的查询三元组 + 4000 组库三元组，s=3, β=0.2, d=768）

| 式 | 论文语句 | 复现核对 | 判定 |
|---|---|---|---|
| (1) | ADCPE β-DCP 定义 | SAP 实现（半径 sβ/4）；M9 实测 0/20000 违反 | VERIFIED |
| (2) | `‖e_q−e_1‖<‖e_q−e_2‖−β ⇒ ‖e'_q−e'_1‖<‖e'_q−e'_2‖` | M7：50,000 组构造成立前提，0 违反；边界 gap=β+1e-6 仍 0 违反；证明链 0/4000 违反 | VERIFIED |
| (3) | `‖e_1−e_2‖<‖e_1−e_3‖−β ⇏ ‖e'_1−e'_2‖<‖e'_1−e'_3‖`（存在性） | M8：构造合法噪声实现使顺序反转（lhs 2.4594 > rhs 2.3283）；随机采样 0/20000 反转（F2/A19） | VERIFIED（存在性；高维随机下几乎不发生） |
| (4) | 证明前提 | 由 M7/M8 构造满足前提的三元组 | VERIFIED |
| (5) | `e'_q = s e_q + η` | 实现恒等式；`proof_chain_check` eq5_identity 0/4000 违反 | VERIFIED |
| (6) | `e'_i = s e_i + λ_i` | 同上 eq6_identity | VERIFIED |
| (7) | `‖η‖<sβ/8`、`‖λ‖<3sβ/8` | M3/M4/M6 严格 <；证明链 0/4000 违反 | VERIFIED |
| (8) | 三角不等式上界 | proof_chain eq8 0/4000 违反 | VERIFIED |
| (9) | 代入前提 | eq9 0/4000 | VERIFIED |
| (10) | 代入 (7) | eq10 0/4000 | VERIFIED |
| (11) | `≤ ‖se_q−se_2‖−‖η−λ‖` | eq11/eq11b 0/4000 | VERIFIED |
| (12) | `≤ ‖(se_q−se_2)−(η−λ)‖ = ‖e'_q−e'_2‖` | **打印式符号笔误**：按打印式 4000/4000 不成立；改为 `+` 后 0/4000 成立（F16） | VERIFIED（修正符号后；论文排版错误） |
| (13) | 上界 `‖λ_1‖+‖λ_2‖+s(‖e_1−e_3‖−β)` | eq13 0/4000 | VERIFIED |
| (14) | 代入 (7) | eq14 0/4000 | VERIFIED |
| (15) | 存在 δ>0：`= ‖se_1−se_3‖−sβ/4−δ` | eq15 δ>0 0/4000 | VERIFIED（存在性，δ 不可显式计算） |
| (16) | `‖e'_1−e'_3‖ = ‖(se_1−se_3)+(λ_1−λ_3)‖` | eq16 恒等式 0/4000 | VERIFIED |
| (17) | 反三角 | eq17 0/4000 | VERIFIED |
| (18) | 三角 | eq18 0/4000 | VERIFIED |
| (19) | `> ‖se_1−se_3‖−3sβ/4` | eq19 0/4000 | VERIFIED |
| (20) | 存在 δ'>0：`= ‖se_1−se_3‖−3sβ/4+δ'` | eq20 δ'>0 0/4000 | VERIFIED |
| 结论 | `δ+δ'<sβ/2 ⇒ ‖e'_1−e'_2‖>‖e'_1−e'_3‖` | 该附加条件在随机噪声下 4000/4000 不触发 → 仅存在性（F2/A19） | VERIFIED（存在性） |

---

## 7. III-B2 DistanceDP、Theorem 1 与 Table III

| 要素 | 论文内容 | 复现实现/实验 | 判定 | 证据 |
|---|---|---|---|---|
| DistanceDP 机制 | "adds calibrated noise ... and expands the retrieval space"（引用 RemoteRAG） | `src/caprise/dp.py`：r~Gamma(n,1/ε)、球面均匀方向、`ẽ=e+r·d` | VERIFIED | M12：均值 0.099989（期望 0.1）、KS p=0.9957、方向 max\|z\|=2.925、‖ẽ−e‖=r 误差 2.1e-17、ε∈[10n,50n] ↔ r̄∈[0.1,0.02] |
| Lemma 1 的 α_k（RemoteRAG） | `k = m·(Ω_{n-1}/Ω_n)∫_0^{α_k} sin^{n-2}` | `theorem.alpha_for_fraction`（Beta 反解） | VERIFIED | `m·F(α_k)=k` 误差 2.2e-12（16 格） |
| Eq.(21) 公式本体 | `k' = k + m·(Ω 比)·∫_{α_k}^{α_k'} sin^{n-2}θ` | `theorem.k_prime`（Beta 形式） | VERIFIED | Beta roundtrip 4.6e-12；直接积分对照 1.1e-16；与 MC 16 格偏差 ≤2.38%；`ceil(Eq.21)` 覆盖明文 top-k 100% |
| Table III 数值 | m=100k, n=768/1536, r=0.033/0.02, k=5/10/15/20 | 16 格 × 四约定（Δα=r / 弦长 / 欧氏扩张 / 逐格反解）+ 参照实现交叉验证 | **FALSIFIED** | 可推导的三种约定逐格最优仍偏离 ≥12.0%：(a) Δα=r 为论文的 0.444–0.737×、(b) 弦长与 (a) 差 <0.02%、(c) 欧氏扩张高估 12–71%；只有 (d) 逐格反解 `Δα/r=1.1537–1.2977`（mean 1.2129±0.0435）能精确复现（≤5.6e-14）。RemoteRAG 高亮例（k'=160）与 Δα=r 自洽（+6.46%）⇒ 表值与被引实现约定不一致。见 F1/A10、`notes/exp5_query_dp.md` §4–5 |
| 定理的"must satisfy"解读 | §III-B2 声称给出检索所需 k' | MC 实测最小 k'≈k（k=5→mean 5.53/p99 8.01；公式/p99=18.4×） | FALSIFIED（作为"必要/最小"）；VERIFIED（作为充分上界） | `results/exp5_query_dp.json` |

---

## 8. Algorithm 2 逐行

| Alg.2 行 | 论文语句 | 实现 | 验证门禁 | 判定 |
|---|---|---|---|---|
| 1–9 Phase 1 | 输入 `{x_i}`；`e_i←LLM_e(x_i)`；KeyGen；EncDB；AES KeyGen/加密；上传 `{(e'_i,x'_i)}` | `PPRAGClient.phase1_upload` + `AESDocumentCipher` | M14：服务器仅密文+nonce+AES；8000/8000 AES 往返；DecDB 9.8e-9 | VERIFIED |
| 10–21 Phase 2 | `e_q←LLM_e(q)`；`k',ẽ_q←DistanceDP`；`e'_q←EncQ`；发送 e'_q；服务器 top-k'；DecDB；ReRank；Top-k | `phase2_retrieve` | M15：只传密文（300 次调用）、k'=55 与定理一致、解密重排与客户端 top-10 完全一致、recall 0.9640(m=8000)/0.9699(m=50k) | VERIFIED |
| 12 | `k', ẽ_q ← DistanceDP(e_q)` | DP 半径 + `theorem.k_prime` | M15 `04_dp_k_prime_from_theorem1` | VERIFIED |
| 18 | `ReRank`（模型未指定，A6） | 解密后明文精确距离重排 | M15 `08_rerank_reproducible` | VERIFIED（A6 已声明） |
| 22–27 Phase 3 | `input ← q + {x_1..x_k}`；`Response ← LLM(input)` | `phase3_build_prompt`；LLM 生成不执行 | M16：prompt 精确、10 篇有序、0 泄漏；LLM BLOCKED | prompt VERIFIED / LLM BLOCKED |

---

## 9. IV Security Analysis

| 声明 | 复现核对 | 判定 |
|---|---|---|
| CAPRISE 继承 ADCPE 的 IND 类安全；条件化不降低安全保证 | 未重做密码学归约证明。机制级：nonce 公开但无 K 不可重建 λ（M5 错误 nonce 0.109）；服务器无密钥（M14）；密文分布统计检查 | **BLOCKED**（形式化证明超出范围，F10） |
| 噪声标定足以破坏 Vec2Text | M11 实测（50 条 + 95% CI）：明文 BLEU 69.9 → 密文 raw 13.7 / 尺度还原攻击 34.6（仍低于明文 51%）；依赖攻击者假设（A11/F3） | VERIFIED（经验）/ 依赖假设 |
| DistanceDP 满足 (ε,δ)-DP | 机制与 RemoteRAG 一致（M12）；DP 证明未重做 | **BLOCKED**（引用文献的证明；F10） |
| 查询分析攻击防护 | DistanceDP + Theorem 1 扩张；k' 随 r 单调上升（exp5）、检索召回保持（exp2） | VERIFIED（机制层面） |

---

## 10. V Experimental Evaluation

### 10.1 Setup

| 论文 | 复现 | 判定 |
|---|---|---|
| MS MARCO 数据集 | MS MARCO passage dev 子集：m=50,000（1053 qrels + 18,544 BM25 硬负例 + 随机填充）、1000 query；Table III 用 m=100,000（解析+MC） | PARTIAL（规模差异已声明；论文未给检索实验 m） |
| gtr-t5-base 嵌入 | `sentence-transformers/gtr-t5-base`，768 维，L2 归一化 | VERIFIED（A4） |
| 全部实验在 NVIDIA A100 | Apple M5 CPU（10 核/16GB，无 GPU）；每个实验记录硬件，绝对值不跨硬件比较 | PARTIAL（硬件差异已声明） |

### 10.2 Table II（Vec2Text）

| 行 | 论文值 | 复现值（50 条，95% bootstrap CI） | 判定 |
|---|---|---|---|
| w/o Enc | BLEU 83.021 / P .947 / R .950 / F1 .948 / ROUGE .950 | BLEU **69.875** [61.99,78.17] / .924 / .928 / .926 / .860 | PARTIAL（0.84×；绝对基线受论文未给的样本/步数/嵌入约定影响，F6/F14/F15） |
| CAPRISE (s=3,β=0.2) | BLEU 12.425 / P .482 / R .498 / F1 .487 / ROUGE .492 | BLEU **13.696** [9.72,17.48] / .534 / .529 / .531 / .409 | VERIFIED（CI 覆盖论文值；A11 两种攻击口径，rescaled 34.58 为最强攻击上界） |
| steps/beam 敏感性（F6） | 未声明 | 同 5 条：20/4→50/10 明文 55.7→77.4（+21.7）、raw 7.21→7.16（−0.05）、rescaled 29.7→33.0（+3.3） | 明文基线依赖预算；加密后水平不敏感 |

### 10.3 Fig.4（效率）

| 要素 | 论文 | 复现 | 判定 |
|---|---|---|---|
| Fig.4a | embeddings/s 随维度上升而下降 | 论文 15 维全部实测；向量化 drop 3.14×（ρ=−1.00，允许 25% 抖动——论文自身 256→312 上升 24.4%）；paper-style 探针 drop 42.1× vs 论文 44.5× | VERIFIED（趋势+量级，A17/F5） |
| Fig.4b | element-level 吞吐在 1536–3072 维持高位/震荡 | paper-style 探针 elements/s 为论文的 1.19–2.39×（中位 1.36×），1536–1792 高平台 2.47M vs 2.04M；论文 d=256 凹陷未复现（论文 −18.1% vs 探针 +9.5%，已登记） | PARTIAL（量级/高位成立；局部震荡不可复现，F5/A17） |
| Phase-1 | 128 query 嵌入 79.52 ms + 加密 15 ms，overhead <19% | MPS 533.05 ms + 1.42 ms → 0.266%；CPU 756.12 ms → 0.188%；论文混合参照 1.786% | VERIFIED（<19% 成立；绝对值跨硬件不可比） |
| Phase-2 | d=768 CAPRISE 2339 vec/s vs RemoteRAG PHE 250–300（9×） | 43,468 vec/s（向量化，18.6× 论文）；Paillier 1.1525/0.1747 vec/s；论文 9× 内部自洽（7.8–9.4×） | PARTIAL（方向/内部自洽成立；倍数跨实现不可比，F12） |

### 10.4 Fig.5（向量分析 ASR）

| 要素 | 论文 | 复现 | 判定 |
|---|---|---|---|
| 协议 | 随机取库向量作测试 query，比较明文/ADCPE/CAPRISE 排序 | exp4：100 个随机库向量、m=5000；ASR@t + Kendall/Spearman + 距离重建误差 | VERIFIED（协议） |
| 趋势 | CAPRISE ASR 在所有维度低于 ADCPE | 27/27 cells 成立（真实 −2.3pt；合成 −6.1/−6.8/−4.0pt；β 越大差距越大） | VERIFIED（方向） |
| 12 个维度点 | 64…2048 | 12 维合成网格全部跑通；d=768 与论文柱高吻合（83.2/88.2 vs 83.1/87.5） | PARTIAL（论文维度来源未说明，F11/A13） |

### 10.5 Table III（top-k'）

见 §7：**FALSIFIED**（Δα=r 无法复现论文表值；需未声明的 Δα≈1.2r）。

---

## 11. VI Discussions 与 VII Conclusion

| 声明 | 复现核对 | 判定 |
|---|---|---|
| β 不能全局固定，需按数据集/嵌入模型标定 | exp2 β∈{0.05,0.1,0.2,0.4} → recall@10 单调 0.9910→0.9401；exp4 β 扫描 ASR 差距 −3.4→−11.6pt | VERIFIED |
| DP 半径决定隐私-精度 trade-off；半径越大检索集越大 | exp5：r=0.02→k'=72、r=0.05→k'=785；exp2：耗时 3.11→11.81 ms/query | VERIFIED |
| 未来工作：自适应参数选择 | 论文未实现，复现未做 | N/A（future work） |
| 结论：robust defenses（vec2text / query analysis / vector analysis） | M11/M13/M10 已量化；vector analysis 幅度有限（F2/F11） | VERIFIED（方向）/ PARTIAL（幅度） |
| 结论：high retrieval utility | exp2：recall@10 vs 明文 0.9699（k'=k）、0.9999（k'=15）、1.0000（DP） | VERIFIED |
| 结论：processing throughput | exp1（M17） | PARTIAL（绝对量级跨实现不可比） |

---

## 12. 范围外 / 无法复现项（BLOCKED / N/A）

| 要素 | 原因 | 处置 |
|---|---|---|
| §IV 密码学安全证明（IND 类） | 需要完整归约证明；论文未给证明细节（只有断言） | BLOCKED（F10），不声称已验证 |
| DistanceDP 的 (ε,δ)-DP 证明 | 引自 RemoteRAG；本文未给证明 | BLOCKED（机制实现已验证 M12） |
| Algorithm 2 行 25 `Response ← LLM(input)` | 论文未指定 LLM；本机无 GPU/托管 LLM | BLOCKED（仅验证 prompt 组装 M16） |
| Table I 中除 RemoteRAG 外的文献行 | 引用文献条目，非本文贡献 | N/A（文献核对） |
| Fig.1/Fig.2/Fig.3 | 示意图，无数值主张 | N/A（结构核对 + 对应实验支撑） |

---

## 13. 论文未声明的实现选择（A1–A19 索引）

所有歧义与决策见 `docs/PAPER_ANALYSIS.md` §12 与 `SPEC.md` §2.2；与判定直接相关的：
A4（归一化）、A5（β 标定）、A6（ReRank）、A9（nonce 传输）、A10（Table III/Δα）、A11（Vec2Text 攻击口径）、
A12（ASR 定义）、A13（Fig.5 维度来源）、A16（Eq.(12) 符号笔误，F16）、A17（实现风格）、A19（Eq.(3) 存在性）。

论文问题清单（F1–F16）与判定映射：F1/A10（Table III 内部不一致）、F2/A19（Eq.3 存在性）、
F3/A11（Vec2Text 攻击口径）、F4/A18（无检索精度数字）、F5/A17（效率实现风格）、F6（Table II 细节缺失）、
F7/A9（nonce 传输）、F8（gap>β 前提）、F9（定理为上界）、F10（形式化安全未复现）、F11（Fig.5 趋势）、
F12（倍数不可比）、F13（良性语料几乎无损失）、F14（vec2text 嵌入约定）、F15（Table II 绝对对齐）、
F16（证明 Eq.(12) 符号笔误）。

> 维护约定：本文件与 `docs/CLAIMS_CHECKLIST.md`、`REPRO_STATE.md`、`results/*.json` 必须数字一致；
> 任何数字改动先改 JSON/实验，再同步本文件。
