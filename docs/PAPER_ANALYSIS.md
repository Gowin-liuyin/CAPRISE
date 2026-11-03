# CAPRISE / ppRAG 论文逐行分析

> 论文: *Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption*
> arXiv:2601.12331v1 [cs.CR], 2026-01-18, Huanyi Ye, Jiale Guo, Ziyao Liu, Kwok-Yan Lam (NTU / DTC)
>
> 分析材料优先级：
> 1. **LaTeX 源码** `../../output/pdf/source/caprise/ppRAG_FLLM.tex` —— 公式最干净，逐行分析的基准
> 2. PDF `../../doc/References/CAPRISE_2026_Privacy_Preserving_RAG_DPE.pdf`
> 3. 提取文本 `../../output/pdf/work-in-progress/CAPRISE_原文提取.txt`（双栏混排，仅作定位参考，行号记为 L***）
>
> 标注约定：**[原文]** = 论文原句/公式；**[含义]** = 数学解读；**[实现]** = 本复现的对应实现；**[验证]** = 所属模块门禁。

---

## 0. 论文结构总览

| 章节 | 内容 | 复现地位 |
|---|---|---|
| Abstract / I Introduction | 动机、贡献三点 | 背景，需核对贡献声明与实验对应关系 |
| II Related Work + Table I | RemoteRAG / DP-RAG 对比 | 决定基线选择（ADCPE、PHE） |
| III-A Threat Model & Intuition | 两参与方、honest-but-curious、三类攻击 | 定义实验威胁模型 |
| III-B Algorithm 1 (CAPRISE) | KeyGen / EncDB / DecDB / EncQ | **核心模块 M1–M9** |
| III-B1 证明 (Eq.2–20) | 距离比较保留/不保留的证明 | **性质门禁 M7/M8** |
| III-B2 DistanceDP + Theorem 1 (Eq.21) | 查询 DP 与检索集扩张 | **模块 M12/M13** |
| Algorithm 2 (ppRAG) | Phase 1/2/3 | **模块 M14–M16** |
| IV Security Analysis | 安全性论证（继承 ADCPE） | 定性核对，不能形式化验证 |
| V Experiments | Table II/III, Fig.4/5 | **模块 M10/M11/M17/M18** |
| VI Discussion / VII Conclusion | β 需按数据标定等 | 报告讨论 |

---

## 1. 摘要与引言 (L1–L108)

**[原文 L24–L60]** 提出 ppRAG，核心是 CAPRISE：加密嵌入但仍允许云端计算"加密查询 vs 加密数据库嵌入"的相似度；**只保留查询-数据库的相对距离顺序，不暴露库内距离**；并用 DP 扰动查询防查询分析攻击。

**[原文 L90–L108]** 贡献 3 点：
1. ppRAG 框架（外包加密数据库上的高效隐私保护 RAG）；
2. CAPRISE 加密方案（条件距离比较保留 + 混淆内部关系，防 Vec2Text 与向量分析）；
3. 理论证明 + 安全分析 + 实验评估。

**[复现映射]** 贡献 3 直接对应：M1–M9（方案）、M14–M16（框架）、M7/M8/M13（理论）、M10/M11/M17/M18（实验）。
**[注意]** 摘要/引言声称 "defends against vector-to-text attack, vector analysis, and query analysis"，但**未给出检索精度数字**——可复现性缺口，本复现在 exp2 补测。

---

## 2. 相关工作与 Table I (L110–L124, L66–L89)

**[原文 Table I]** 对比 Cloud / Local / DP / HE / OT / ADCPE 六列：
- 文献 [11]–[19]：Local ✓
- RemoteRAG [6]：Cloud ✓, Local ✓, DP ✓, HE ✓, OT ✓
- DP-RAG [20]：Local ✓, DP ✓
- ppRAG*：Cloud ✓, Local ✓, DP ✓, ADCPE ✓

**[含义]** 论文把自身定位为"Cloud 部署 + DP + ADCPE（无 HE/OT）"。
**[实现]** 基线选择：
- ADCPE（SAP，Fuchsbauer 2022）→ 向量分析对照（M9/M10）
- PHE（RemoteRAG 用）→ 效率对照（M17，Paillier 微基准）
- DP（DistanceDP）→ 查询扰动（M12）
**[注意]** 表中 RemoteRAG 同时有 Cloud 与 Local ✓ 是"数据在本地/云端两种模式"，与其 PHE 检索路径一致；这不影响我们复现。

---

## 3. Threat Model 与设计直觉 (III-A, L159–L203)

**[原文 L162–L170]** 两个参与方：
- **User**：本地存储有限，外包私有数据（个人文件）到云；本地嵌入 + 加密后上传；查询时本地嵌入、上传 query embedding、取回 top-k 加密文件并本地解密。
- **Cloud Server**：honest-but-curious —— 正确执行所有操作，但不可信、可能推断敏感信息。

**[原文 L171–L180]** 三类风险：
1. **Vec2Text / vector-to-text 反演攻击**：明文嵌入可被反演回原文（Morris et al. 2023;[21][22]）→ 只加密文件（AES）不够，因为检索需要嵌入；
2. **Vector analysis 攻击**：分析库内嵌入两两距离可推断文件间语义关系，**无需查询**；
3. **Query analysis 攻击**：重复查询可暴露频繁访问模式/查询频率。

**[原文 L181–L186]** 设计目标：保护嵌入语义内容 **且** 保护嵌入间关系信息，**即使在重复访问模式下**。

**[复现映射]**
- 威胁模型 → 实验设置：服务器只见 `(e'_i, nonce, AES_ct)` 与 `e'_q`；攻击者不持有 (s,K)。
- 攻击 1 → M11（Vec2Text，Table II）
- 攻击 2 → M10（向量分析 ASR，Fig 5）
- 攻击 3 → M12/M13（DistanceDP + Theorem 1，Table III）
**[验证]** 威胁模型假设检查：所有实验脚本必须断言"服务器侧无明文嵌入/无明文文档"。

---

## 4. Algorithm 1 —— CAPRISE 逐行分析

LaTeX 源码 `ppRAG_FLLM.tex` L211–258。以下逐行。

### 4.1 KeyGen（行 1–5）

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 1 | procedure KeyGen | 生成密钥 | `caprise.KeyGen` | M2-G1 |
| 2 | `s ←$ S` | 从 keyspace 采样缩放因子；正文说 `\|S\| ≤ 2^λ`，λ 为安全参数 | `s ~ U{1..2^16}`，实验固定 s=3 | M2-G2 |
| 3 | `K ←$ {0,1}^k` | PRF 密钥，k 未给定 | k=128 bit（16 字节） | M2-G3 |
| 4 | `return (s,K)` | — | `CryptoKey(s, K, beta, dim)` | M2-G4 |

**[歧义 A1]** 论文未给 S 的大小/分布/λ 取值 → 实现取整数区间，并在报告中声明（与 ADCPE 的 S 整数 keyspace 一致）。
**[歧义 A2]** k 未给定 → 128 bit（AES/PRF 通用安全强度）。

### 4.2 EncDB（行 6–14）

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 6 | procedure EncDB((s,K), e_i) | 数据库嵌入加密 | `EncDB` / `EncDB_one` | M4 |
| 7 | `r ←$ {0,1}^ℓ` | 每向量独立 nonce；ℓ 未给定 | ℓ=128 bit；nonce 随密文返回 | M4-G1 |
| 8 | `randbits1‖randbits2 ← PRF(K,r)` | 一次 PRF 输出切两段随机流 | HMAC-SHA256 计数模式输出 32B 切两半 | M1 |
| 9 | `n ← N(0,I_d)(randbits1)` | d 维标准高斯方向（**不归一化**，归一化在公式内） | PCG64(seed=randbits1).standard_normal(d) | M1-G2 |
| 10 | `u ← U_{0,1}(randbits2)` | 均匀标量；**必须严格 > 0**（否则 u^{1/d}=0，虽然合法但退化） | `(int(randbits2)+1)/2^128 ∈ (0,1)` | M1-G2 |
| 11 | `λ_ei ← (3/8)·(n·s·β/‖n‖)·u^{1/d}` | **噪声向量核心**：方向=n/‖n‖（球面均匀）；模长=(3/8)sβ·u^{1/d}；u^{1/d} 是 d 维球内均匀采样技巧 | `_noise(key,r,3sβ/8)` | M3-G1/M4-G2 |
| 12 | `e'_i ← s·e_i + λ_ei` | Scale-And-Perturb | `key.s*e + λ` | M4 |
| 13 | `return (e'_i, r)` | 密文 + nonce | `(m,d)` 矩阵 + `list[bytes]` | M4-G3 |

**[关键分析]**
- 模长严格 `< 3sβ/8`（u<1），满足正文 Eq.(7) 的严格上界；这正好是证明所需（证明里用 `<`）。
- 方向由高斯归一化给出 → 等价于球面均匀；模长独立均匀 `u^{1/d}` → 联合分布 = d 维球内均匀（ADCPE 原文 Sec.5.1 明示此技巧）。
- **相对 ADCPE 的关键改动**：ADCPE 用 `sβ/4`；CAPRISE 数据库侧用 `3sβ/8`（=1.5 倍）。噪声更大 ⇒ 破坏库内距离顺序（Eq.3），但仍保留查询-库顺序（Eq.2）。
- β 的量纲与嵌入尺度有关；论文实验 β=0.2、GTR 嵌入（L2 归一化后两两距离 ∈[0,2]）⇒ β=0.2 是"相对距离分辨力"。

### 4.3 DecDB（行 15–22）

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 15 | procedure DecDB((s,K),(e'_i,r)) | 解密 | `DecDB` / `DecDB_one` | M5 |
| 16–18 | 同 EncDB 行 8–10 | 重放 PRF 重建 n,u | `coins(K,r,d)` | M5-G1 |
| 19 | `λ_ei ← (3/8)·(n·s·β/‖n‖)·u^{1/d}` | 重建同一噪声（确定性） | 同 `_noise` | M5-G1 |
| 20 | `e_i ← (e'_i − λ_ei)/s` | 去噪+去缩放 | 同 | M5-G2 |
| 21 | `return e_i` | — | — | M5-G2 |

**[分析]** 解密安全性完全依赖 PRF：nonce 公开，噪声由 (K,r) 唯一决定；服务器没有 K。
**[歧义 A9]** 论文 Algorithm 2 只写"服务器返回 (e'_i, x'_i)"，未提 nonce 也要返回；但 DecDB 需要 r ⇒ 实现中 nonce 必须与密文一同存储/下发（报告里声明）。

### 4.4 EncQ（行 23–31）

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 23 | procedure EncQ((s,K), e_q) | 查询嵌入加密 | `EncQ` | M6 |
| 24–27 | 同 EncDB 行 7–10 | 独立 nonce/PRF/噪声 | `coins` | M6-G1 |
| 28 | `η_eq ← (1/8)·(n·s·β/‖n‖)·u^{1/d}` | 查询噪声半径 = sβ/8（= 库侧 1/3） | `_noise(..., sβ/8)` | M6-G2 |
| 29 | `e'_q ← s·e_q + η_eq` | — | 同 | M6 |
| 30 | `return (e'_q, r)` | — | `(d,), bytes` | M6-G3 |

**[关键分析]** Eq.(7): `‖η‖ < sβ/8`, `‖λ‖ < 3sβ/8`。证明 Eq.(2) 用 `‖η‖+‖λ‖ < sβ/2 = (1/8+3/8)sβ` —— 这是整个构造的"预算分配"：查询噪声小、库噪声大，恰好让"查询-库"比较在 gap>β 时稳定，而"库-库"比较被大噪声破坏（Eq.3 用 `‖λ1‖+‖λ3‖ < 3sβ/4`）。

### 4.5 ADCPE / SAP 基线（对比方法，Fuchsbauer 2022 Sec.5.1）

| 项 | [原文] | [实现] |
|---|---|---|
| 噪声 | `λ_m ← (sβ/4)(u/‖u‖)(x0)^{1/d}`，即球半径 sβ/4 | `ADCPE.Enc/EncDB/EncQ/Dec`，半径 `sβ/4` |
| 性质 | β-DCP：`‖x−y‖<‖y−z‖−β ⇒ ‖f(x)−f(y)‖<‖f(y)−f(z)‖` | M9 门禁：成对顺序保留率 ≈100% |

**[注意]** ADCPE 的证明要求 `‖λ‖ < sβ/4` **每向量**；CAPRISE 库侧 3sβ/8 故意超界 ⇒ 正是"条件化"（conditional）的由来。

---

## 5. 形式化性质与证明逐行 (Eq.2–20, L271–L368)

### 5.1 Claim (Eq.2, Eq.3)

```
‖eq − e1‖ < ‖eq − e2‖ − β   ⇒   ‖e'q − e'1‖ < ‖e'q − e'2‖      (2)
‖e1 − e2‖ < ‖e1 − e3‖ − β   ⇏   ‖e'1 − e'2‖ < ‖e'1 − e'3‖      (3)
```

**[含义]** (2) 是"近似"距离比较保留（gap 必须 > β）；(3) 是**存在性否定**：库内比较不再保证。
**[实现]** 性质门禁 M7（Eq.2 蒙特卡洛 100% 通过）、M8（Eq.3 找出反例，统计反转率）。

### 5.2 逐条不等式核对（证明无漏洞，但存在细节假设）

| 式 | 原文 | 核对 |
|---|---|---|
| (5) | `e'_q = s e_q + η_eq` | 与 EncQ 行 29 一致 |
| (6) | `e'_i = s e_i + λ_ei` | 与 EncDB 行 12 一致 |
| (7) | `‖η‖ < sβ/8, ‖λ‖ < 3sβ/8` | 由 u<1 保证严格小于；M3/M4 门禁验证 |
| (8) | `‖e'_q − e'_1‖ ≤ ‖η‖ + s‖eq−e1‖ + ‖λ‖` | 三角不等式（把 s·eq−s·e1 与噪声拆开），正确 |
| (9) | `< ‖η‖+‖λ‖ + s(‖eq−e2‖−β)` | 代入假设 `‖eq−e1‖ < ‖eq−e2‖−β` |
| (10) | `< sβ/8 + 3sβ/8 + ‖seq−se2‖ − sβ` | 代 (7) |
| (11) | `= ‖seq−se2‖ − sβ/2 < ‖seq−se2‖ − (‖η‖+‖λ_e2‖)` | 用 sβ/2 = sβ/8+3sβ/8 与 (7) 严格性 |
| (12) | `≤ ‖(seq−se2) − (η−λ_e2)‖ = ‖e'_q−e'_2‖` | 反向三角不等式，正确 |
| (13) | `‖e'1−e'2‖ < ‖λ1‖+‖λ2‖+s(‖e1−e3‖−β)` | 用假设 `‖e1−e2‖<‖e1−e3‖−β` |
| (14) | `< 3sβ/8+3sβ/8 + ‖se1−se3‖ − sβ` | 代 (7) |
| (15) | `= ‖se1−se3‖ − sβ/4 − δ` | 定义 δ>0 吸收严格不等式余量（**存在性**，不是可计算常数） |
| (16)–(19) | `‖e'1−e'3‖ ≥ ‖se1−se3‖ − ‖λ1−λ3‖ > ‖se1−se3‖ − 3sβ/4` | 反三角 + 三角，正确 |
| (20) | `= ‖se1−se3‖ − 3sβ/4 + δ'` | 同理 |
| 结论 | `δ+δ' < sβ/2 ⇒ ‖e'1−e'2‖ > ‖e'1−e'3‖` | 代入 (15)(20) 线性比较即可 |

**[分析结论]**
1. 证明 (2) 依赖"两两独立噪声 + 有界球"，逻辑成立；**它是充分非必要条件**：gap ≤ β 的比较可能反转。
2. 证明 (3) 只证"存在 δ,δ' 的反例"，不保证"经常反转"。**实验中要测反转率，而非假设 100% 反转**。
3. **[复现发现 A19]** 进一步量化：噪声方向在 d 维球面均匀 ⇒ 三个向量的噪声要同时"恰好对齐"各自的差向量才能制造反转，该事件概率随 d 指数衰减（≈ e^{-Ω(d)}）。M8 门禁实测：在满足前提（gap>β）的 20,000 组三元组上，CAPRISE 与 ADCPE 的**实测反转率都是 0**；而合法的对抗性噪声样本确实可以制造反转（M8 构造验证通过）。因此 Eq.(3) 作为数学命题成立，但"库内顺序被破坏"在高维实际采样下几乎不发生。CAPRISE vs ADCPE 的真实可测差异是**各向同性噪声对距离的投影扰动** ‖λ‖/√d：M8 测得距离重建误差 CAPRISE≈0.0046 vs ADCPE≈0.0025（约 1.8×，与半径比 1.5× 一致）。→ 向量分析实验 (exp4) 必须报告 ASR 差值的真实大小，不能假设论文 Fig.5 的差距成立。
3. 证明中 δ, δ' 不可显式计算 ⇒ 不能给出"保证"的检索精度界；只能实验测。
4. 证明 (11) 的 `≤ ‖η−λ‖` 用了 `‖η‖+‖λ‖ ≥ ‖η−λ‖`，与 (7) 的严格不等式配合得到严格 `<`，逻辑无漏洞。

### 5.3 [歧义 A4] 嵌入归一化

论文未显式说嵌入归一化。GTR（gtr-t5-base）标准用法是 L2 归一化（否则内积≠余弦），且 β=0.2 只有在归一化尺度下才合理。**实现：L2 归一化**，报告声明。

---

## 6. DistanceDP 与 Theorem 1（III-B2, L370–L457）

### 6.1 DistanceDP（来自 RemoteRAG, arXiv:2412.12775）

[原文 L378–L387, L406–L409]：采用 DistanceDP："adds calibrated noise to the query embedding e_q and expanding the retrieval space"（加噪声 + 扩张检索集）。

[RemoteRAG 原文 Sec.3.2.1–3.2.2]（我们下载核对的细节，arXiv:2412.12775）：
- `(n,ε)-DistanceDP` 定义：`L(K(x), K(x')) ≤ ε‖x−x'‖`
- 机制：径向 `r ~ Gamma(shape=n, scale=1/ε)`，`E[r] = n/ε`；方向为 n 维单位球面均匀方向
- 归一化嵌入下典型 `r ∈ [0.02, 0.1]`，对应 ε ∈ [10n, 50n]
- Lemma 1：`k = N·(Ω_{n-1}/Ω_n)·∫_0^{α_k} sin^{n-2}θ dθ`；Theorem 1：`k'−k = N·(Ω_{n-1}/Ω_n)·∫_{α_k}^{α_k'} sin^{n-2}θ dθ`
- §3.2.2 原文："Since the perturbation is small (r ≪ 1), the perturbed angle Δα_k between e_k and e_k′ can be approximately by r."
  **注意：CAPRISE 本文既未复述该近似，也未定义 Δα_k 与 r 的换算**——这是 A10 的根源。

**[实现]** `caprise/dp.py`: `distance_dp(e_q, radius="gamma", eps)` → `e_q + r·direction`，返回 (ẽ_q, r)。
**[门禁 M12]** 半径分布均值 ≈ n/ε；方向球面均匀（|cos| 均值≈0、方差≈1/(n-1)）；‖ẽ−e‖=r。

### 6.2 Theorem 1（Eq.21）逐项

```
k' = m · (Ω_{n-1}(π)/Ω_n(π)) · ∫_{α_k}^{α_k'} sin^{n-2}θ dθ + k
Ω_n(π) = 2π^{n/2} / Γ(n/2)
α_k' = α_k + Δα_k
```

**[含义]** 均匀 n 维单位球面上，落入角区间 (α_k, α_k'] 的比例 = 该积分；乘以库大小 m 得期望新增候选数。
**[推导核对]** `Ω_{n-1}/Ω_n = Γ(n/2)/(√π Γ((n-1)/2))`，且 `(Ω_{n-1}/Ω_n)∫_0^π sin^{n-2} = 1` ⇒ 公式就是"球冠面积占比 × m"。**公式本体正确**。
**[数值陷阱]** 直接数值积分 `∫ sin^{n-2}` 在 n≥768 时下溢（sin^{766}(1.43)≈e^{-7.9} 尚可，但尾部区间会到 e^{-3000}）；必须用 Beta 分布：
`F(α) = P(θ≤α) = 1 − I_{(1+cos α)/2}((n−1)/2,(n−1)/2)`，`α_k = F^{-1}(k/m)`。
**[歧义 A10] 关键发现**：论文 §V-4 只说"perturb the query embedding with r"，**未定义 Δα_k 与 r 的换算**。M13 对 Table III 16 格系统枚举四种约定：
(a) `Δα=r`（RemoteRAG §3.2.2）→ 论文的 0.444–0.737×（相对误差 −26%…−56%）；
(b) 弦长 `Δα=2asin(r/2)` → 与 (a) 差 <0.02%，同样低估；
(c) 欧氏扩张 `α'=2asin((2sin(α_k/2)+r)/2)` → 高估 +12%…+71%；
(d) 逐格反解 → 精确复现（≤5.6e-14），`Δα/r = 1.1537–1.2977`（mean 1.2129±0.0435）。
三种可推导约定逐格取最优仍偏离 ≥12% ⇒ 论文表值只能由**未声明**的 `Δα≈1.2r` 复现。
参照实现交叉验证：RemoteRAG 的高亮例（k=5,r=0.03,n=768,k'=160）在 N=1e6 下 `Δα=r` 给 170.34（+6.46%），点态反解 `Δα/r=0.9804` ⇒ 被引实现按 `Δα≈r`，与论文表值不同。详见 M13/exp5 报告。
**[验证]** 解析 vs Monte Carlo（预计算随机向量，真实扰动测 k'）。

---

## 7. Algorithm 2 —— ppRAG 流程逐行 (LaTeX L400–437)

### 7.1 Phase 1: Data Preparation and Upload

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 2 | Input {x_1..x_m} | 明文文档 | `texts` | M14-G1 |
| 3 | `e_i ← LLM_e(x_i), i∈{1..n}` | 嵌入（索引笔误 n/m） | embedder 可注入；exp2 用 gtr-t5-base | M14-G2 |
| 4 | `(s,K) ← KeyGen` | — | `KeyGen(dim,beta,s)` | M2 |
| 5 | `e'_i ← EncDB((s,K), e_i)` | 加密嵌入 | `EncDB` | M4 |
| 6 | `k_AES ← KeyGen_AES` | AES 密钥 | `AESKey.generate()` | M14-G3 |
| 7 | `x'_i ← Enc_AES(k_AES, x_i)` | 加密原文（模式未指定） | AES-256-GCM | M14-G3 |
| 8 | Upload {(e'_i, x'_i)} | 实际上传还需 nonces | `PPRAGServer.upload` | M14-G4 |

### 7.2 Phase 2: Query and Retrieval

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 12 | `e_q ← LLM_e(q)` | 查询嵌入 | 同嵌入器 | M15-G1 |
| 13 | `k', ẽ_q ← DistanceDP(e_q)` | DP 扰动 + 由 Theorem 1 得 k' | `distance_dp` + `theorem.k_prime` | M12/M13 |
| 14 | `e'_q ← EncQ((s,K), ẽ_q)` | 加密扰动后查询 | `EncQ` | M6 |
| 15 | send e'_q | — | 只发密文 | M15-G2 |
| 16 | `{(e'_i,x'_i)} ← Database_Search(e'_q, k')` | **服务器**密文空间 top-k' | `EncryptedVectorDB.search` | M15-G3 |
| 17 | `{e_i} ← DecDB((s,K),(e'_i,r))` | 本地解密候选嵌入（需 nonce） | `DecDB_one` / `DecDB` | M5/M15-G4 |
| 18 | `Reranked ← ReRank{...}` | 重排（模型未指定） | **明文精确距离重排**（声明） | M15-G5 |
| 19 | top-k → 输出 | — | 返回 ids/docs | M15-G5 |

**[歧义 A6]** ReRank 模型未指定（常见是 cross-encoder）。我们选择"解密后用明文精确距离重排"：它给出**加密检索损失的上界最优 rerank**，把"加密噪声损失"与"rerank 模型损失"解耦；报告中声明并说明真实系统可换 cross-encoder（只会更差或持平）。
**[分析]** 注意 k' 由 DP 半径决定（Theorem 1）；无 DP 时 k'=k（但加密噪声仍可能让 k 个中部分错位——所以严格实现应 k'=k 且直接取 top-k；exp2 会对比无 DP 时 k'=k 的精度损失）。

### 7.3 Phase 3: Prompting the LLM

| 行 | [原文] | [含义] | [实现] | [验证] |
|---|---|---|---|---|
| 24 | `input ← q + {x_1..x_k}` | 拼接 prompt | `phase3_build_prompt` | M16-G1 |
| 25 | `Response ← LLM(input)` | LLM 生成 | **超出复现范围**（无 LLM 服务/GPU），只验证 prompt 组装 | — |

**[范围声明]** 论文核心声明是检索与隐私；生成质量依赖具体 LLM（未指定）。本复现验证 Phase 1/2 + prompt 组装，不测生成质量。

---

## 8. 安全分析 (IV, L420–L457) 逐条核对

| [原文] | 核对/复现 |
|---|---|
| CAPRISE 继承 ADCPE 安全性质；条件化"does not compromise the original security guarantees" | **定性声明**。我们无法形式化验证 IND 类安全（需要密码学证明），但可验证：密文统计不可区分性 sanity + Vec2Text 反演失败（M11）。报告如实说明局限。 |
| 噪声标定使 Vec2Text 敏感失效（minor perturbations 破坏重建） | M11：攻击实验验证（Table II 复现） |
| DistanceDP 满足 (ε,δ)-DP w.r.t. 查询嵌入空间 | 引用 RemoteRAG 机制；我们验证机制实现（M12），不重做 DP 证明 |
| 查询分析攻击防护 | M13/Table III：k' 扩张 = 隐私/精度 trade-off |

---

## 9. 实验逐项分析 (V, L459–L517)

### 9.1 Setup

[原文] MS MARCO 数据集 [24]；嵌入模型 **gtr-t5-base** [25]；全部实验在 **NVIDIA A100** 上。
**[复现]** MS MARCO dev 子集（m=50k 起步），gtr-t5-base，**Apple M5 CPU（无 GPU）** —— 效率数字不可直接比，报告声明。

### 9.2 Table II：Vec2Text 重建攻击 (L466–L489)

[原文] 比较加密前后嵌入被 Vec2Text 反演的质量：BLEU / Precision / Recall / F1 / ROUGE。
- w/o Enc: **83.021 / 0.947 / 0.950 / 0.948 / 0.950**
- CAPRISE: **12.425 / 0.482 / 0.498 / 0.487 / 0.492**
- 参数："secure parameter setting of s = 3, β = 0.2"

**[分析]**
- BLEU 0–100（sacrebleu 口径）；P/R/F1 token 级 0–1；ROUGE 0–1（大概率 ROUGE-L F1）。
- 加密后嵌入 = 3e + λ（‖λ‖≤0.225），攻击者需适配（归一化后攻击对攻击者最有利 → 我们主表用归一化攻击，保守）。
- **[歧义 A11]** 论文未说明攻击如何处理加密向量 → our exp3 记录 raw/normalized 两种。
- **[验证 M11]** 基线 BLEU 应显著高、CAPRISE 明显下降；绝对值受 CPU/模型差异影响，报告如实对比。

### 9.3 Table III：top-k' 扩张 (L540–L557)

[原文] m=100,000；n=768/1536；r=0.033/0.02；k=5,10,15,20 → 给 top-k' 与 k'/k。
**[复现发现 A10]** 论文未给 r→Δα_k 换算；M13 枚举四种约定后确认：按被引用来源 RemoteRAG 的 Δα≈r 代入 Eq.(21) 只得论文的 0.444–0.737×；弦长约定与之一致；欧氏扩张约定反向高估 12–71%；只有未声明的逐格反解 `Δα/r=1.1537–1.2977`（mean 1.213±0.044）才能精确复现表值。参照实现 RemoteRAG 的高亮例（k'=160）反而与 Δα=r 自洽（+6.46%）——即**表值与被引用实现的约定不一致**。这是本复现最重要的"论文内部不一致/声明缺失"发现，需要如实报告，不强行凑数。

### 9.4 Fig.4：效率 (L410–L439, L464–L508)

[原文]
- 4a: 加密吞吐（embeddings/s）随嵌入维度增大而下降；
- 4b: 元素级吞吐（element/s）在 1536–3072 维度维持高位/震荡；
- Phase 1: 128 query 嵌入 ≈ 79.52 ms，CAPRISE 加密 +15 ms，overhead < 19%；
- Phase 2: d=768 时 CAPRISE **2339 vectors/s**；RemoteRAG PHE **250–300 vectors/s** → **9x**。

**[分析]**
- 2339 vec/s @768 ≈ 1.8M elements/s，对应 Python 逐元素实现；我们的 numpy 向量化实现会更快（实现差异，报告如实说明，并给出"若按论文实现风格"的对照）。
- Phase-1 overhead 与硬件强相关；我们同时报告：加密耗时（真测）、嵌入耗时（真测，CPU）、比值。
- **[验证 M17]**

### 9.5 Fig.5：向量分析攻击 ASR (L490–L524)

[原文] 随机选一个库向量作测试 query，计算其到其他向量的距离；分别用 ADCPE、CAPRISE 加密数据库后重复计算；与明文排序比较，用 **ASR（Attack Success Rate）** 衡量一致性；CAPRISE 在所有嵌入尺寸上 ASR 更低。
**[分析]** 论文**未给 ASR 公式**（[歧义 A12]）；且测试 query 是库向量（服务器可用的 e'_j，非 EncQ）。我们定义并报告：
- 主指标：ASR@t = |top_t(密文排序) ∩ top_t(明文排序)| / t，t ∈ {1,5,10}
- 辅助：Kendall τ、Spearman ρ（越低泄漏越少）
- 数据：d=768 用真实 GTR 嵌入；d=1536/3072 用各向同性高斯向量（说明理由：GTR 只产 768 维）
**[验证 M10]**

---

## 10. Discussion（VI, L499–L517）与结论

[原文] 关键部署挑战：不同数据集/嵌入模型的特征分布不同，**β 不能全局固定**，需要按数据集标定；DP 扰动半径决定隐私-精度 trade-off；未来工作研究自适应参数选择。
**[复现映射]** 这正是 exp2（β 敏感性）与 exp5（r/k' trade-off）的实验依据；报告中把 β、r 作为可调参数扫描。

---

## 11. 复现映射总表（论文 → 模块 → 门禁）

| 论文要素 | 模块 | 门禁产物 |
|---|---|---|
| PRF 实例化 (Algorithm 1 行 8) | M1 | results/gate_M1_prf.json |
| KeyGen (行 1–5) | M2 | results/gate_M2_keygen.json |
| 球内均匀噪声 (行 11) | M3 | results/gate_M3_noise.json |
| EncDB (行 6–14) | M4 | results/gate_M4_encdb.json |
| DecDB (行 15–22) | M5 | results/gate_M5_decdb.json |
| EncQ (行 23–31) | M6 | results/gate_M6_encq.json |
| Eq.2 距离比较保留 | M7 | results/gate_M7_eq2.json |
| Eq.3 库内不保留 | M8 | results/gate_M8_eq3.json |
| ADCPE/SAP 基线 | M9 | results/gate_M9_adcpe.json |
| 向量分析攻击 (Fig.5) | M10 | results/exp4_vector_analysis.json |
| Vec2Text (Table II) | M11 | results/exp3_vec2text.json |
| DistanceDP (III-B2) | M12 | results/gate_M12_dp.json |
| Theorem 1 / Table III | M13 | results/exp5_query_dp.json |
| Phase 1 上传 (Alg.2 1–9) | M14 | results/gate_M14_phase1.json |
| Phase 2 检索 (Alg.2 10–21) | M15 | results/gate_M15_phase2.json |
| Phase 3 prompt (Alg.2 22–27) | M16 | results/gate_M16_phase3.json |
| 效率 (Fig.4) | M17 | results/exp1_efficiency.json |
| 检索精度 (补充，论文未给) | M18 | results/exp2_retrieval.json |
| MS MARCO 数据 | M0 | results/gate_M0_data.json |
| 报告汇总 | M19 | REPRODUCTION_REPORT.md |

**执行纪律（用户要求）**：按 M0 → M1 → … 顺序；每个模块必须产出 gate 文件且 pass=true，才允许进入下一个模块；任何模块失败必须先修，不允许跳过。

---

## 12. 歧义 / 不一致清单（复现决策）

| # | 问题 | 论文位置 | 复现决策 | 影响 |
|---|---|---|---|---|
| A1 | keyspace S 未定义 | Alg.1 行 2 | 整数 [1,2^16]，实验固定 s=3 | 低 |
| A2 | PRF 的 k, ℓ 未给 | Alg.1 行 3,7 | k=ℓ=128 bit | 低 |
| A3 | 「从 randbits 采样」未定义映射 | Alg.1 行 9–10 | PCG64(128bit seed) 高斯；128bit→(0,1) 均匀 | 低（实现选择） |
| A4 | 嵌入是否归一化未说明 | III, V | L2 归一化（GTR 标准） | 中（影响 β 语义） |
| A5 | β 如何标定未给 | VI Discussion | 主实验 β=0.2 + 敏感性扫描 | 中 |
| A6 | ReRank 模型未指定 | Alg.2 行 18 | 解密后明文精确距离重排 | 中（精度上界） |
| A7 | AES 模式未指定 | Alg.2 行 6–7 | AES-256-GCM | 低 |
| A8 | LLM 未指定 | Alg.2 行 25 | 超出范围，仅 prompt 组装 | 范围声明 |
| A9 | nonce 未提如何返回 | Alg.2 行 16–17 | 与密文同存同传 | 低（必需） |
| A10 | **Table III 与 Δα≈r 不一致；论文未定义 Δα_k 与 r 的换算** | Table III / Eq.21 / §V-4 | 报告 [论文值 \| Δα=r \| 弦长约定 \| 欧氏扩张约定 \| 反解 Δα/r] + MC 验证；`Δα≈r` 引自被引用来源 RemoteRAG §3.2.2 | **高（论文内部不一致）** |
| A11 | Vec2Text 攻击如何适配密文未说明 | Table II | raw + normalized 两种攻击都测 | 中 |
| A12 | ASR 无公式 | Fig.5 | 定义 ASR@t + Kendall τ/Spearman | 中 |
| A13 | Fig.5 的 12 个"嵌入尺寸"（64…2048）超出 GTR 的 768 维，但论文未说明如何得到其他维度 | Fig.5 | 12 个尺寸全部用 L2 归一化高斯合成向量（与论文同分布假设），另用真实 GTR d=768 作锚点；受控维度扫描分离"维度效应"与"数据分布效应" | 中 |
| A14 | 表 III m=100k 与实验数据规模脱节 | Table III | 解析 + MC（不依赖真实库） | 低 |
| A15 | Eq.21 中 α_k 定义隐含 F(α_k)=k/m | Eq.21 | 显式化 | 低 |
| A16 | 证明 Eq.(12) 符号笔误：打印为 `(se_q−se_2) − (η−λ)`，应为 `+`（见 F16） | §III-B1 L317 | 按数学上下文用 `+` 复核；`scripts/check_proof_chain.py` 记录打印式 4000/4000 不成立、改正式 0/4000 违反 | 低（不改变结论） |
| A17 | 效率数字实现风格（逐元素 Python） | §V-1 | 报告实现差异，不强行对齐 | 中 |
| A18 | 论文无检索精度数字 | §V | 补测 exp2 并声明为补充实验 | 中 |
| **A19** | **Eq.(3) 仅存在性可达** | §III-B1 Eq.(13)–(20), Fig.5 | M8 门禁：构造性给出合法噪声反例（命题为真）；但随机采样下 0/20000 反转（高维需噪声方向跨向量对齐，概率 e^{-Ω(d)}）。实际泄漏差异用距离重建误差/ASR 量化，exp4 如实报告 | **高（对 Fig.5 的解读）** |
