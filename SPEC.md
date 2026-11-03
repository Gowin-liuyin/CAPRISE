# CAPRISE / ppRAG 复现规范 (SPEC)

> 论文: *Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving Encryption*
> arXiv:2601.12331 (2026-01-18), Huanyi Ye, Jiale Guo, Ziyao Liu, Kwok-Yan Lam (NTU / DTC)
> 本地 PDF: `../../doc/References/CAPRISE_2026_Privacy_Preserving_RAG_DPE.pdf`
> LaTeX 源码（含干净公式）: `../../output/pdf/source/caprise/ppRAG_FLLM.tex`

本文件是复现的实现规范：公式、约定、API、实验定义。所有成员写代码前先读本文件。

---

## 1. 论文要点

ppRAG 是一个面向不可信云的两阶段隐私保护 RAG 框架:

- **Phase 1 (Upload)**: 用户本地嵌入文件 → CAPRISE 加密嵌入 + AES 加密原文 → 上传云端。
- **Phase 2 (Retrieve)**: 用户对 query 嵌入先做 DistanceDP 扰动，再 CAPRISE 加密 → 云端返回 top-k' 密文向量与密文文件 → 用户本地解密、重排、取 top-k。
- **Phase 3 (Augment & Generate)**: 检索到的明文拼进 prompt 交给 LLM。

核心贡献 **CAPRISE** (Conditional Approximate Distance-Comparison-Preserving Symmetric Encryption):

- 保留 **查询密文 ↔ 数据库密文** 之间的距离比较顺序（检索需要）。
- **不保留数据库密文之间**的距离比较（防向量结构分析）。
- 相对 ADCPE (Fuchsbauer et al., 2022, SAP 方案) 的改进：ADCPE 噪声半径 `sβ/4` 对所有向量一致，保留全部两两距离；CAPRISE 把数据库侧噪声放宽到 `3sβ/8`，查询侧收紧到 `sβ/8`，从而只保留"查询-数据库"距离顺序。

### 1.1 形式化性质 (论文 Eq. 2, 3)

```
‖eq − e1‖ < ‖eq − e2‖ − β   ⇒   ‖e'q − e'1‖ < ‖e'q − e'2‖        (2) 距离比较保留
‖e1 − e2‖ < ‖e1 − e3‖ − β   ⇏   ‖e'1 − e'2‖ < ‖e'1 − e'3‖        (3) 数据库内部不保留
```

### 1.2 噪声界 (论文 Eq. 7)

```
‖η_{eq}‖ < sβ/8       (查询噪声)
‖λ_{ei}‖ < 3sβ/8      (数据库噪声)
```

---

## 2. Algorithm 1 — CAPRISE 精确公式

LaTeX 源码（`ppRAG_FLLM.tex` 行 211–258）给出的规范伪代码，**逐字实现**:

```
procedure KeyGen:
    s ←$ S                 # 缩放因子，S 为 keyspace，|S| ≤ 2^λ
    K ←$ {0,1}^k           # PRF 密钥
    return (s, K)

procedure EncDB((s,K), e_i):
    r ←$ {0,1}^ℓ
    randbits1 ‖ randbits2 ← PRF(K, r)
    n ← N(0, I_d)(randbits1)        # d 维标准高斯方向
    u ← U_{0,1}(randbits2)          # 均匀标量
    λ_{e_i} ← (3/8) · (n · s · β / ‖n‖) · u^(1/d)
    e'_i ← s·e_i + λ_{e_i}
    return (e'_i, r)

procedure DecDB((s,K), (e'_i, r)):
    randbits1 ‖ randbits2 ← PRF(K, r)
    n ← N(0, I_d)(randbits1)
    u ← U_{0,1}(randbits2)
    λ_{e_i} ← (3/8) · (n · s · β / ‖n‖) · u^(1/d)
    e_i ← (e'_i − λ_{e_i}) / s
    return e_i

procedure EncQ((s,K), e_q):
    r ←$ {0,1}^ℓ
    randbits1 ‖ randbits2 ← PRF(K, r)
    n ← N(0, I_d)(randbits1)
    u ← U_{0,1}(randbits2)
    η_{e_q} ← (1/8) · (n · s · β / ‖n‖) · u^(1/d)
    e'_q ← s·e_q + η_{e_q}
    return (e'_q, r)
```

注意:
- `u^(1/d)` 是"d 维球内均匀采样"标准技巧（ADCPE 原文 Sec 5.1 亦如此），保证噪声范数严格 `< 半径`（`u ∈ (0,1)`）。
- 噪声是 **有界均匀球** 而非高斯；`n` 只提供方向。
- 解密需要 nonce `r` 重放 PRF 才能重建噪声（密文与 nonce 成对存储/传输）。

### 2.1 ADCPE / SAP 基线（对比方法）

Fuchsbauer et al. 2022 (DPEC) Algorithm 2 "Scale-And-Perturb":

```
λ_m ← (sβ/4) · (u/‖u‖) · (x0)^(1/d),   c ← s·m + λ_m
```

即噪声半径 `sβ/4`，其余与 CAPRISE 相同。本复现用它作 **向量分析攻击 (ASR)** 的对照组。

### 2.2 实现约定（我们的实例化选择，需在报告里声明）

| 项 | 论文符号 | 我们的实现 | 说明 |
|---|---|---|---|
| PRF | PRF: {0,1}^k × {0,1}^ℓ → {0,1}* | HMAC-SHA256 计数模式 (TapeGen 风格) | k=128 bit 密钥，任意长输出 |
| randbits1→高斯 | N(0,I_d)(randbits1) | `numpy.random.Generator(PCG64(seed=PRF(K,r‖1)[:16])).standard_normal(d)` | 确定性、解密可重放 |
| randbits2→均匀 | U_{0,1}(randbits2) | `(int(PRF(K,r‖2)[:16]) + 1) / 2^128` ∈ (0,1) | 避免 u=0 |
| keyspace S | \|S\| ≤ 2^λ | 整数集合，默认 S={1,...,2^16}；实验用固定 s=3（论文隐私实验设置） | 可配置 |
| β | 距离分辨参数 | 主实验 β=0.2（论文 Vec2Text 设置），另做敏感性 {0.05,0.1,0.2,0.4} | 论文 Discussion 指出需按数据集标定 |
| ℓ | nonce 长度 | 128 bit | |
| 向量归一化 | GTR 嵌入 | L2 归一化（GTR 要求） | 距离 ∈ [0,2] |

**重要数学验证**（单元测试覆盖）:
1. `DecDB(EncDB(e)) ≈ e`（数值误差 < 1e-5，float64）。
2. 噪声范数界: `‖λ‖ < 3sβ/8`，`‖η‖ < sβ/8`。
3. 性质 (2) 在 gap > β 的随机三元组上 100% 成立（蒙特卡洛大样本）。
4. 性质 (3) 存在反例（CAPRISE 不保留 DB 内部顺序）——用统计比例说明。
5. 密文距离 = s·明文距离 + ‖η−λ‖ 量级扰动。

---

## 3. Algorithm 2 — ppRAG 流程

```
Phase 1: {x_i} → e_i = LLM_e(x_i) → e'_i = EncDB(e_i);  x'_i = AES(x_i);  upload {(e'_i, x'_i)}
Phase 2: q → e_q = LLM_e(q)
         (k', ẽ_q) ← DistanceDP(e_q)               # DP 扰动 + 检索半径扩张
         e'_q ← EncQ(ẽ_q)
         {(e'_i, x'_i)}_{i=1..k'} ← Server.Search(e'_q, k')   # 云端
         {e_i} ← DecDB(e'_i);  ReRank;  Top-k
Phase 3: input = q + {x_1..x_k}; Response = LLM(input)
```

实现约定:
- **ReRank**（论文未指定模型）: 我们采用"客户端解密后用**明文精确距离**对 k' 个候选重排"——这是信息上等价且最有利于精度上限的 rerank，能干净地分离"加密检索损失"与"rerank 模型损失"。报告中明确声明。
- **Server.Search**: 密文空间 L2 距离（等价于 `‖q‖²+‖E'‖²−2q·E'`，向量化 numpy）；返回 top-k'。
- **DistanceDP**: 见 §4。
- Phase 3 的 LLM 生成超出本复现范围（无 GPU/无 LLM 服务）；检索部分作为核心验证对象。报告里说明。

---

## 4. DistanceDP（查询侧差分隐私，来自 RemoteRAG）

RemoteRAG (arXiv:2412.12775) 定义 `(n, ε)-DistanceDP`、机制为:

- 径向扰动 `r ~ Gamma(shape=n, scale=1/ε)`，期望 `r̄ = n/ε`；典型 `r̄ ∈ [0.02, 0.1]`（嵌入已归一化）。
- 方向: n 维单位球面均匀方向。
- 扰动: `ẽ_q = e_q + r·direction`（‖ẽ_q − e_q‖ = r）。
- "perturbed angle Δα_k 可近似为 r"（RemoteRAG 原文），据此得 Theorem 1 扩张检索集。

我们实现两种调用方式:
1. `radius="gamma"` + `eps`: 论文/RemoteRAG 的原始机制（质量随机）。
2. `radius=r` 固定: 用于 Table III / 消融实验（确定性半径）。

DP 记账: `(n,ε)-DistanceDP` 的 ε 作为隐私预算报告；同时给出扰动半径 r 的分布（论文 Table III 用 r 作为自变量）。

---

## 5. Theorem 1（检索集扩张，论文 Eq. 21）

```
k' = k + m · (Ω_{n-1}(π)/Ω_n(π)) · ∫_{α_k}^{α_k'} sin^{n-2}θ dθ
```

其中:
- 均匀 n 维单位球面上，方向与查询夹角 θ 的分布: `F(α) = P(θ ≤ α) = 1 − I_{(1+cos α)/2}((n−1)/2, (n−1)/2)`（正则化不完全 Beta 函数），**数值稳定实现必须用 Beta 而非直接积分 sin^{n-2}（会下溢）**。
- `α_k` 由 `F(α_k) = k/m` 反解（`betaincinv`）。
- `Ω_n(π) = 2π^{n/2}/Γ(n/2)`；比值 `Ω_{n-1}/Ω_n = Γ(n/2)/(√π·Γ((n−1)/2))`。
- `Δα_k` = 噪声造成的角度增量。

**复现发现（待实验确认，已初验）**: 论文 Table III 的 k' 数值无法用 RemoteRAG 的 `Δα_k ≈ r` 约定复现（我们的值约为论文的 0.55–0.75 倍）；论文表值对应 `Δα_k ≈ 1.2r`。exp5 需:
- 给出 `Δα=r` 与论文 Table III 的并排对比；
- 反推每格所需 `Δα/r`；
- 用 Monte Carlo（实际随机向量扰动）验证正确约定；
- 在报告中把这一不一致作为复现结论之一。

---

## 6. 实验定义（对应论文图表）

| 实验 | 论文位置 | 内容 | 产物 |
|---|---|---|---|
| exp1 | Fig. 4, §V-1 | CAPRISE 加密吞吐 vs 嵌入维度; Phase-1 overhead (128 query: 79.52ms 嵌入 + 15ms 加密, <19%); Phase-2 throughput (2339 vec/s @768 vs RemoteRAG PHE 250–300) | results/exp1_*.json, figures/fig4_* |
| exp2 | §V (无专门表格，声称 high retrieval accuracy) | MS MARCO 检索精度: plaintext / CAPRISE / CAPRISE+DP / ADCPE；Recall@10 vs 明文、qrels MRR；β 敏感性 | results/exp2_*, figures/fig_retrieval |
| exp3 | Table II | Vec2Text 重建攻击: 明文 vs CAPRISE(s=3,β=0.2) 的 BLEU/P/R/F1/ROUGE。论文值: 83.021/0.947/0.950/0.948/0.950 → 12.425/0.482/0.498/0.487/0.492 | results/exp3_*, figures/fig_vec2text |
| exp4 | Fig. 5 | 向量分析攻击 ASR: ADCPE vs CAPRISE，随维度变化（越低越好） | results/exp4_*, figures/fig5_* |
| exp5 | Table III + Theorem 1 | top-k' 扩张表 + Monte Carlo 验证 + Δα/r 反推 | results/exp5_*, figures/fig6_* |

**硬件现实**: 无 GPU（Apple M5, 10 核, 16GB）。所有实验必须报告真实测得的硬件与规模，并与论文 A100 设置对照，差异写进 notes/report。禁止编造数字。

---

## 7. 代码结构与 API

```
src/caprise/
  __init__.py
  prf.py         # PRF(K, r, outlen) HMAC-SHA256 计数模式；coins(K,r,d)
  caprise.py     # KeyGen, EncDB, DecDB, EncQ, EncDBBatch；ADCPE(SAP) 同类 API
  dp.py          # distance_dp(e_q, radius=..., eps=...) -> (perturbed, r_used)
  theorem.py     # cap_fraction, alpha_for_fraction, k_prime, table_III
  server.py      # PlaintextDB, EncryptedVectorDB (search top-k')
  pprag.py       # PPRA GClient: upload / retrieve / (augment stub)
  aescipher.py   # AES-GCM 加解密文档
  metrics.py     # Recall@k, MRR, nDCG, BLEU/P/R/F1/ROUGE, ASR/Kendall
```

统一 API（供各实验 import）:

```python
from caprise.caprise import KeyGen, EncDB, DecDB, EncQ, ADCPE
from caprise.server import EncryptedVectorDB, PlaintextDB
from caprise.dp import distance_dp
from caprise.theorem import k_prime, cap_fraction
from caprise import metrics

key = KeyGen(s=3, beta=0.2, dim=768)          # -> CryptoKey(s, K, beta, dim)
E_prime, nonces = EncDB(key, E)               # E: (m,d) float64 -> (m,d), list[bytes]
q_prime, nonce = EncQ(key, q)                 # q: (d,) -> (d,), bytes
E_hat = DecDB(key, E_prime, nonces)           # 解密还原
E_hat = ADCPE.encrypt(key, E)                 # 对照组 (噪声半径 sβ/4)
```

`caprise` 包以 `src/` 为根，实验脚本 `sys.path.insert(0, "src")` 后 `import caprise`。

---

## 8. 验收标准（每个实验）

1. 数字真实、可复跑（固定随机种子、记录命令）。
2. 与论文对应数值/趋势对照，写清偏差与原因。
3. 产物: results/*.json（机器可读）+ notes/*.md（人读说明）+ figures/*（能出图就出）。
4. 不修改他人写作用域；核心模块只由 Lead 修改。
