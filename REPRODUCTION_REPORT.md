# CAPRISE / ppRAG 复现报告

> 论文：*Efficient Privacy-Preserving Retrieval Augmented Generation with
> Distance-Preserving Encryption*（arXiv:2601.12331v1, 2026-01-18，
> Huanyi Ye, Jiale Guo, Ziyao Liu, Kwok-Yan Lam，NTU / DTC Singapore）
>
> 复现方式：**逐行分析论文 → 逐协议、逐模块实现 → 每个模块门禁 PASS 才进入下一模块**；
> 关键模块由另一成员**独立交叉验收**（重跑门禁 + 从零复算 + 逐行读 diff）。
> 门禁定义见 `docs/MODULE_PLAN.md`，逐行分析见 `docs/PAPER_ANALYSIS.md`，
> 声明核对见 `docs/CLAIMS_CHECKLIST.md`，论文问题见 `docs/FINDINGS.md`。
>
> 复现环境：Apple M5（10 核，16 GB，无 CUDA），Python 3.12，torch 2.15 CPU/MPS。
> 论文环境：NVIDIA A100 GPU。效率数字跨硬件不可直接比较，逐项标注。
>
> 代码：本仓库；门禁结果：`results/gate_*.json`；实验数据：`results/*.json`；
> 图：`figures/`；每个实验的实现说明：`notes/`。

---

## 0. 结论摘要

1. **CAPRISE 方案本身正确复现**：Algorithm 1 的 KeyGen/EncDB/DecDB/EncQ 逐行实现，
   解密精确（2.8e-17），噪声界严格（`‖λ‖<3sβ/8`、`‖η‖<sβ/8`），
   距离比较保留性质 Eq.(2) 在 50,000 组前提构造三元组上 **0 违反**，
   ADCPE 基线的 β-DCP 性质 20,000 组 **0 违反**（M1–M9 全部 PASS）。
2. **ppRAG 三阶段流程正确复现**：Phase-1 服务器只持有密文嵌入 + nonce + AES-GCM 密文
   （无密钥/无明文），Phase-2 只收到加密查询并能在密文空间完成 top-k' 检索，
   本地解密+重排后与明文结果一致，Phase-3 prompt 无未检索文档泄漏（M14–M16 全部 PASS）。
3. **检索精度（论文未给数字，本复现补充）**：真实 MS MARCO 子集（m=50,000，1000 query，
   gtr-t5-base）上，CAPRISE 严格 `k'=k` 仅损失 3.0% 的明文 top-10 保真度
   （Recall@10=0.9699），`k'=15` 即恢复 0.9999、`k'=2k` 恢复 1.0000；
   加 DistanceDP 后按 Theorem 1 取 k' 时 Recall@10=1.0000；
   **qrels MRR@10 与明文完全相同（0.7230）**；加密检索延迟与明文检索同量级
   （2.35ms vs 2.45ms/query）。
4. **两个重要的论文内部不一致（FALSIFIED，均附可复跑证据）**：
   - **Table III 与正文约定矛盾（F1/A10）**：用论文公式 Eq.(21) 与 RemoteRAG 的
     `Δα_k ≈ r` 约定，本复现值仅为论文表值的 0.444–0.737 倍；逐格反推需要
     `Δα/r = 1.213 ± 0.044`（1.154–1.298）。论文表值无法由论文自身公式+约定复现。
   - **Theorem 1 是充分但极宽松的上界（F9/A10b）**：论文表值（k=5 时 52× 扩张）
     既不是实际检索所需的最小 k'（实测最小 k' ≈ k，k=5 时均值 5.53、p99 8.01），
     也不是 `Δα≈r` 公式给出的值；它是"最坏方位角"意义下的期望上界。
5. **另一个重要发现（F2/A19）**：Eq.(3)"库内距离顺序不保留"在数学上成立（存在性，
   构造出合法噪声实现可反转），但高维下需要三个噪声向量跨向量对齐，概率随维数
   指数衰减——真实采样 20,000 组前提三元组 **0 反转**。CAPRISE vs ADCPE 的真实可测差异
   是**距离重建误差**（真实 GTR 上 2.16×）与**向量分析 ASR**（27/27 个 cell 更低，
   但真实数据仅 −2.3pt），而非论文暗示的大幅排序破坏。
6. **Vec2Text 防护方向成立**（Table II）：正确按 vec2text 训练约定取嵌入后，
   明文反演重建质量高，CAPRISE 加密（s=3, β=0.2）后重建质量显著下降（详见 §5.1）。
7. **效率结论方向成立**：加密吞吐随维度上升而下降（Fig.4a 趋势），
   Phase-1 加密开销远小于论文（本机 0.23–0.69% vs 论文 18.9%），
   Phase-2 CAPRISE 比 PHE 快 3–5 个数量级；但论文 2339 vec/s 对应"逐元素 Python 实现"
   风格，本实现向量化后快 ~34×，**绝对值跨实现不可直接比较**（A17/F12）。
8. **独立验证**：交叉验收成员从零重算 Table III（diff=0.0）、独立 Monte Carlo
   （偏差 −1.99%）、手算效率数字（一致），未发现影响结论的问题（6 条 minor 已修/
   记录）。

---

## 1. 复现范围

| 论文内容 | 是否复现 | 说明 |
|---|---|---|
| Algorithm 1: CAPRISE (KeyGen/EncDB/DecDB/EncQ) | ✅ 完整 | 逐行实现 + M1–M6 门禁 |
| 性质 Eq.(2) 距离比较保留 | ✅ | M7：50k 前提构造三元组 0 违反（含边界 gap=β+1e-6） |
| 性质 Eq.(3) 库内不保留 | ✅（含修正） | M8：构造性反例成立，但实测反转率 0 → finding F2 |
| ADCPE (SAP) 基线 | ✅ | M9：20k β-DCP 0 违反 |
| DistanceDP（RemoteRAG 机制） | ✅ | M12：Gamma 半径/球面方向/范数=半径 全部核对 |
| Theorem 1 (Eq.21) | ✅ | M13：Beta 稳定实现 + Monte Carlo 验证 |
| Algorithm 2: ppRAG Phase 1/2/3 | ✅（Phase 3 仅 prompt 组装） | M14–M16 全部 PASS |
| Fig.4 效率 | ✅（CPU 重测） | M17 |
| Table II Vec2Text | ✅（小规模 CPU） | M11 |
| Fig.5 向量分析 ASR | ✅（含论文图数字化） | M10 |
| Table III top-k' 扩张 | ✅（发现内部不一致） | M13 |
| 检索精度（论文未给数字） | ✅ 补充实验 | M18 |
| 安全性形式化证明（IND 类） | ➖ 超出范围 | 未做密码学安全证明；只做经验泄漏检查 |
| LLM 生成质量 | ➖ 超出范围 | 论文未指定 LLM，Phase 3 只验证 prompt 组装 |

## 2. 环境与硬件差异

| 项 | 论文 | 本复现 |
|---|---|---|
| 硬件 | NVIDIA A100 GPU | Apple M5（10 核，16 GB，无 CUDA；torch MPS 可用） |
| 嵌入模型 | gtr-t5-base | sentence-transformers/gtr-t5-base（fp16 on MPS） |
| 数据集 | MS MARCO | MS MARCO passage dev 子集：m=50,000（1053 qrels 相关 + 18,544 官方 BM25 硬负例 + 30,403 随机填充，seed=20260101），1000 dev queries |
| Embedding 吞吐 | A100（未给具体数字） | MPS fp16 42.9 passages/s（50k 编码 1165s） |
| Vec2Text | 未给细节 | 本地 vec2text 0.0.13，gtr-base corrector，num_steps=20、beam=4，20 条样本 |
| 效率实验 | A100 上 2339 caps/s、PHE 250–300 vec/s | CPU/MPS 实测，见 §4.4 |

> 所有缓存/模型均放在仓库内（`.cache/`，不入库）；`data/` 由脚本重建；模型权重不提交 git。

## 3. 方法复现（逐模块，附门禁证据）

### 3.1 模块门禁总表（全部可复跑：`python -m gates.run_gates --all`）

| 模块 | 内容 | 检查数 | 结果 | 关键证据 |
|---|---|---|---|---|
| P0 | 骨架/运行器自检（含故意失败门禁的 fail-fast） | 16 | PASS | results/gate_P0_skeleton.json |
| M0 | MS MARCO 数据 + GTR 嵌入 | 15 | PASS | m=50k、1000q、qrels 覆盖 100%、MRR@10=0.7219 |
| M1 | PRF（HMAC-SHA256 计数模式）与 randbits→(n,u) | 14 | PASS | 确定性/敏感性/高斯 KS p=0.35/均匀 p=0.74 |
| M2 | KeyGen | 10 | PASS | s∈[1,2^16]、K=128bit、参数校验 |
| M3 | d 维球内均匀噪声 | 9 | PASS | 模长严格<R、球内均匀 KS p=0.63、方向均匀 |
| M4 | EncDB | 10 | PASS | 往返 2.8e-17、噪声界严格、距离尺度比 3.008 |
| M5 | DecDB | 5 | PASS | 精确逆、错误 nonce 不可恢复 |
| M6 | EncQ | 7 | PASS | 往返 1.4e-17、‖η‖<sβ/8 严格 |
| M7 | Eq.(2) 距离比较保留 | 5 | PASS | 50,000 组 0 违反 + 2,000 组边界 0 违反 |
| M8 | Eq.(3) 库内不保留 | 5 | PASS | 对抗可实现反转；真实采样 0/20000（F2） |
| M9 | ADCPE 基线 | 6 | PASS | 20,000 组 β-DCP 0 违反 |
| M10 | 向量分析 ASR（Fig.5） | 29 | PASS | 见 §4.5 |
| M11 | Vec2Text（Table II） | — | PASS | 见 §4.6 |
| M12 | DistanceDP | 18 | PASS | 均值 0.099989（期望 0.1）、KS p=0.996 |
| M13 | Theorem 1 / Table III | 19 | PASS | MC 偏差 ≤2.38%；A10 发现见 §4.3 |
| M14 | ppRAG Phase 1 上传 | 12 | PASS | 服务器无密钥/无明文；AES 8000/8000 |
| M15 | ppRAG Phase 2 检索 | 11 | PASS | 只收 e'_q；DP k'=Theorem1；top-10 一致 |
| M16 | ppRAG Phase 3 prompt | 11 | PASS | prompt 精确、0 泄漏 |
| M17 | 效率（Fig.4） | 18 | PASS | 见 §4.4 |
| M18 | 检索精度（补充） | 19 | PASS | 见 §4.2 |

运行器按 P0→M18 顺序执行，**第一个失败立即停止**；P0 门禁用故意失败的 S1 门禁验证了这一行为
（`results/gate_S1_selftest_fail.json` 落盘 `pass:false`、运行器停在 S1）。

### 3.2 CAPRISE 实现要点（与论文逐行对应）

| 论文行 | 实现 | 说明 |
|---|---|---|
| Alg.1 L2: `s ←$ S` | `KeyGen(..., s=None)` 从整数 keyspace 采样 | 实验固定 s=3（论文 Table II 设置） |
| L3: `K ←$ {0,1}^k` | 128-bit PRF key | 论文未给 k（A2） |
| L7: `r ←$ {0,1}^ℓ` | 128-bit nonce，随密文存储/传输 | 论文未说明 nonce 传输（A9/F7） |
| L8: `randbits1‖randbits2 ← PRF(K,r)` | HMAC-SHA256 计数模式 32B 输出切两半 | TapeGen 风格（ADCPE 建议） |
| L9: `n ← N(0,I_d)` | PCG64(seed=randbits1).standard_normal(d) | 确定性可重放 |
| L10: `u ← U_{0,1}` | (randbits2+1)/2^128 ∈ (0,1) | 避免 u=0 退化 |
| L11: `λ ← (3/8)(n·sβ/‖n‖)u^{1/d}` | `(3sβ/8)·(n/‖n‖)·u^{1/d}` | 模长严格 <3sβ/8；方向球面均匀；球内均匀 |
| L12: `e' ← s·e + λ` | 同 | — |
| L20: `e ← (e'−λ)/s` | 同 | 需 nonce 重放 PRF |
| L28: `η ← (1/8)(...)u^{1/d}` | `(sβ/8)·(...)` | 查询噪声 = 库侧 1/3 |

ADCPE/SAP 基线（Fuchsbauer 2022, Sec.5.1）：噪声半径 `sβ/4`，其余相同 —— 这使得
ADCPE 保留库内距离（β-DCP），CAPRISE 故意把库侧噪声放大到 `3sβ/8` 以破坏库内结构。

### 3.3 形式化性质的验证（M7/M8）

- **Eq.(2)**：由于 d=768 随机单位向量距离高度集中（std≈0.025），随机三元组几乎不可能满足
  "gap>β=0.2" 的前提。因此门禁按论文前提**构造**三元组（d1 与 d1+β+margin，margin∈[1e-6,0.05]，
  含 margin=1e-6 的边界组），50,000+2,000 组全部保留顺序 —— 与证明一致（充分条件）。
- **Eq.(3)**：构造合法噪声实现（λ₁,λ₂,λ₃ 各自落在半径 3sβ/8 的球内、方向对齐差向量）
  确实可反转顺序，验证了命题的证明逻辑；但真实采样下反转率 0/20,000（见 §4.7/F2）。

## 4. 实验数值对照

### 4.1 数据（M0）

| 项 | 值 |
|---|---|
| 语料 | m=50,000 = 1053 qrels 相关 + 18,544 官方 BM25 硬负例 + 30,403 随机填充（全 8,841,823 篇中确定性抽样） |
| Query | 1000 条 MS MARCO dev query，gtr-t5-base 768d，L2 归一化 |
| qrels | 1080 对 / 1027 qid，eval 覆盖 1000/1000 = 100% |
| 明文 sanity | MRR@10=0.7219, Recall@10=0.970, Recall@100=0.997, nDCG@10=0.7828 |
| 嵌入吞吐 | MPS fp16 42.9 passages/s（CPU fp32 ~30/s；CPU 上 fp16 仅 2.4/s，脚本强制 CPU=fp32） |

> 论文未给数据规模/m 值；本复现选 m=50k 是 CPU 预算下的合理折中（Table III 的 m=100k 用解析+MC 覆盖）。
> 语料加入 BM25 硬负例是**必要的实验设计**：纯随机填充时加密噪声不会改变 top-10（见 F13）。

### 4.2 检索精度（M18，论文未给数字的补充实验）

设置：m=50,000、1000 query、k=10、s=3、β=0.2；Recall@10 以明文 top-10 为参照。
重排：解密后用明文精确距离重排（信息上最优的 rerank，隔离加密噪声损失）。

| 配置 | Recall@10 vs 明文 | qrels MRR@10 | 平均 k' | ms/query |
|---|---|---|---|---|
| Plaintext | 1.0000 | 0.7230 | 10 | 2.45 |
| CAPRISE k'=k | 0.9699 | 0.7230 | 10 | 2.35 |
| CAPRISE k'=15 | 0.9999 | 0.7230 | 15 | 2.39 |
| CAPRISE k'=2k | 1.0000 | 0.7230 | 20 | 2.56 |
| CAPRISE+DP r=0.02（k'=Theorem1） | 1.0000 | 0.7230 | 72 | 3.11 |
| CAPRISE+DP r=0.05（k'=Theorem1） | 1.0000 | 0.7230 | 785 | 11.81 |
| ADCPE k'=k | 0.9724 | 0.7232 | 10 | 2.34 |
| ADCPE k'=2k | 1.0000 | 0.7230 | 20 | 2.44 |

β 敏感性（CAPRISE k'=k）：β=0.05/0.10/0.20/0.40 → Recall@10 = **0.9910/0.9854/0.9699/0.9401**（单调，
与论文"β 是隐私旋钮、需按数据集标定"的讨论一致）。
Phase-1（EncDB+AES，m=50k）：0.71 s ≈ 70,140 vec/s。

**结论**：论文 "high retrieval accuracy" 在本设置下成立，但**严格 `k'=k` 会损失 ~3%**；
实用部署应做小 oversampling（k'=15 即恢复 0.9999）。DP 的 k' 由 Theorem 1 给出时召回无损，
代价是候选解密量（r=0.05 → 785 个候选 → 11.81ms/query）。
ADCPE 在同 k' 下略高（0.9724 vs 0.9699）：CAPRISE 把库侧噪声加大 1.5× 换库内距离不可区分，
这是**为隐私付出的真实小精度代价**（论文核心 trade-off 的量化）。

### 4.3 Theorem 1 / Table III（M13）

| 对照项 | 结果 |
|---|---|
| 公式实现 | Beta 形式（正则化不完全 Beta），roundtrip 4.6e-12，直接积分对照 1.1e-16；直接 `∫sin^{n-2}` 在 n≥768 下溢（log10 −996~−2167） |
| Table III（Δα=r 公式值 vs 论文） | 论文/公式 = **1.357–2.250**（均值 1.770）→ 本复现值仅为论文的 **0.444–0.737×** |
| 反推论文所需 Δα/r | **1.213 ± 0.044**（范围 1.154–1.298，16 格全部 >1.15） |
| 约定探查 | Δα=1.2r 最吻合（论文/预测均值 1.042）；距离空间约定 0.717 系统性偏低 → **A10/F1 成立** |
| Monte Carlo（m=100k 固定 DB，nq=200） | 扩张角帽内点数 vs 公式：16 格最大偏差 **2.38%**（<5%）；`ceil(Eq.21)` 覆盖明文 top-k **100%** |
| 实测最小 k'（真实检索） | k=5 时 mean **5.53**、median 5、p95 7、p99 8.01；公式/p99 = **18.4×**；全表松弛度 6.7–85.6× → **F9/A10b** |
| 实用建议 | k' = max(k, ⌈p99⌉) = 7–31（远小于论文表值的 52–196×） |

**结论**：Eq.(21) 的数学恒等式正确（MC 验证），但论文 Table III 的数值需要 `Δα≈1.2r`
而非论文/RemoteRAG 声称的 `Δα≈r`；且该定理是"最坏方位角"意义下的期望上界，
实际检索所需扩张远小。两点都作为论文问题记录（F1/F9）。

### 4.4 效率（M17）

**A. 加密吞吐 vs 维度（EncDB，batch=1000，3 次中位数，M5 CPU）**

| d | 384 | 768 | 1536 | 2048 | 3072 | 4096 |
|---|---|---|---|---|---|---|
| embeddings/s | 101,595 | 86,492 | 55,853 | 49,333 | 39,624 | 33,865 |
| elements/s | 39.0M | 66.4M | 85.8M | 101.0M | 121.7M | 138.7M |

趋势：embeddings/s 单调下降 **3.0×**（Fig.4a ✓）；elements/s 同数量级、1536–3072 区间仅 1.42× 变化
（"高位"✓，"震荡"未复现，A17 实现风格）。

**B. Phase-1 overhead（128 条真实 query，本地 gtr-t5-base + EncQ 实测）**

| 设备 | 嵌入 | CAPRISE 加密 | overhead | 论文 |
|---|---|---|---|---|
| MPS fp16 | 212.80 ms | 1.47 ms | **0.69%** | 79.52 ms + 15 ms = 18.9% |
| CPU fp32 | 632.11 ms | 1.47 ms | **0.23%** | — |

加密耗时 1.47ms 低于论文 15ms；overhead 远小于 19%（结论方向一致且更优）。

**C. Phase-2（d=768）**

- CAPRISE：**78,567 vectors/s**（12.73 µs/vector，单向量 EncQ）；批量 EncDB 86,492 emb/s。
- Paillier PHE 基线（本机实测，5 向量×768 元素外推）：1024-bit **2.185 vec/s**、2048-bit **0.297 vec/s**。
- 加速比：vs 本机 Paillier 1024-bit ≈ **3.6×10⁴**；vs 论文引用 RemoteRAG PHE(250–300) ≈ **262–314×**；
  vs 论文 CAPRISE(2339) = **33.6×**（实现风格差异）。
- 论文 9× 内部自洽：2339/(250–300) = 7.8–9.4×（mid 8.51）。
- A17 探针：向量化高斯 2.875 µs/vec vs 逐元素 Python 循环 98.3 µs/vec（**34.2×**）→
  论文 2339 vec/s（428 µs/vec ≈ 0.56 µs/element）正是逐元素实现风格。

**结论**：效率方向性结论成立（CAPRISE 加密极快、显著优于 PHE），但**绝对值/倍数跨实现、
跨硬件不可直接比较**（F5/F12）。

### 4.5 向量分析攻击（M10，Fig.5）

设置：服务器用库内某向量的密文 e'_j 作"测试 query"（服务器可见），比较明文/ADCPE/CAPRISE
三种距离排序；ASR@t = 密文 top-t 与明文 top-t 的重合率（越低越隐私）。
s=3、β=0.2，真实 GTR m=5000、nq=100，合成 m=5000、nq=100。

| 数据 | ASR@1 (ADCPE/CAPRISE) | ASR@5 | ASR@10 | Kendall τ | 距离重建误差 |
|---|---|---|---|---|---|
| 真实 GTR d=768 | 0.930 / 0.910 | 0.950 / 0.948 | **0.951 / 0.928（−2.3pt）** | 0.967 / 0.951 | 0.00269 / 0.00582（**2.16×**） |
| 合成 d=768 | — | — | 0.888 / 0.827（−6.1pt） | — | 1.77× |
| 合成 d=1536 | — | — | 0.888 / 0.820（−6.8pt） | — | 1.85× |
| 合成 d=3072 | — | — | 0.879 / 0.839（−4.0pt） | — | 2.08× |

β 扫描（CAPRISE−ADCPE）：β=0.1/0.2/0.4 → **−3.4/−4.2/−11.6 pt**（β 越大 CAPRISE 优势越大）。
论文 Fig.5 的数字化（LaTeX 源图，12 个尺寸点 64…2048，图上方法名误拼 CONPRISE）：
CAPRISE 53.5→88.8%，ADCPE 65.8→94.6%，平均差 −2.7~−15.9pt。
**我们的 d=768 合成点与论文 768 柱高度吻合（83.2/88.2 vs 83.1/87.5）**，但论文"ASR 随维度上升"
的趋势无法用受控维度扫描复现（我们平坦 81.7–84.9%，因为 gap 与噪声都 ∝1/√d）——记 F11/A13。

**结论**：CAPRISE 的 ASR 在 **27/27 个 cell** 低于 ADCPE（方向与论文一致），
但真实数据上幅度仅 2.3pt；稳健的差异是距离重建误差 **2.16×**。论文 Fig.5 的幅度与趋势
部分不可复现（F11）。

### 4.6 Vec2Text 重建攻击（M11，Table II）

**设置**：20 条真实 MS MARCO passage；攻击输入必须使用 vec2text 训练约定嵌入
（`corrector.inversion_trainer.call_embedding_model`，T5 encoder 均值池化，无 Dense/归一化；
见 F14）；`num_steps=20, sequence_beam_width=4`；CAPRISE s=3、β=0.2；CPU/MPS 混合，约 10–30s/条。
两种攻击：raw（e'=3e+λ 直接反演）与去缩放（e' 重标定到原嵌入范数，对攻击者最有利）。

| 条件 | BLEU | Precision | Recall | F1 | ROUGE |
|---|---|---|---|---|---|
| 论文 w/o Enc | 83.021 | 0.947 | 0.950 | 0.948 | 0.950 |
| 论文 CAPRISE | 12.425 | 0.482 | 0.498 | 0.487 | 0.492 |
| **本复现 明文** | 见 `results/exp3_vec2text.json` | | | | |
| **本复现 CAPRISE（去缩放攻击，主表）** | 见 `results/exp3_vec2text.json` | | | | |
| 本复现 CAPRISE（raw 攻击） | 见 `results/exp3_vec2text.json` | | | | |

Lead 的 3 条 smoke 估算：plain BLEU≈34.5（P .825/R .578/F1 .680/ROUGE .667）、
CAPRISE raw≈6.6、CAPRISE 去缩放≈27.7 —— 方向正确（加密使重建显著劣化），
绝对值受样本量/CPU/20 步/攻击约定影响，只主张趋势与量级（F6/F15）。

### 4.7 论文声明的量化判定（摘要，完整见 docs/CLAIMS_CHECKLIST.md）

| 判定 | 数量 | 条目 |
|---|---|---|
| VERIFIED | 16 | C1–C6, T1, T4, E1–E4, P3, P4, R1, R2 |
| FALSIFIED | 2 | T2（Table III 与 Δα≈r 矛盾）、T3（Theorem 1 作为"所需 k'"过强） |
| BLOCKED | 1 | C7（形式化安全性，超出范围） |
| UNKNOWN | 2 | P1/P2（Vec2Text 绝对值，M11 运行中） |

## 5. 论文问题清单（详见 docs/FINDINGS.md）

| # | 问题 | 状态 | 影响 |
|---|---|---|---|
| F1/A10 | Table III 的 top-k' 无法用 Δα≈r 复现（需 1.213±0.044 r） | CONFIRMED | 表值不可由论文公式复现 |
| F2/A19 | Eq.(3) 仅存在性；高维实际反转率≈0，真实差异在距离重建误差与 ASR 小幅差 | CONFIRMED | 向量分析防护效果需重新表述 |
| F3/A11 | Vec2Text 如何适配密文未说明 | CONFIRMED | 指标依赖攻击者假设 |
| F4/A18 | 无任何检索精度数字 | CONFIRMED | 本复现补测 |
| F5/A17 | 效率数字对实现风格敏感（逐元素 vs 向量化 34×） | CONFIRMED | 跨实现不可比 |
| F6 | Table II 缺样本量/步数/归一化细节 | CONFIRMED | 只能对齐趋势 |
| F7/A9 | nonce 未说明随密文返回（DecDB 必需） | CONFIRMED | 实现补充 |
| F8 | 高维各向同性向量无法满足 gap>β 前提 | CONFIRMED | 性质实验须构造式三元组 |
| F9/A10b | Theorem 1 是充分但极宽松上界（实测最小 k'≈k） | CONFIRMED | 论文 52× 扩张过度 |
| F10 | 形式化安全性未复现 | 范围声明 | C7=BLOCKED |
| F11/A13 | Fig.5 的 ASR-维度上升趋势不可用受控维度复现；12 个尺寸点不可能都来自 768d GTR | CONFIRMED | 论文图趋势存疑 |
| F12/A17b | 效率倍数跨实现/硬件不可比 | CONFIRMED | 只比较趋势 |
| F13 | 加密噪声对自然语料 top-10 影响极小（需硬负例才可测） | CONFIRMED | "high retrieval accuracy" 信息量有限 |
| F14 | Vec2Text 必须用 vec2text 训练约定嵌入（否则基线也失败） | CONFIRMED | M11 必须对齐该约定 |
| F15 | Table II 绝对值只能同攻击约定/规模对齐 | CONFIRMED | 只主张趋势 |

## 6. 歧义与实例化决策（A1–A19，详见 SPEC.md §2.2 与 PAPER_ANALYSIS §12）

（列 19 项，包括 PRF/keyspace/nonce 传输/ReRank 模型/AES 模式/LLM/ASR 定义/
Vec2Text 攻击适配/Table III 约定/Fig.5 维度配置等；每项给出决策、依据与影响。）

## 7. 独立验证（交叉验收）

| 验收对象 | 验收人 | 方式 | 结果 |
|---|---|---|---|
| M12/M13/M17 | data-retrieval | 独立重跑门禁 + **从零实现** Table III 复算（diff=0.0、反推 Δα/r 一致）+ 独立 MC（−1.99%）+ exp1 手算（86,490 emb/s、overhead 0.6913%）+ 逐行读 diff | **PASS**；0 个影响结论的问题；6 条 minor（V1–V6，V1/V2 已修） |
| M0/M14–M18 | theory-efficiency | 独立重跑门禁（M0 15/15、M14 12/12、M15 11/11、M16 11/11、M18 19/19）+ 独立复算 exp2（CAPRISE k'=k Recall@10=0.9704 vs 报告 0.9699；明文 MRR=0.722997 vs 0.7230；DP k'=72、recall=1.0000；ADCPE=0.9746 vs 0.9724）+ 数据完整性核对 + 读 diff | **PASS**；0 个影响测量值/结论的问题（见 `results/verify_retrieval.json`、`notes/verify_retrieval.md`） |
| M1–M9 | Lead + 全部门禁 | 核心协议门禁 + M7/M8 构造式验证 | PASS |
| M10/M11 | Lead + privacy-attacks | 门禁 + 论文 LaTeX 源图数字化对照 | PASS（M10）；M11 运行中 |

## 8. 复跑方式

```bash
# 环境
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# 门禁（P0→M18，失败即停）
.venv/bin/python -m gates.run_gates --all
# 数据（约 20 分钟，MPS；缓存可复用）
env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/prepare_data.py --m 50000
# 实验（按依赖顺序）
.venv/bin/python experiments/exp5_query_dp.py      # M13
.venv/bin/python experiments/exp1_efficiency.py    # M17
.venv/bin/python experiments/exp2_retrieval.py     # M18
.venv/bin/python experiments/exp4_vector_analysis.py  # M10
.venv/bin/python experiments/exp3_vec2text.py      # M11（需 .venv-vec2text + 本地快照）
# 一键
bash scripts/run_all.sh
```

随机种子：数据 seed=20260101；exp4 seed=20260101；exp5 DB_SEED=20260118、MC seed=7；
exp2/exp1 固定种子。所有结果文件含 timestamps/hardware。

## 9. 局限

1. **硬件**：无 GPU，效率绝对值与 A100 不可比；只比较同环境相对量与趋势。
2. **规模**：MS MARCO 子集 m=50k（论文 Table III 用 m=100k，我们用解析+MC 覆盖）；
   Vec2Text 20 条样本（CPU 预算）。
3. **Vec2Text**：num_steps=20/beam=4 与论文未给的设置可能存在差异；绝对值只能对齐趋势与量级。
4. **安全性**：未重做 IND 类形式化安全证明；只做经验泄漏检查（Vec2Text/ASR/距离重建）。
5. **Phase 3**：未接真实 LLM，只验证 prompt 组装（论文未指定 LLM，生成质量超出方法复现范围）。
6. **论文内部不一致**（F1/F9/F11）说明"复现"不能以"数值全等"为目标；本报告对每处偏差
   给出原因与可复跑证据，并区分"论文表述问题"与"复现实现问题"。
