# exp1 — 效率实验（Fig.4 / Phase-1 / Phase-2，M17）

> 论文: arXiv:2601.12331 Sec. V-1 "Efficiency"、Fig.4 (Rate_Processing / Rate_Encryption)、
> Phase-1（128 query: 79.52 ms 嵌入 + 15 ms CAPRISE，overhead <19%）、Phase-2（d=768: CAPRISE
> 2339 vectors/s vs RemoteRAG PHE 250–300 vectors/s，约 9×）。
> 实验: `experiments/exp1_efficiency.py`；结果: `results/exp1_efficiency.json`；
> 图: `figures/fig4_efficiency.png`；门禁: `gates/gate_m17_efficiency.py` → `results/gate_M17_efficiency.json`。
> 硬件: Apple M5（10 核, 16 GB, **无 CUDA**），numpy 2.5.4；论文硬件为 **NVIDIA A100 GPU**。
> 复跑: `.venv/bin/python experiments/exp1_efficiency.py`（约 30 s）。

---

## 1. CAPRISE 加密吞吐 vs 嵌入维度（论文 Fig.4a/4b）

设置：`EncDB`，batch=1000 条 L2 归一化随机嵌入（模拟 GTR 嵌入），预热后重复 3 次取**中位数**；
维度 d ∈ {384, 768, 1536, 2048, 3072, 4096}。CAPRISE 参数 s=3, β=0.2（论文隐私实验设置）。

| d | embeddings/s | elements/s | ms/vector | 相对 d=384 |
|---|---|---|---|---|
| 384 | 101,595.0 | 39.01 M | 0.0098 | 1.00× |
| 768 | 86,491.5 | 66.43 M | 0.0116 | 0.85× |
| 1536 | 55,853.4 | 85.79 M | 0.0179 | 0.55× |
| 2048 | 49,333.4 | 101.03 M | 0.0203 | 0.49× |
| 3072 | 39,624.4 | 121.73 M | 0.0252 | 0.39× |
| 4096 | 33,864.9 | 138.71 M | 0.0295 | 0.33× |

对照论文：
- **Fig.4a 趋势（embeddings/s 随 d 增大而下降）复现**：本实现 101.6k → 33.9k（3.0× 下降），
  单向量开销 9.8 µs → 29.5 µs（主要来自 d 维高斯噪声生成，O(d)）。
- **Fig.4b（element-level 吞吐）不完全同形**：论文描述"在 1536–3072 维高位震荡"，
  本实现 elements/s **随 d 单调上升**（39.0M → 138.7M），因为每向量有固定开销
  （2 次 HMAC-SHA256 PRF + PCG64 种子初始化 ≈ 7.9 µs），d 越大越被摊薄；
  d∈[1536,3072] 区间内 elements/s 只变化 1.42×（85.8M → 121.7M），"高位"部分与论文一致。
- 绝对吞吐比论文高约 **35–37×**（论文 2339 vec/s @768 ≈ 1.8M elem/s；本实现 d=768 为 66.4M elem/s）。
  这是**实现风格差异**（finding A17），不是方案本身差异：
  论文数字对应逐元素 Python 实现（2339 vec/s × 768 = 428 µs/vector ≈ 0.56 µs/element）；
  本实现用一次 PCG64 调用生成全部 d 个高斯坐标、numpy 向量化加噪，
  每向量仅 11.6 µs（≈ 15 ns/element）。§4 的 style probe 量化了这一点。

## 2. Phase-1 overhead（论文：79.52 ms 嵌入 + 15 ms 加密 = 18.9%，<19%）

设置：同一批 **128 条 MS MARCO dev query**（`data/queries.jsonl` 前 128 条）；
- 嵌入：本地 gtr-t5-base（`data/raw/gtr-t5-base`，即 data-retrieval 下载的 gtr-t5-base 权重），
  分别测 MPS(float16) 与 CPU(float32)，batch=128，3 次取中位数；
- 加密：CAPRISE `EncQ` 逐条加密同一 128 条嵌入（3 次中位数）。

| 配置 | 128 query 嵌入 | + CAPRISE 加密 | overhead% |
|---|---|---|---|
| **本机 MPS fp16** | 212.80 ms | **1.47 ms** | **0.69%** |
| **本机 CPU fp32** | 632.11 ms | 1.47 ms | **0.23%** |
| 论文（A100 GPU 嵌入 + 本实现加密，混合参照） | 79.52 ms | 1.47 ms | 1.85% |
| **论文（A100）** | 79.52 ms | 15 ms | **18.9%（<19%）** |

结论：
1. **"加密开销远小于嵌入时间（<19%）"这一结论成立且更极端**：本实现 0.23–0.69%（比论文 18.9%
   低一个数量级以上）。原因有两层：(a) 嵌入本来就是 Phase-1 瓶颈（CPU 上 128 条 0.63 s，
   MPS 0.21 s）；(b) 本实现的加密比论文的逐元素实现快约 10×（1.47 ms vs 15 ms，A17）。
2. 嵌入的**绝对**时间与论文不可直接比（M5 CPU/MPS 无 CUDA，vs A100；模型 dtype 也有差异），
   因此上表同时给出三种参照，报告以"overhead 比例"作为可跨硬件比较的主指标。
3. 结论与论文一致：**CAPRISE 加密不是 Phase-1 的瓶颈，瓶颈是嵌入计算**（finding A17c）。

## 3. Phase-2 吞吐（d=768，论文 2339 vec/s；RemoteRAG PHE 250–300 vec/s）

| 方案 | vectors/s | 说明 |
|---|---|---|
| **CAPRISE（本复现，单向量 EncQ 循环，2000 条）** | **78,567**（12.73 µs/vector） | M5 CPU，numpy 向量化实现 |
| CAPRISE（本复现 EncDB 批量，d=768） | 86,491 | 见 §1，批量 1000 |
| CAPRISE（论文，A100） | 2,339 | 逐元素 Python 风格实现（A17） |
| **Paillier 1024-bit（python-phe 1.5.0 + gmpy2）** | **2.185**（595.9 µs/element, 768 elem/vector ≈ 0.458 s/vector） | 5 个向量外推；M5 CPU |
| Paillier 2048-bit（python-phe + gmpy2） | 0.297（4.39 ms/element ≈ 3.37 s/vector） | 5 个向量外推 |
| Paillier 1024-bit（纯 CPython 内建 pow，无 GMP） | 0.155（8.39 ms/element） | 无 GMP 时的最坏情况下界 |
| Paillier 2048-bit（纯 CPython 内建 pow） | 0.020（63.9 ms/element） | 同上 |
| RemoteRAG PHE（论文引用，A100/其实现） | 250–300 | 论文正文引用的 250–300 vectors/s |

加速比（同一机器上测得的 CAPRISE vs Paillier）：
- **vs 本机实测 Paillier 1024-bit：78,567 / 2.185 ≈ 3.6×10⁴ 倍**；
  2048-bit：≈ 2.6×10⁵ 倍。
- **vs 论文引用的 RemoteRAG PHE 250–300：262–314 倍**（注意跨硬件/跨实现，不可直接当"方案加速比"）。
- vs 论文 CAPRISE 2339 vec/s：本实现快 **33.6 倍**（实现风格差异，A17）。

**重要解读（避免误读为"CAPRISE 比 PHE 快 10⁴ 倍"）**：
1. Paillier 每个元素需要一次模幂（r^n mod n²），而 CAPRISE 每次只需 PRF + 高斯/均匀采样 + 浮点加乘；
   算法层面 CAPRISE 必然快，但 **3–5 个数量级的差距主要来自实现/硬件**：
   本复现的 Paillier 是 Python + GMP 单线程，RemoteRAG 的数字对应其优化实现（可能是 C/C++/GPU、
   更小 key 或不同库），硬件为 A100；跨硬件直接比绝对时间不公平。
2. **论文自身的 9× 主张内部自洽**：2339 / (250–300) = 7.8–9.4×；CAPRISE 与 PHE 都在同一论文
   A100 环境下测的话，9× 量级可信。
3. 本复现能独立确认的定性结论：**在同机同实现风格的公平微基准下，CAPRISE 加密比 1024/2048-bit
   Paillier 快 4–5 个数量级**（因为 Paillier 逐元素模幂的固有成本 + Python 实现开销），
   与论文"CAPRISE 显著优于 PHE"的方向一致；但具体倍数不可跨硬件比较。

## 4. 实现风格探针（finding A17）

| 操作 | 每向量耗时 | 折算 vectors/s |
|---|---|---|
| numpy 一次调用生成 d=768 个标准高斯（本实现方式） | 2.875 µs | 347,800 |
| 逐元素 Python 标量循环生成 768 个高斯（论文风格） | 98.3 µs | 10,170 |

逐元素循环比向量化慢 **34.2×**。论文 2339 vec/s = 428 µs/vector = 0.56 µs/element，
对应"每元素若干次 Python/numpy 标量操作"的实现风格（0.5–0.7 µs/element 量级），
与本实现（向量化，15 ns/element）相差约 35×。**因此论文 Fig.4 的绝对吞吐只在"逐元素 Python 实现"
这一前提下有意义；报告复现值时必须注明实现风格（A17）。**

## 5. 与论文声明对照（供 CLAIMS_CHECKLIST 更新）

| 声明 | 论文 | 本复现 | 判定 |
|---|---|---|---|
| E1 embeddings/s 随 d 增大而下降 | 定性 | 101.6k → 33.9k（d=384→4096，3.0× 下降） | ✅ 趋势复现 |
| E2 element/s 在 1536–3072 维维持高位 | 定性 | 85.8M → 138.7M（区间内 1.42×，处于高位）；但整体随 d 缓慢上升而非震荡 | ⚠️ 部分（实现风格差异） |
| E3 Phase-1 128 query: 嵌入 79.52 ms + 加密 15 ms，overhead <19% | 数值 | 本机 MPS 212.8 ms + 1.47 ms = 0.69%；CPU 632.1 ms + 1.47 ms = 0.23%；论文嵌入混合参照 1.85% | ✅ 结论成立（<19%），绝对时间硬件相关 |
| E4 Phase-2 d=768: CAPRISE 2339 vec/s vs PHE 250–300（9×） | 数值 | CAPRISE 本机 78.6k vec/s；Paillier 1024-bit 2.19 vec/s、2048-bit 0.30 vec/s（实测）；vs 论文 PHE 引用 262–314×；论文 9× 内部自洽 | ⚠️ 方向成立，倍数不可跨实现直接比（A17/A17b） |

## 6. Finding 登记

- **A17**：论文 Fig.4/Phase-2 的绝对吞吐是"逐元素 Python 实现"风格的产物（2339 vec/s ≈ 428 µs/vector
  ≈ 0.56 µs/element）；本 numpy 向量化实现快 ~35×（78.6k vec/s @768）。**倍数是实现风格差异，不是
  密码学方案差异**；报告并列给出、不强行对齐。
- **A17b**：PHE 基线跨硬件/跨实现不可比。本复现实测 Paillier（python-phe+gmpy2，1024-bit: 2.19 vec/s；
  2048-bit: 0.30 vec/s）比 CAPRISE 慢 4–5 个数量级；RemoteRAG 的 250–300 vec/s 是其在 A100 上的
  优化实现。论文 9× = 2339/(250–300) 内部自洽；我们只能确认定性方向（CAPRISE 显著更快）。
- **A17c**：Phase-1 瓶颈是嵌入计算（128 条 query CPU 上 ~0.63 s / MPS ~0.21 s），CAPRISE 加密仅 ~1.47 ms；
  overhead 0.23–0.69%，与论文"<19%"结论一致且更优。

## 7. 限制

1. 无 CUDA：本机嵌入用 Apple MPS(float16) 与 CPU(float32)，不是 A100；Paillier 为 CPU 单线程 Python。
2. Paillier 需 1024/2048-bit key 尺寸差异大；RemoteRAG 未报告其 key 尺寸/实现库/是否并行，
   所以"CAPRISE vs PHE"的倍数只能定性引用论文的 9× 与本文实测的相对量级。
3. 本实验为公平起见在 `.venv` 里安装了 `phe==1.5.0` 与 `gmpy2==2.3.2`（仅用于 PHE 基线；
   `requirements.txt` 未改动，是否固化由 Lead 决定）；纯 Python 无 GMP 路径用内建 `pow` 单独测量并记录。
4. 吞吐测量受系统负载/缓存影响：单次运行内 3 次取中位数，且预热一次；不同运行间差异约 ±5%。

## 8. 产物

- `experiments/exp1_efficiency.py`（约 30 s 可复跑；embedding 测量依赖 `data/raw/gtr-t5-base`，
  缺失时回退到 `results/embed_timing.json` 外推）
- `results/exp1_efficiency.json`（throughput / phase1_overhead / phase2 / implementation_style_probe / findings）
- `figures/fig4_efficiency.png`：(a) embeddings/s vs d（Fig.4a 风格）(b) elements/s vs d（Fig.4b 风格）
  (c) Phase-1 overhead 堆叠图（本机 MPS/CPU vs 论文 A100）(d) Phase-2 vectors/s 对数条形图（CAPRISE vs PHE 各实现）
- `gates/gate_m17_efficiency.py` → `results/gate_M17_efficiency.json`
