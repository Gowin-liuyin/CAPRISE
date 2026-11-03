# exp1 — 效率实验（Fig.4 / Phase-1 / Phase-2，M17）

> 论文: arXiv:2601.12331 Sec. V-1 "Efficiency"、Fig.4 (Rate_Processing / Rate_Encryption)、
> Phase-1（128 query: 79.52 ms 嵌入 + 15 ms CAPRISE，overhead <19%）、Phase-2（d=768: CAPRISE
> 2339 vectors/s vs RemoteRAG PHE 250–300 vectors/s，约 9×）。
> 实验: `experiments/exp1_efficiency.py`；结果: `results/exp1_efficiency.json`；
> 图: `figures/fig4_efficiency.png`；门禁: `gates/gate_m17_efficiency.py` → `results/gate_M17_efficiency.json`。
> 硬件: Apple M5（10 核, 16 GB, **无 CUDA**），numpy 2.5.4；论文硬件为 **NVIDIA A100 GPU**。
> 复跑: `.venv/bin/python experiments/exp1_efficiency.py`（约 30–45 s）。

> **计时口径与负载说明（重要）**：本机是共享工作机，实验运行期间其他成员的实验同时在跑
> （load average ≈ 4.8 / 10 核）。JSON 同时记录 **wall-clock**（`runs_s` / `*_per_s`，生产口径）
> 与 **CPU time**（`runs_cpu_s` / `*_per_s_cpu`，`time.process_time()`）。两者在本机几乎相等
> （ratio≈1.01），说明吞吐波动来自**同机资源竞争/频率**而非进程调度；多次运行间
> d=768 EncDB 吞吐实测在 **~69k–102k emb/s** 间波动（±~20%），但**趋势、overhead 比例、
> 相对加速比等结论不受影响**。下表是 `results/exp1_efficiency.json` 这一次（load ≈4.8）的真实测量。

---

## 1. CAPRISE 加密吞吐 vs 嵌入维度（论文 Fig.4a/4b）

设置：`EncDB`，batch=1000 条 L2 归一化随机嵌入（模拟 GTR 嵌入），预热后重复 3 次取**中位数**；
维度 d ∈ {384, 768, 1536, 2048, 3072, 4096}。CAPRISE 参数 s=3, β=0.2（论文隐私实验设置）。

| d | embeddings/s (wall) | elements/s (wall) | ms/vector | 相对 d=384 | embeddings/s (CPU time) |
|---|---|---|---|---|---|
| 384 | 69,838.1 | 26.82 M | 0.0143 | 1.00× | 70,240 |
| 768 | 68,660.3 | 52.73 M | 0.0146 | 0.98× | 69,147 |
| 1536 | 47,704.8 | 73.27 M | 0.0210 | 0.68× | 48,384 |
| 2048 | 39,789.7 | 81.49 M | 0.0251 | 0.57× | 40,214 |
| 3072 | 37,080.3 | 113.91 M | 0.0270 | 0.53× | 37,309 |
| 4096 | 24,812.8 | 101.63 M | 0.0403 | 0.36× | 25,569 |

对照论文：
- **Fig.4a 趋势（embeddings/s 随 d 增大而下降）复现**：本实现 69.8k → 24.8k（约 2.8× 下降），
  单向量开销 14.3 µs → 40.3 µs（主要来自 d 维高斯噪声生成，O(d)）。
- **Fig.4b（element-level 吞吐）不完全同形**：论文描述"在 1536–3072 维高位震荡"，
  本实现 elements/s 整体随 d 上升（26.8M → 113.9M，max/min≈4.2，同数量级），
  因为每向量有固定开销（2 次 HMAC-SHA256 PRF + PCG64 种子初始化 ≈ 8 µs），d 越大越被摊薄；
  d∈[1536,3072] 区间内 elements/s 只变化 1.55×（73.3M → 113.9M），"高位"部分与论文定性一致。
- 绝对吞吐比论文高 **~29×**（同口径单向量：本实现 Phase-2 68.0k vec/s vs 论文 2339 vec/s，
  见 §3/§4；论文 2339 vec/s ≈ 1.8M elem/s，本实现 d=768 为 52.7M elem/s）。这是**实现风格差异**
  （finding A17），不是方案本身差异。

## 2. Phase-1 overhead（论文：79.52 ms 嵌入 + 15 ms 加密 = 18.9%，<19%）

设置：同一批 **128 条 MS MARCO dev query**（`data/queries.jsonl` 前 128 条）；
- 嵌入：本地 gtr-t5-base（`data/raw/gtr-t5-base`），分别测 MPS(float16) 与 CPU(float32)，
  batch=128，3 次取中位数；
- 加密：CAPRISE `EncQ` 逐条加密同一 128 条嵌入（3 次中位数；wall 2.281 ms / CPU time 2.228 ms）。

| 配置 | 128 query 嵌入 | + CAPRISE 加密 | overhead% |
|---|---|---|---|
| **本机 MPS fp16** | 224.68 ms | **2.281 ms** | **1.015%** |
| **本机 CPU fp32** | 747.82 ms | 2.281 ms | **0.305%** |
| 论文（A100 GPU 嵌入 + 本实现加密，混合参照） | 79.52 ms | 2.281 ms | 2.868% |
| **论文（A100）** | 79.52 ms | 15 ms | **18.86%（<19%）** |

结论：
1. **"加密开销远小于嵌入时间（<19%）"这一结论成立且更极端**：本实现 0.31–1.02%（比论文 18.86%
   低一个数量级以上）。原因有两层：(a) 嵌入本来就是 Phase-1 瓶颈；(b) 本实现的加密比论文的
   逐元素实现快（1.5–2.3 ms vs 15 ms，A17）。
2. 嵌入的**绝对**时间与论文不可直接比（M5 CPU/MPS 无 CUDA，vs A100；模型 dtype 不同），
   因此上表同时给出三种参照，报告以"overhead 比例"作为跨硬件主指标。
3. 结论与论文一致：**CAPRISE 加密不是 Phase-1 的瓶颈，瓶颈是嵌入计算**（finding A17c）。

## 3. Phase-2 吞吐（d=768，论文 2339 vec/s；RemoteRAG PHE 250–300 vec/s）

| 方案 | vectors/s | 说明 |
|---|---|---|
| **CAPRISE（本复现，单向量 EncQ 循环，2000 条）** | **67,996**（14.71 µs/vector；CPU time 68,435） | M5 CPU，numpy 向量化实现，无 batching |
| CAPRISE（本复现 EncDB 批量，d=768） | 68,660（见 §1） | 批量 1000，不同口径 |
| CAPRISE（论文，A100） | 2,339 | 逐元素 Python 风格实现（A17） |
| **Paillier 1024-bit（python-phe 1.5.0 + gmpy2）** | **1.625**（801.1 µs/element, 768 elem/vector ≈ 0.615 s/vector） | 5 个向量外推；M5 CPU |
| Paillier 2048-bit（python-phe + gmpy2） | 0.251（5.18 ms/element ≈ 3.98 s/vector） | 5 个向量外推 |
| Paillier 1024-bit（纯 CPython 内建 pow，无 GMP） | 0.142（9.15 ms/element） | 无 GMP 时下界（合成模数，见 §7.4） |
| Paillier 2048-bit（纯 CPython 内建 pow） | 0.019（67.9 ms/element） | 同上 |
| RemoteRAG PHE（论文引用，A100/其实现） | 250–300 | 论文正文引用的 250–300 vectors/s |

加速比（同一机器上测得的 CAPRISE vs Paillier）：
- **vs 本机实测 Paillier 1024-bit：67,996 / 1.625 ≈ 4.2×10⁴ 倍**；
  2048-bit：≈ 2.7×10⁵ 倍。
- **vs 论文引用的 RemoteRAG PHE 250–300：227–272 倍**（跨硬件/跨实现，不可直接当"方案加速比"）。
- **vs 论文 CAPRISE 2339 vec/s（同口径单向量）：29.1 倍**（实现风格差异，A17；不同负载下
  实测 29–34×）。

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
| numpy 一次调用生成 d=768 个标准高斯（本实现方式） | 2.916 µs/vector | 342,900 |
| 逐元素 Python 标量循环生成 768 个高斯（模拟逐元素风格） | 98.438 µs/vector | 10,159 |

逐元素循环比向量化慢 **33.8×**。注意口径：上面是**每向量**耗时；对应**每元素**标量 draw 实测
仅 **0.128 µs/element**。论文 2339 vec/s = 428 µs/vector = 0.56 µs/element，比裸标量 draw 高约
4×，与"每个元素再做若干 numpy 标量运算/函数调用"的实现风格相符——因此
**"论文是逐元素实现风格"是合理重建（A17），并非对论文代码的测量**。

## 5. 与论文声明对照（供 CLAIMS_CHECKLIST 更新）

| 声明 | 论文 | 本复现 | 判定 |
|---|---|---|---|
| E1 embeddings/s 随 d 增大而下降 | 定性 | 69.8k → 24.8k（d=384→4096，2.8× 下降） | ✅ 趋势复现 |
| E2 element/s 在 1536–3072 维维持高位 | 定性 | 73.3M → 113.9M（区间内 1.55×，处于高位）；整体随 d 上升而非震荡（实现风格差异） | ⚠️ 部分 |
| E3 Phase-1 128 query: 嵌入 79.52 ms + 加密 15 ms，overhead <19% | 数值 | 本机 MPS 224.68 ms + 2.281 ms = 1.02%；CPU 747.82 ms + 2.281 ms = 0.31%；论文嵌入混合参照 2.87% | ✅ 结论成立（<19%），绝对时间硬件相关 |
| E4 Phase-2 d=768: CAPRISE 2339 vec/s vs PHE 250–300（9×） | 数值 | CAPRISE 本机 68.0k vec/s；Paillier 1024-bit 1.63 vec/s、2048-bit 0.25 vec/s（实测）；vs 论文 PHE 引用 227–272×；论文 9× 内部自洽 | ⚠️ 方向成立，倍数不可跨实现直接比（A17/A17b） |

## 6. Finding 登记

- **A17**：论文 Fig.4/Phase-2 的绝对吞吐是"逐元素 Python 实现"风格的产物（2339 vec/s ≈ 428 µs/vector
  ≈ 0.56 µs/element）；本 numpy 向量化实现同口径（单向量）快 **~29–34×**（本 JSON 29.1×；
  跨运行实测 29–34×）。**倍数是实现风格差异，不是密码学方案差异**；报告并列给出、不强行对齐。
  （批量 EncDB 吞吐 ~68.7k 与论文 2339 不是同一口径，故不作为 A17 的对比分子。）
- **A17b**：PHE 基线跨硬件/跨实现不可比。本复现实测 Paillier（python-phe+gmpy2，1024-bit: 1.63 vec/s；
  2048-bit: 0.25 vec/s）比 CAPRISE 慢 4–5 个数量级；RemoteRAG 的 250–300 vec/s 是其在 A100 上的
  优化实现。论文 9× = 2339/(250–300) 内部自洽；我们只能确认定性方向（CAPRISE 显著更快）。
- **A17c**：Phase-1 瓶颈是嵌入计算（128 条 query CPU 上 ~0.75 s / MPS ~0.22 s），CAPRISE 加密仅 ~2.3 ms；
  overhead 0.31–1.02%，与论文"<19%"结论一致且更优。

## 7. 限制、口径与交叉验收修正（V1–V5）

1. 无 CUDA：本机嵌入用 Apple MPS(float16) 与 CPU(float32)，不是 A100；Paillier 为 CPU 单线程 Python。
2. Paillier 需 1024/2048-bit key 尺寸差异大；RemoteRAG 未报告其 key 尺寸/实现库/是否并行，
   所以"CAPRISE vs PHE"的倍数只能定性引用论文的 9× 与本文实测的相对量级。
3. 本实验为公平起见在 `.venv` 里安装了 `phe==1.5.0` 与 `gmpy2==2.3.2`（仅用于 PHE 基线；
   `requirements.txt` 未改动，是否固化由 Lead 决定）。
4. **合成模数声明（V4）**：`phase2.pure_python_baselines` 中的 "pure CPython builtin pow" 用
   **合成 1024/2048-bit 奇数模数** 模拟 Paillier 模幂成本（无素数性要求，仅计时），JSON 字段
   `modulus_type` 已显式声明；主 Paillier 数字用 python-phe 的真实密钥。
5. **计时波动**：共享机器上其他任务并发时，wall-clock 吞吐可下降 ~20–30%（d=768 EncDB 实测
   69k–102k emb/s 区间）；`wall≈CPU time` 说明是同机资源竞争。JSON 同时保存两种口径的原始
   runs，门禁用内一致性 + 负载不变量（d=768/d=1536 比值）做回归，不做绝对吞吐的硬绑定。
6. **交叉验收修正（V1/V2/V3/V5，来自 data-retrieval 的 verify_theory）**：
   - V1：`findings[A17c]` 原先写死旧数字；现改为**从本次 JSON 实测中位数动态生成**（`embed`/`enc`/`p1`）。
   - V2：A17 对比口径改为**同口径单向量**（67.9k vec/s 视负载 vs 论文 2339；本 JSON 29.1×），
     批量 EncDB 数字仅注明为不同口径。
   - V3：style probe 字段改名为 `*_us_per_vector` / `*_us_per_element`，并明确"逐元素风格"是
     合理重建而非测量。
   - V4：合成模数已写入 JSON（`modulus_type`）与本文件 §7.4。
   - V5：M17 门禁的 recheck 由 `[0.33,3]` 改为：①`throughput_internal_consistency`（从 raw runs
     精确复算，rel tol 1e-4=JSON 舍入级）②`independent_recheck_d768`（CPU time，5 轮**交错**测量
     d=768 与 d=1536，用"每轮内 cost(768)/cost(1536)"的中位数做**负载漂移抵消**的比值检验
     <10%（最近实测偏差 3.3%），外加绝对窗口 [0.5,2.0] 与顺序保持 768>1536）。
     绝对窗口未能按建议收紧到 [0.9,1.1]：同机并发负载使绝对吞吐在数分钟内波动 0.8–1.1×
     （实测 69k/76k/56k），[0.9,1.1] 会 flaky；故改用"内一致性严格 + 轮内比值"的组合。

## 8. 产物

- `experiments/exp1_efficiency.py`（约 30–45 s 可复跑；embedding 测量依赖 `data/raw/gtr-t5-base`，
  缺失时回退到 `results/embed_timing.json` 外推）
- `results/exp1_efficiency.json`（throughput（wall+CPU time 原始 runs）/ phase1_overhead / phase2 /
  implementation_style_probe / findings A17,A17b,A17c）
- `figures/fig4_efficiency.png`：(a) embeddings/s vs d（Fig.4a 风格）(b) elements/s vs d（Fig.4b 风格）
  (c) Phase-1 overhead 堆叠图（本机 MPS/CPU vs 论文 A100）(d) Phase-2 vectors/s 对数条形图（CAPRISE vs PHE 各实现）
- `gates/gate_m17_efficiency.py` → `results/gate_M17_efficiency.json`
