# exp1 — 效率实验（论文 Fig.4 的 15 个嵌入维度 / Phase-1 / Phase-2，M17）

> 论文: *Efficient Privacy-Preserving Retrieval Augmented Generation with Distance-Preserving
> Encryption* (arXiv:2601.12331v1) §V-1 Efficiency、Fig.4（`figures/Rate_Processing.png` /
> `figures/Rate_Encryption.png`）、Phase-1（128 query: 79.52 ms 嵌入 + 15 ms CAPRISE，overhead <19%）、
> Phase-2（d=768: CAPRISE 2339 vectors/s vs RemoteRAG PHE 250–300 vectors/s，约 9×）。
> 实验: `experiments/exp1_efficiency.py`；结果: `results/exp1_efficiency.json`（本节所有数字均取自该 JSON，
> 生成脚本直读，无手抄）；图: `figures/fig4_efficiency.png`；门禁: `gates/gate_m17_efficiency.py` →
> `results/gate_M17_efficiency.json`。
> 硬件: Apple M5（10 核 = 4P+6E, 16 GB, **无 CUDA**），numpy 2.5.4；论文硬件 **NVIDIA A100 GPU**。
> 复跑: `.venv/bin/python experiments/exp1_efficiency.py`（本轮 164 s，含 Phase-1 的 gtr-t5-base 与
> Phase-2 的 python-phe 基线）。

---

## 0. 结论速览

1. **维度网格与论文完全对齐**：Fig.4 x 轴的 15 个嵌入尺寸 64, 128, 256, 312, 384, 512, 768, 896, 1024, 1536, 1792, 2048, 2560, 3072, 3584（+ d=4096 标 `in_paper=false` 的额外点，不参与任何论文网格断言）。
2. **论文图数字化**：Fig.4a/Fig.4b 各 15 点由像素数字化得到（±2%）；两面板互相自洽（elements/s ≈ embeddings/s × d 的中位偏差 0.59%），且 Fig.4a 在 d=768 的数字化值 2306 emb/s 与论文正文 2339 vectors/s 相差 -1.39%。
3. **两种实现风格**：向量化 `EncDB`（本复现 src 实现）+ 「论文实现风格」逐元素 Python 标量探针（与 `caprise._noise` 数值等价，最差相对误差 7.9e-16）。
4. **A17 成立（定量）**：逐元素探针在 15 个维度上落在论文数字化值的 [1.19, 2.39]×（中位 1.36×，elements/s），首末 drop 42.1× vs 论文 44.5×（相差 0.06 个 log 单位 ≪ log2）；同机同轮向量化实现比逐元素探针快 2.2–30.1×（中位 15.1×）——见 §5。
5. **向量化实现不跟随论文曲线**：其 15 点首末 drop 仅 3.14×（论文 44.5×），element 级吞吐随 d 上升 17.8×（论文为高位/震荡）——这正是 A17/A17d 的判据：论文绝对吞吐是**实现风格**的产物。
6. **未复现的局部形状已如实登记**：论文 Fig.4b 在 d=256 的凹陷（相对 d=128 **-18.07%**）本复现没有（-1.99%），记录为 `paper_style_probe.not_reproduced_features`，不强行对齐。
7. **Phase-1/Phase-2 结论不变**：overhead 0.266%/0.188%/1.786% 均 < 19%；Phase-2 CAPRISE 43,468 vectors/s（d=768），Paillier 1024-bit 1.15 vec/s，论文 9× 内部自洽。

## 1. 论文原文（§V-1，逐字）

> “We evaluate the efficiency of the embedding encryption process in CAPRISE. Figure 4 presents the
> encryption throughput across varying embedding sizes. We observe that as the embedding size
> increases, the encryption throughput, measured in embeddings per second, decreases noticeably.
> While the overall embedding-level throughput decreases with increasing embedding size, Figure 4
> shows that the encryption throughput at the element level oscillates but generally remains high
> when embedding size range from 1536 to 3072. This indicates that the per-element encryption
> process becomes more efficient for higher-dimensional embeddings.”
>
> “Specifically, in Phase 1 … generating embeddings for a batch of 128 queries takes approximately
> 79.52 ms, while applying CAPRISE encryption to the same batch introduces only an additional 15 ms.
> This means that the encryption overhead accounts for less than 19% of the embedding time …”
>
> “In Phase 2 … With an embedding size of 768, RemoteRAG reports an encryption speed of
> approximately 250–300 vectors per second. In contrast, CAPRISE achieves a significantly higher
> throughput of 2,339 vectors per second, representing a 9× improvement in efficiency.”

论文**没有**给 Fig.4 的任何数值表、也没有给实现语言/硬件（正文只说所有实验在 A100 上），因此本节把 15 个点
从论文自己的 PNG 里数字化出来，与本复现逐点并列（§3、§4）。

## 2. 测量口径

### 2.1 维度网格 = 论文 Fig.4 x 轴的 15 点

`settings.paper_dims = [64, 128, 256, 312, 384, 512, 768, 896, 1024, 1536, 1792, 2048, 2560, 3072, 3584]`（顺序即图上刻度顺序）；额外点 `settings.extra_dims = [4096]` 在 JSON 每行标 `in_paper=false`，不参与论文网格断言。

### 2.2 两种实现风格

| 风格 | 代码 | batch | 说明 |
|---|---|---|---|
| 向量化 EncDB（本复现 src 实现） | `caprise.caprise.EncDB` | `max(1000, ceil(4e6/d))` 条（≥1000） | 一次 PCG64 调用抽全部 d 个高斯 + numpy 向量化 |
| 「论文实现风格」探针 | `experiments/exp1_efficiency.py: scalar_style_encdb_one` | `max(8, ceil(4e5/d))` 条 | 逐坐标 Python 标量循环：λ_i=(3sβ/8)(n_i/‖n‖)u^(1/d)、e'_i=s·e_i+λ_i |

**探针不是论文代码，也不是 src 核心实现**（JSON: `kind=implementation_style_probe`, `is_paper_code=false`,
`is_src_core_implementation=false`）。它与 src 的噪声采样**数值等价**：同一 (K, r) 下 λ 的最差相对差
 7.9e-16（8 次检查，维度 [64, 312, 768, 3072]），差异只来自范数求和顺序的浮点舍入。
探针的意义：**同样的数学、同样的硬件，只换实现风格**，看能否复现论文 Fig.4 的量级与形状。

### 2.3 计时口径

- 每维 **5 轮交错的完整网格扫描**（旋转起点，使每个维度在所有时间位置均匀出现），取中位数；
  每轮处理的元素数 ≈ 4,000,000（batch=max(1000, ceil(target_elements_per_run / d))）。
- 同时记录 **wall-clock**（`runs_s` / `embeddings_per_s`）与 **CPU time**（`time.process_time()`，
  `runs_cpu_s` / `embeddings_per_s_cpu`）。本机是共享工作机，wall 会被其它 worker 压低；CPU time 更接近
  “本进程拿到的实际算力”，但两者都会随机器状态整体缩放（见 §2.4）。
- nonce：网格测量使用**预生成 nonce**（每个维度一次性抽好，移出计时区），因为论文的 EncDB 由 PRF 派生随机性、
  不是每向量一次 OS 熵采样；`throughput.nonce_mode_probe` 量化了两种口径的差异（见 JSON）。

### 2.4 机器负载校准（关键：绝对吞吐必须连同负载一起读）

本机被多个复现 worker 共享，同一份代码的绝对吞吐在数分钟内可差 3–4×。为此 JSON 里有两类校准：

- `load_calibration`（**固定工作量** d=768、batch=2000、3 次中位）：API-default 口径 **22.63 µs/vector**、预生成 nonce 口径 21.20 µs/vector；与参照轮 14.56 µs/vector（上一版 6 点网格、batch=1000 的同一实现）相比，本轮 **speed factor = 1.554×**（=1 表示与参照轮同速，>1 表示绝对吞吐低于参照轮）。本轮 calibration 时 loadavg = 7.07 / 5.19 / 4.66。
- `throughput.recheck_reference`（固定工作量 d=768、batch=1500、3 次中位）：门禁用同一段工作量做**负载不变量**
  复核——`(rate_d768 / rate_calibration)` 在存档与门禁两侧各自计算后相比，负载因子在比值里约掉。

**参照值 14.564 µs/vector 的来历**（JSON `load_calibration.idle_reference`）：它来自**上一版本实验的存档**
（git HEAD 的 `results/exp1_efficiency.json`：6 点网格 384–4096、batch=1000、3 次中位、EncDB API 默认 nonce，
d=768 行 median 0.014564 s）。**它不是“保证空闲”的基线**：上一版笔记记录当时 1-min loadavg ≈ 4.8。因此
`speed_factor_vs_reference_run` 只作为**机器状态指示器**（1.0 = 与那次参照同速），**没有任何断言依赖它**；
负载不变量的断言改用**同一次运行内**的归一化（§8.3）。

**历史轮次（`previous_runs`，JSON 中保留原始 runs）**：

| 轮次 | speed factor | d=768 向量化 emb/s | d=768 逐元素 emb/s | 逐元素/论文 elements 中位比 | loadavg(1m) |
|---|---|---|---|---|---|
| 2026-11-03T20:00:42+0800 | 1.526× | 27,079 | 3,290 | 1.41× | 4.67 |
| 2026-11-03T20:08:38+0800 | 2.729× | 19,863 | 1,613 | 0.69× | 5.33 |
| 2026-11-03T20:18:32+0800 | 3.13× | 21,225 | 1,522 | 0.65× | 1.35 |
| **2026-11-03T20:46:34+0800（最终采用）** | **1.554×** | **48,225** | **3,187** | **1.36×** | 7.07 |

> 该表的用途：说明“绝对吞吐差 3–4×”来自机器状态而不是实现；判据一律用比值/形状（§8）。

## 3. 论文 Fig.4 的数字化（方法、误差、校验）

论文 Fig.4 的数值只存在于 PNG 里（PDF 文本层没有数字）。本复现直接从论文 LaTeX 源图做**像素数字化**，
代码在 `experiments/exp1_efficiency.py: digitize_paper_fig4()`，门禁会**重新执行同一函数**校验（tol 0.1%）：

1. **坐标框**：找黑像素占比 >60% 的行/列 → 上/下/左/右轴脊（4 个连续像素宽）；
2. **y 轴刻度**：左轴脊左侧 6 列内的短黑刻线 → 刻度像素行；刻度值由图上印刷的轴标签读出
   （Fig.4a: [20.0, 15.0, 10.0, 5.0, 0.0] ×1e+03 embeddings/s；Fig.4b: [2000.0, 1800.0, 1600.0, 1400.0, 1200.0, 1000.0] ×1e+03 elements/s），
   用最小二乘拟合 行→值（Fig.4a 拟合残差 0.032 行、Fig.4b 1.92 行）；
3. **marker**：填充圆点颜色 sRGB (43,86,142)±30 → 连通域质心（每图恰 15 个，面积 ~171 px）；
4. **x 轴**：下轴脊下方 5 行内的短刻线 → 15 个 x 刻度列，与 15 个 marker 一一对应（1:1，最大偏差
   0.0 px）；拟合 `col = a + b·d` 验证 x 轴对 d 是**线性**的
   （Fig.4a 残差 ≤ 0.45 px、Fig.4b ≤ 0.50 px）。

**误差**：marker 质心精度 ~0.1 px；Fig.4a 1 px = 78.99 embeddings/s，
Fig.4b 1 px = 4115 elements/s。对最小读数（d=3584, 490 emb/s）1 px 约 0.9%，加上质心/抗锯齿误差取 **±2%**（JSON `uncertainty_pct=2.0`）。

**三重自校验**（全部写进 JSON）：

- **面板互校**：Fig.4b 的 elements/s 应等于 Fig.4a 的 embeddings/s × d；15 点中位偏差 **0.59%**（<2%，其中 13/15 点 ≤2%，最大 6.11% 出现在 d=3584，属两面板各自的像素量化差异）；
- **论文正文锚点**：Fig.4a 在 d=768 数字化为 2306 emb/s，正文 Phase-2 说 2339 vectors/s → 差 -1.39%；Fig.4b 在 d=768 为 1,795,607 elements/s = 2339×768 的 -0.04%；
- **论文自身曲线的非单调**：数字化后 Fig.4a 在 **256→312 上升 +24.39%**（3886.3 → 4834.2），其余 13/14 步下降 → 本复现的单调性容差按此校准为 25%（§8）。

## 4. 15 点逐维对照表（论文数字化 vs 本复现两风格）

机器：本轮 calibration speed factor **1.55×**（相对参照轮；calibration 时 loadavg(1m)=7.07，见 JSON `load_calibration`）。`paper-style/paper` 为同机实测 ÷ 论文数字化值。

| d | 论文 Fig.4a (emb/s) | 论文 Fig.4b (elem/s) | 向量化 (emb/s) | 向量化 (elem/s) | paper-style (emb/s) | paper-style (elem/s) | paper-style/论文 emb | /elem | 向量化/论文 emb |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 21,817 | 1,396,435 | 65,417 | 4.19 M | 29,138 | 1.86 M | 1.34× | 1.33× | 3.00× |
| 128 | 9,416 | 1,207,137 | 64,529 | 8.26 M | 16,871 | 2.16 M | 1.79× | 1.79× | 6.85× |
| 256 | 3,886 | 989,033 | 58,923 | 15.08 M | 9,233 | 2.36 M | 2.38× | 2.39× | 15.16× |
| 312 | 4,834 | 1,515,775 | 58,834 | 18.36 M | 7,657 | 2.39 M | 1.58× | 1.58× | 12.17× |
| 384 | 3,491 | 1,342,938 | 56,180 | 21.57 M | 6,256 | 2.40 M | 1.79× | 1.79× | 16.09× |
| 512 | 2,938 | 1,482,853 | 53,436 | 27.36 M | 4,726 | 2.42 M | 1.61× | 1.63× | 18.19× |
| 768 | 2,306 | 1,795,607 | 48,225 | 37.04 M | 3,187 | 2.45 M | 1.38× | 1.36× | 20.91× |
| 896 | 1,833 | 1,651,576 | 45,356 | 40.64 M | 2,735 | 2.45 M | 1.49× | 1.48× | 24.75× |
| 1024 | 1,596 | 1,663,921 | 35,114 | 35.96 M | 2,395 | 2.45 M | 1.50× | 1.47× | 22.01× |
| 1536 | 1,359 | 2,071,323 | 30,534 | 46.90 M | 1,606 | 2.47 M | 1.18× | 1.19× | 22.47× |
| 1792 | 1,122 | 2,017,826 | 28,222 | 50.57 M | 1,379 | 2.47 M | 1.23× | 1.23× | 25.16× |
| 2048 | 964 | 1,947,868 | 27,717 | 56.76 M | 1,209 | 2.48 M | 1.25× | 1.27× | 28.76× |
| 2560 | 806 | 2,063,093 | 24,984 | 63.96 M | 969 | 2.48 M | 1.20× | 1.20× | 31.01× |
| 3072 | 648 | 2,071,323 | 22,710 | 69.76 M | 808 | 2.48 M | 1.25× | 1.20× | 35.06× |
| 3584 | 490 | 1,869,680 | 20,823 | 74.63 M | 693 | 2.48 M | 1.41× | 1.33× | 42.51× |

图 `figures/fig4_efficiency.png` 六面板：(a) embeddings/s vs d（论文数字化 / 逐元素探针 / 向量化三线）、
(b) elements/s 同理、(c) 两条实现相对论文的比值、(d) Phase-1 overhead、(e) Phase-2 vs PHE、(f) 逐元素探针 vs
论文的 1:1 散点（含 1.5×/3× 带）。

## 5. 偏差量化：A17 的定量证据

### 5.1 逐元素（论文风格）探针复现了论文的**量级**与**形状**

- 每维 elements/s 与论文数字化值之比：**[1.19, 2.39]×**，中位 **1.36×**；
- embeddings/s 之比：[1.18, 2.38]×，中位 1.41×；
- d=768：探针 3,187 emb/s vs 论文 2,306（1.38×），每向量 314 µs vs 论文 ~434 µs；
- d=3584：探针 693 emb/s vs 论文 490（1.41×）；
- **首末 drop**：探针 **42.1×** vs 论文 **44.5×**（|log 比|=0.057 ≪ log2 → 2× 以内）。

### 5.2 向量化实现比论文快 1–2 个量级，且**不**跟随论文曲线

- 同机同轮 **向量化 / 逐元素探针** 的 embeddings/s 比：d=64 2.2×、d=768 15.1×、d=1536 19.0×、d=3072 28.1×（小 d 端比值最小，因为固定开销 PRF+PCG64 初始化在两种风格里都要付）；
- 对论文数字化值：向量化 3.0–42.5×（中位 22.0×）；
- **形状相反**：向量化首末 drop 仅 3.14×（论文 44.5×，|log 比|=2.66 ≫ log2）；其 element 级吞吐随 d **上升** 17.8×（4.19 M → 74.63 M），而论文是“1536–3072 高位/震荡”。

> 机制：向量化实现每向量只付一次固定开销（2 次 HMAC-SHA256 PRF + PCG64 种子初始化 ≈ 数 µs），d 越大越被摊薄，
> 因此 element 级吞吐随 d 上升；逐元素实现每元素都要付 Python 解释器开销，所以 element 级吞吐近似恒定。
> 论文的报告值（d=768 处 2339 vec/s = 428 µs/vector ≈ 0.56 µs/element）正好落在逐元素风格的成本区间。

### 5.3 未复现的局部形状（如实登记，不强行对齐）

- 论文 Fig.4b 在 d=256 有一个局部凹陷（相对 d=128 **-18.07%**），本复现的逐元素探针为 **+9.46%**（`reproduced=false`）。该凹陷更像论文自身实现/机器的计时伪影，
  我们没有义务复刻，但已写入 JSON `paper_style_probe.not_reproduced_features.paper_fig4b_256_local_dip` 并在门禁中断言
  “该事实被记录”。
- 论文 1536–1792 的高平台：探针在该区 2.47 M vs 论文 2.04 M（1.21×），量级一致，逐点震荡不同。

## 6. Phase-1 overhead（论文 79.52 ms + 15 ms < 19%）

128 条真实 MS MARCO dev query（`data/queries.jsonl` 前 128 条），gtr-t5-base（本地权重）分别测 MPS(fp16) / CPU(fp32)，
CAPRISE `EncQ` 逐条加密同一批嵌入（3 次中位）：

| 配置 | 128 query 嵌入 (ms) | + CAPRISE 加密 (ms) | overhead% |
|---|---|---|---|
| mps | 533.05 | 1.420 | **0.266%** |
| cpu | 756.12 | 1.420 | **0.188%** |
| paper (A100 GPU) | 79.52 | 1.420 | **1.786%** |

论文自身：15/79.52 = **18.86%**（<19%）。本复现三种参照全部 <19%，且更极端——因为 (a) 嵌入
是 Phase-1 瓶颈（比加密大 2–3 个数量级），(b) 本实现的加密比论文的逐元素实现快（A17）。绝对嵌入时间与论文不可直接
比（M5 的 MPS/CPU vs A100；且本轮 MPS 与其它 worker 共用 GPU），故以 **overhead 比例**作为跨硬件主指标。

## 7. Phase-2 吞吐（论文 2339 vectors/s，RemoteRAG PHE 250–300）

| 方案 | vectors/s | 说明 |
|---|---|---|
| **CAPRISE（本复现，单向量 EncQ 循环，n=2000）** | **43,468** | M5 CPU，numpy 向量化；23.01 µs/vector |
| CAPRISE（本复现 EncDB 批量，d=768） | 48,225 | 批量 5209，不同口径 |
| CAPRISE（论文，A100） | 2,339 | 逐元素实现风格（A17） |
| Paillier 1024-bit（python-phe phe.paillier + gmpy2 (GMP backend)） | 1.1525 | 1129.75 µs/element，5 向量外推 |
| Paillier 2048-bit（python-phe phe.paillier + gmpy2 (GMP backend)） | 0.1747 | 7454.63 µs/element，5 向量外推 |
| Paillier 1024-bit（pure CPython builtin pow (no GMP)） | 0.051 | 25468.5 µs/element |
| Paillier 2048-bit（pure CPython builtin pow (no GMP)） | 0.007 | 181890.4 µs/element |
| RemoteRAG PHE（论文引用） | 250–300 | 论文正文引用值 |

加速比：vs 本机实测 Paillier 1024-bit **37,716×**、2048-bit 248,813×；vs 论文引用 RemoteRAG PHE 145–174×
（跨硬件/跨实现，只作定性）；论文自身 2339/(250–300) = 7.8–9.4× 内部自洽。**同机公平微基准下 CAPRISE 加密比
Paillier 快 4–5 个数量级**（Paillier 每元素一次模幂），方向与论文一致，绝对倍数不可跨硬件比较（A17b）。

## 8. 判据区间的依据（不是拟合数据）

### 8.1 单调性容差 25%：来自**论文自己的曲线**

数字化后论文 Fig.4a 自身在 256→312 **上升 +24.39%**（3886.3 → 4834.2 emb/s），其余 13/14 步下降。
要求本复现严格单调就比论文本身更严；故判据为：**首末 drop > 2× + Spearman ρ ≤ −0.9 + 相邻上升不超过论文自身的
最大相邻上升（+24.4% → 取 25%）**。

**判据以 CPU-time 速率（`embeddings_per_s_cpu`）为主**：wall-clock 在共享机上会被其它 worker 整体压低（本轮 calibration
speed factor 1.55×），而 CPU time（`time.process_time()`）不受同机抢占影响；相邻比值再取“同一轮内配对”（两维背靠背
测量后取速率比），进一步抵消轮内漂移。wall-clock 的 drop / ρ / 违规只作 informational。本轮实测（CPU-time 主判据）：
drop = **3.14×**，ρ = **-1.000**，相邻比值违规 `round_paired_violations_cpu = []`，最大相邻上升 −0.65%（即无上升，< 25%）；
wall-clock 对照：drop 3.14×、ρ −1.000、配对违规 []。

### 8.2 paper-style 量级区间 [0.4, 3.0]× / 中位 [0.6, 2.6]：来自**同一探针的三轮实测**

| 轮次 | 校准 speed factor | 逐元素/论文 elements 中位比 | loadavg(1m) |
|---|---|---|---|
| 20:00 轮 | 1.53× | 1.43× | 4.67 |
| 20:08 轮 | 2.73× | 0.69× | 4.54 |
| 20:18 轮 | 3.13× | 0.65× | 1.35 |
| **本轮（2026-11-03T20:46:34+0800，最终采用）** | **1.55×** | **1.36×**（elements/s 中位；embeddings/s 1.41×） | 7.07 |

同一份标量探针的比值随机器状态移动约 2×（0.69–1.43）。**“速度因子校正”不可用**：标量 Python 循环对负载的敏感度
远低于 numpy 校准工作量（同一时刻实测 1.7× vs 3.7×），用 speed factor 去乘会高估/低估约 2×。因此该判据只能是
“同一机器上 per-element 实现复现论文量级（半个数量级内）”，区间须同时吸收 (a) 共享机 1.5–3.7× 的速度状态、
(b) 本机 M5 与论文硬件的代际差。它**不是**“与论文数字对齐”，也不用于宣称方案更快/更慢。

### 8.3 负载不变量复核（门禁 `independent_recheck_d768`）

存档轮与门禁轮**不能**直接比绝对吞吐（同一份代码在不同时刻实测 12k–70k emb/s）。门禁的复核因此改成：两侧各跑同一段
固定工作量（d=768、batch=1500、3 次中位）作为分母，比较

```
(rate_d768 / rate_calibration)_gate / (rate_d768 / rate_calibration)_artifact ∈ [0.8, 1.25]
```

机器速度因子在比值里约掉；容差 25% 是因为两侧是不同进程、不同负载状态（仍远严于“随便过”）。

### 8.4 `cost(768)/cost(1536)` 检查的去向（不是悄悄删）

旧门禁的 `independent_recheck_d768` 里还包含“每向量成本比 cost(768)/cost(1536) 偏差 <10%”。该量在共享机上随负载
状态漂移 17–28%（固定开销 PRF/PCG64 与内存受限部分对 cache 争用的反应不同），因此**降级为 informational note**，
其语义由 §8.3 的负载不变量归一化复核**替代**（同一语义：存档轮与门禁轮的相对吞吐一致）；门禁 detail 仍打印该比值。

## 9. Finding 登记

- **A17（主证据）**：论文 Fig.4 的绝对吞吐是**逐元素 Python 实现风格**的产物。逐元素探针在 15 个维度上复现论文量级
  （elements/s 1.19–2.39×，中位 1.36×）与曲线形状（drop 42.1× vs 44.5×）；向量化实现在同机同轮
  比逐元素探针快 2–30×（随 d 增大），且它的曲线是平的（drop 3.14×），不可能是论文 Fig.4 的来源。
  **报告并列给出、不强行对齐**；绝对值受实现风格与机器状态双重影响。
- **A17d**：element 级吞吐的风格依赖性方向相反——逐元素实现的 elements/s 近似恒定（本文 1.33× 内），向量化实现随 d 上升 17.8×。因此“element 级保持高位”这条论文
  陈述的门禁断言放在逐元素探针上，而不是向量化行上。
- **A17b**：PHE 基线跨硬件/跨实现不可比；RemoteRAG 的 250–300 vec/s 是其 A100 上的优化实现。论文 9× 内部自洽。
- **A17c**：Phase-1 瓶颈是嵌入（本机 MPS/CPU 上 128 条 query 0.5–0.8 s 量级，随机器状态变化），CAPRISE 加密仅 1.4 ms；
  overhead 0.19–0.27% ≪ 19%。

## 10. 限制与口径

1. **无 CUDA**：嵌入用 Apple MPS/CPU，Paillier 为 CPU 单线程 Python，与论文 A100 不可直接比绝对时间。
2. **共享机器**：绝对吞吐随负载波动 2–4×；JSON 同时保存 wall/CPU 两套原始 runs、`load_calibration`、`previous_runs`，
   门禁判据一律用轮内配对/比值/形状（负载不变量），绝对速率只报告不断言。
3. **Paillier 合成模数**：`pure_python_baselines` 用合成 1024/2048-bit 奇数模数模拟模幂成本（仅计时），JSON `modulus_type`
   已声明；主 Paillier 数字用 python-phe 的真实密钥。
4. **数字化误差 ±2%**；Fig.4b 在 d=3584 两面板互校偏差 6.1%（见 §3）。
5. **逐元素探针不是论文代码**：它是“同样数学、换实现风格”的重建，用于解释论文数字的来源（A17）；
   与 src 的噪声采样数值等价（1e-16 级）。
6. **未复现的局部形状**：Fig.4b d=256 凹陷（§5.3），已登记在 JSON 与门禁。

## 11. 产物

- `experiments/exp1_efficiency.py`（15 维网格 + 两风格 + 数字化 + Phase-1/Phase-2；复跑约 2–4 分钟）
- `results/exp1_efficiency.json`（`throughput` 含 `5` 轮原始 runs / `monotonicity` / `nonce_mode_probe` /
  `recheck_reference`；`paper_style_probe` 含逐元素 rows / 等价性 / 形状 / 未复现特征；`paper_fig4_digitisation` 含标定与
  两面板统计；`paper_comparison` 15 行并列表；`load_calibration`；`previous_runs`；Phase-1/2；findings A17/A17b/A17c/A17d）
- `figures/fig4_efficiency.png`（6 面板，见 §4）
- `gates/gate_m17_efficiency.py` → `results/gate_M17_efficiency.json`

