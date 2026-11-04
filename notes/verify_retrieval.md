# 交叉验收：data-retrieval 的 M0 / M14–M16 / M18（task-5）

> **历史说明（2026-11-04）**：本文档记录当时 m=50,000 语料轮的独立验收；语料已重建为 m=100,000
> 并重跑 exp2/exp4/gates（见 `results/exp2_retrieval.json`、`results/exp4_vector_analysis.json` 与报告 §4.2/§4.5）。
> 本文档的数值保留为那一轮的验收记录。

> 验收人: theory-efficiency ｜ 验收对象: data-retrieval（M0 数据管线、M14–M16 ppRAG 三阶段、M18 检索精度 exp2）
> 结论: **PASS** —— 5 个门禁独立重跑全部 PASS；核心数字独立复算一致；**0 个影响测量值/结论的问题**；
> 7 条 minor（门禁强度/notes 精度/计时可复现性）建议，不改变判定。
> 机器产物: `results/verify_retrieval.json`（机器可读明细）；本文件为人读说明。
> 约束遵守: 未 import/修改被验收方脚本与文件；独立复算只用 `src/caprise` 核心原语 + `data/*`；
> 本文件与 JSON 是本次验收仅有的两个写文件（重跑门禁会按运行器设计重写对应 `results/gate_*.json`）。

---

## 0. 复核命令与实际输出摘要

```bash
cd "code/CAPRISE复现"
MPLCONFIGDIR=$PWD/.cache/mpl .venv/bin/python -m gates.run_gates --from M0 --to M0
MPLCONFIGDIR=$PWD/.cache/mpl .venv/bin/python -m gates.run_gates --from M14 --to M16
MPLCONFIGDIR=$PWD/.cache/mpl .venv/bin/python -m gates.run_gates --from M18 --to M18
```

| 模块 | 独立重跑结果 | checks | 关键输出（本次实跑） |
|---|---|---|---|
| **M0** data | **PASS** | 15/15 | m=50000, nq=1000；L2 范数最大偏差 4.868e-4（fp16 所致，见 §5）；qrels 覆盖率 1000/1000；MRR@10=0.7219；Recall@10=0.9700 |
| **M14** phase1 | **PASS** | 12/12 | 真实数据 m=8000（前缀含全部 1053 qrels passage）；server 仅 `['db','doc_ciphertexts']`、无 CryptoKey/明文；8000 个唯一 16B nonce；DecDB 往返 9.47e-9；噪声越界 0 |
| **M15** phase2 | **PASS** | 11/11 | k'=k recall=0.9630（m=8000, 100 q, 每次 nonce 随机）；k'=20 recall=1.0；DP r=0.02 k'=55=ceil(Theorem1(10,8000,768,0.02))；server.search 收到 q'(768)，‖q'‖=3.0035 vs s‖q‖=3.0012 |
| **M16** phase3 | **PASS** | 11/11 | prompt 模板精确、10/10 文档按序、无泄漏；20/20 批量 prompt 正确 |
| **M18** retrieval | **PASS** | 19/19 | 11 个配置齐全；CAPRISE k'=k 0.9699；DP r=0.02 k'=72/recall 1.0；β 单调 0.9910/0.9854/0.9699/0.9401；CSV 11 行；图 142 KB |

`run_gates` 三次运行均输出 `OVERALL: PASS`（M14–M16 一次运行覆盖三模块）。

---

## 1. 独立复算（不使用对方 exp2 脚本，只 import `src/caprise` 核心 + 读 `data/`）

方法：自写 1000 query × m=50000 × k=10 管线：明文 top-10（float64 精确距离）作为参照；
CAPRISE `EncDB` 全库加密 → `EncQ` 查询 → 密文空间 top-k' → `DecDB_one` 解密 → 明文距离重排。
DP 用 `distance_dp(radius=0.02)` 固定半径扰动原始 query 后再 `EncQ`，k' 由 `theorem.k_prime` 给出。

| 复算项 | 我的独立结果 | exp2 JSON 报告 | 差异 / 判定 |
|---|---|---|---|
| 明文基线 qrels MRR@10（1000 qrels 查询） | **0.722997** | 0.7230 | 一致 ✅ |
| 明文基线 qrels Recall@10（二值） | 0.9700 | 0.970 | 一致 ✅ |
| **CAPRISE k'=k：Recall@10 vs 明文 top-10** | **0.9704** | **0.9699** | Δ=+0.0005（nonce 随机噪声 σ≈0.005）→ 一致 ✅ |
| CAPRISE k'=k：qrels MRR@10 | 0.723197 | 0.7230 | 一致 ✅ |
| CAPRISE k'=15：Recall@10 | 0.9998 | 0.9999 | 一致 ✅ |
| CAPRISE k'=20：Recall@10 | 1.0000 | 1.0000 | 一致 ✅ |
| **CAPRISE+DP r=0.02：k'** | theorem k'=71.025370 → **ceil=72**；独立 scipy-beta 复算 71.025370（diff 1.1e-12） | 72（`k_prime_set=[72]`） | **一致 ✅** |
| CAPRISE+DP r=0.02：Recall@10 | 1.0000 | 1.0000 | 一致 ✅ |
| CAPRISE+DP r=0.02：qrels MRR@10 | 0.722997 | 0.7230 | 一致 ✅ |
| CAPRISE+DP r=0.05：k' | ceil(784.820258)=**785** | 785 | 一致 ✅ |
| ADCPE k'=k：Recall@10 | 0.9746 | 0.9724 | Δ=+0.0022（同量级 nonce 噪声；方向一致：ADCPE 略高于 CAPRISE）✅ |

**关于 k'≈72 的口径（纠正验收 brief 的一个假设）**：exp2 的 DP 配置使用**固定半径** r=0.02
（`pprag.phase2_retrieve(dp_radius=0.02)` → `distance_dp(radius=0.02)`，r_used 恒为 0.02），
因此 k' 是**逐 query 相同的 72**（JSON `k_prime_set=[72]`），**不是**"每 query 抽样半径后平均"。
作为对照，我另外算了若每 query 从 `Gamma(shape=768, scale=0.02/768)`（均值 0.02）抽半径再取
`ceil(k_prime)`，期望为 **71.68（median 71）**——即两种约定下"平均 k'"都 ≈72，**结论不受影响**，
但 notes 里应写明是固定半径。

**关于 DP recall=1.0**：这是 k'=72 这个 Theorem-1 充分上界的自然结果（M13/F9 已证明定理上界很宽松），
不是 DP 带来额外精度，报告里已按"保证的扩张集"表述，口径正确。

**延迟**：exp2 报告明文 float32 内核 2.45 ms/query、加密端到端 2.35 ms/query（server search 2.17 ms）。
我在当前机器负载（load≈4.8，其他成员实验并发）下独立测得明文 11.15 ms/query、加密 search 4.97 ms/query（m=20000），
绝对值慢 2–4×，**无法逐位复现**；但同 run 内相对关系（加密 ≤2× 明文、server search 主导）与 exp2 一致。
延迟是负载敏感量，建议 notes 标注（见 §5 问题 V-R6）。

---

## 2. M15 门禁是否真的检查了 `server.search` 的入参？

**结论：是（机制真实，不只是日志），但两条断言比措辞弱。**

- `gates/gate_m15_phase2.py` 用 `server.db.search = _spy_search` 替换服务器搜索方法，
  `_spy_search(q_prime, k)` 内部 `np.array(q_prime, copy=True)` 保存**实际传入的数组参数**与 k，
  再调用原方法——这是入口级插桩，不是日志/事后自述。
- check 06 对**第 1 次调用**做了内容检查：shape=(768,)、不等于明文 q、不等于 s·q、
  且 `|‖q'‖ − s‖q‖| ≤ sβ/8`（本次实跑：‖q'‖=3.0035 vs s‖q‖=3.0012，界 0.075）。
- **我的独立 spy 复核**（自建 PPRAGServer/Client，m=2000）：捕获参数 `(numpy.ndarray shape=(768,), int k'=44)`；
  ‖q'‖=3.0010 vs s‖q‖=3.0012；`allclose(q',q)=False`、`allclose(q',3q)=False`；server 属性仅
  `['db','doc_ciphertexts']`、无 CryptoKey。
- **弱点**：
  1. check 07"300 次调用都是 (d,) 密文查询"实际只断言了 **shape 与调用次数**，未逐次检查内容
     （措辞强于实现）；
  2. check 04 用**客户端返回值** `res.k_prime_requested` 与 Theorem1 比对，spy 记录的**服务器实际收到的 k**
     没有被断言（实现当前一致，但门禁抓不到"客户端报告 72、实际向服务器要别的 k"的坏实现）。
  → 建议见 §5 V-R2。

---

## 3. 数据完整性独立核对（不采信 notes）

| 检查 | 结果 |
|---|---|
| `eval_qids.json` 数量 / 与 `query_ids.npy` 集合一致 / 与 `queries.jsonl` 顺序一致 | 1000 / 是 / 是 ✅ |
| `passages.jsonl` 顺序与 `passage_ids.npy` 对齐 | 是 ✅ |
| 语料来源计数（`corpus_sources.npy`） | qrels 1053 + bm25_hard_neg 18544 + random_fill 30403 = 50000，与 manifest 完全一致 ✅ |
| 前 1053 行是否全为 qrels-relevant（manifest 声称的排序） | 是 ✅ |
| `qrels.tsv` 对数 / 是否全部在语料内 / eval 覆盖率 | 1080 / 1080 / 1000/1000 ✅ |
| 嵌入维度 / 范数偏差 | 768 / passage 4.87e-4、query 4.72e-4（MPS fp16 所致，manifest §3 已说明）✅ |

---

## 4. 静态审查（逐行读脚本，查隐藏折衷/阈值/口径/样本量）

| 检查项 | 结论 |
|---|---|
| 是否用明文距离冒充密文检索 | **否**。server 端 `EncryptedVectorDB.search(E', q')` 在密文空间检索；明文距离只在客户端解密后 rerank（SPEC §3 已声明的"上限式 rerank"）。经验佐证：若服务端用明文检索，k'=k 的 recall 会恒为 1.0，实测 0.9699。 |
| 是否跳过 DP / 手工设定 k' | **否**。DP 配置真实调用 `distance_dp`；k' 由 `pprag` 内 `ceil(theorem.k_prime(k,m,dim,r_used))` 计算（exp2 另用同一定理函数预写 meta 作交叉记录）。 |
| 是否偷偷缩小工作集/样本量 | **否**。exp2 JSON 记录 m=50000、n_queries=1000，`meta.command` 无 `--n-queries/--m` 覆盖；独立复算在同规模上复现。 |
| 指标口径是否与 CLAIMS_CHECKLIST 一致 | **是**。Recall@10 = 与明文 top-10 集合重合率/10；qrels MRR@10 = 首个相关文档排名的倒数（仅 qrels 覆盖的 1000 query）；nDCG@10 二值增益。与 checklist"统计口径说明"一致。 |
| 硬编码/编造指标 | **未见**。exp2 结果由实测逐 query 数组聚合；CSV 11 行与 JSON 一致；notes 数字与 JSON 一致（除 §5 V-R7a 的一处 nonce 相关 0.959/0.963 表述）。 |
| 语料组成 | qrels + BM25 硬负例 + 随机填充（1053/18544/30403）。**这是对 task-1"纯随机填充"的偏离，但 data_manifest.md 明确披露**（含 random_fill_only_control MRR=0.9486），Lead 已同意；作为**已披露偏差**记录（F13），非隐藏折衷。加硬负例会把检索变难，使加密噪声可见（否则 recall 几乎恒 1，实验无区分度）。 |
| 门禁阈值 | 均偏松但当前数据远在窗口内，未见"改阈值掩盖坏数据"（详见 §5 V-R2/V-R3）。 |
| 种子/复现性 | exp2 key 生成有 fixed seed，但每次加密用 `os.urandom` 128-bit nonce（已在 meta 声明），逐次 recall 抖动 ~±0.001–0.005；实测复现差异 ≤0.0022。 |

---

## 5. 发现的问题（均 minor / note；不影响 PASS）

| ID | 严重度 | 位置 | 问题 | 影响 | 建议 |
|---|---|---|---|---|---|
| **V-R1** | note（验收 brief 纠正） | exp2 DP 配置 | brief 假设"每 query 抽样半径后平均 k'"；实际是**固定半径 r=0.02 → k'≡72**（`k_prime_set=[72]`）。若真按 Gamma 抽样，E[ceil k']≈71.68，结论相同。 | 无数字影响 | exp2 notes 明确"fixed-radius; k' 恒 72/query" |
| **V-R2** | minor（门禁强度） | gate_m15 checks 04/06/07 | check 07 只断言 shape/次数而非逐次内容；check 04 用客户端 k' 而非服务器实际收到的 k | 当前实现无假 PASS（我已独立复核）；坏实现可能漏检 | spy 记录上逐次断言范数/非明文性；断言 spy 的 k |
| **V-R3** | minor（阈值） | gate_m18 | rec_k≥0.90（实测 0.9699）、明文 MRR∈[0.2,0.9]（0.723）、DP/2k recall≥0.99、MRR Δ≤0.01 均偏松 | 大余量通过，无假 PASS | 收紧至 0.95 / MRR [0.65,0.80] / 0.999 等 |
| **V-R4** | note（门禁设计） | gate_m18 check 10 | exp2 k' vs "expected" 由 exp2 内同一 theorem 函数计算（自洽检查，非独立验证） | 不影响（Theorem 1 由 M13 独立门禁） | docstring 指回 M13/verify_theory |
| **V-R5** | minor（抽样） | gate_m14 check 04 | "server 无明文文档"只抽查前 200/8000 | AES-GCM 下泄漏概率极低，但非穷尽 | 全量扫描（很便宜） |
| **V-R6** | note（计时） | exp2 latency/upload | 绝对延迟在共享机上不可跨 session 复现（我实测慢 2–4×），仅相对关系稳定 | 报告引用绝对 ms 时需谨慎 | notes 标注 load-sensitive、优先报比值 |
| **V-R7** | note（表述） | exp2 notes / data_manifest | (a) notes 引 m=8000 recall 0.959，当前 M15 gate 实跑 0.9630（nonce 逐次随机）；(b) 语料组成偏离 task-1 spec 但已披露 | 无 | (a) 写成"~0.96（nonce 相关）"；(b) 最终报告保留 F13 披露 |

**明确未发现**：数字虚高、编造、用小样本冒充全量、以明文代密文、跳过 DP、手工设定 k'、
指标口径与 checklist 不一致等实质问题。

---

## 6. 与 exp2/notes 数值逐项互查（摘要）

- exp2 notes 表格：plaintext 1.0/0.7230/0.970；CAPRISE k'=k 0.9699；k'=15 0.9999；k'=2k 1.0；
  DP r=0.02 1.0（k'=72, 3.11 ms）；DP r=0.05 1.0（k'=785, 11.81 ms）；ADCPE k'=k 0.9724、2k 1.0；
  β 扫描 0.9910/0.9854/0.9699/0.9401；Phase-1 0.7129 s / 70,140 vec/s —— 与 JSON 逐项一致 ✅
- qrels MRR 区间：JSON 全 11 配置在 [0.722997, 0.723197]，与 notes "±0.0002" 一致 ✅
- CSV：11 行 = 11 配置 ✅；图 `figures/fig_retrieval.png` 142 KB 非空 ✅

## 7. 复现命令（本次验收）

```bash
cd "code/CAPRISE复现"
MPLCONFIGDIR=$PWD/.cache/mpl .venv/bin/python -m gates.run_gates --from M0 --to M0
MPLCONFIGDIR=$PWD/.cache/mpl .venv/bin/python -m gates.run_gates --from M14 --to M16
MPLCONFIGDIR=$PWD/.cache/mpl .venv/bin/python -m gates.run_gates --from M18 --to M18
# 独立复算（自写代码，只 import src/caprise 核心；见 results/verify_retrieval.json → independent_recompute）
```
