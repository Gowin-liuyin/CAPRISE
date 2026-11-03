# M0 数据模块清单（MS MARCO passage dev + gtr-t5-base 嵌入）

> 负责人: data-retrieval ｜ 门禁: `results/gate_M0_data.json` (**PASS**, 15/15 checks)
> 论文: CAPRISE / ppRAG, arXiv:2601.12331 §V Experimental setup（MS MARCO + gtr-t5-base, 768-d）
> 复跑: `env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/prepare_data.py --m 50000 --hard-negs-per-query 20 --n-eval-queries 1000 --batch-size 64`

## 1. 数据来源（全部官方 MS MARCO / 公开 HF）

| 内容 | 来源 | 落地文件 | 大小 |
|---|---|---|---|
| dev 小 qrels（7437 pairs / 6980 qids） | `https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv` | `data/qrels.dev.small.tsv` | 143 KB |
| dev 6980 queries（官方 tar 内） | `.../collectionandqueries.tar.gz`（成员 `queries.dev.small.tsv`） | `data/queries.dev.small.tsv` | 290 KB |
| passage collection（8,841,823 条） | 同上（成员 `collection.tsv`，流式解压读取，不落 3.2GB 中间文件） | `data/raw/collectionandqueries.tar.gz` | 1.06 GB |
| BM25 top-1000 dev 候选（硬负例用） | `.../top1000.dev.tar.gz`（成员 `top1000.dev`） | `data/raw/top1000.dev.tar.gz` | 656 MB |
| gtr-t5-base 权重 | HF `sentence-transformers/gtr-t5-base`（fp16 权重） | `data/raw/gtr-t5-base/` | 229 MB |

下载踩坑记录：直连 Azure CDN 单连接 ~0.4–0.8 MB/s，**16 段 HTTP Range × 8 并发**可达 ~5 MB/s（脚本内 `download_parallel`，断点续传）。`collectionandqueries.tar.gz` 的真实 Content-Length 是 1,057,717,952（早期 HEAD 缓存返回过错误的 1,035,009,698，导致第一次拼装 gzip 校验失败；已按真实大小重下尾段并 `gzip -t` 验证）。HF Hub 需 `env -u ALL_PROXY`（环境里的 `ALL_PROXY=socks5://…` 会让 httpx 报 `socksio not installed`）。

## 2. 语料构造（m = 50,000，顺序固定、可复现）

行顺序 = **① qrels 相关 passage（按 pid 升序）→ ② BM25 硬负例（按 pid 升序）→ ③ 均匀随机填充（按 pid 升序）**，
因此任何长度 ≥ 19,597 的前缀都包含全部相关 passage，方便下游做 m-scaling。

| 组成 | 数量 | 说明 |
|---|---|---|
| qrels-relevant | **1,053** | 前 1000 个 qrels qid 的全部相关 passage（1053 对，全部强制入语料 → eval 覆盖率 100%） |
| BM25 hard negatives | **18,544** | 从官方 `top1000.dev` 取每个 eval query 的候选，按"候选 passage 含 query token 的比例"取 top-20（该文件无 rank 列，只能近似 BM25 排序）；去掉与相关 passage 重复者 |
| random filler | **30,403** | 从全部 8,841,823 条 collection 中做**确定性均匀抽样**（`blake2b(seed:pid)` 保留哈希最小的 k 条），seed=20260101 |

> **与 task-1 原始 spec 的差异（已向 Lead 报告并同意方向）**：task-1 原本要求"其余名额从 BeIR/msmarco 语料随机填充"。纯随机填充时明文检索过于容易（实测 1000 queries MRR@10=0.9486），加密噪声几乎不会改变 top-10，M18（exp2）将失去区分度（与 Lead 合成数据 smoke test 中 CAPRISE k'=k recall≈0.87 的现象不符）。因此加入 18,544 条真实 BM25 硬负例，使检索难度接近 MS MARCO reranking（实测 MRR@10=0.7219）。`results/data_sanity.json` 同时保留 **`random_fill_only_control`**（去掉硬负例、纯随机填充的对照，MRR@10=0.9486），两套数字都可复算；passages.jsonl 每行带 `"source": "qrels" | "bm25_hard_neg" | "random_fill"`，`data/corpus_sources.npy` 与之对齐，下游可自行过滤。
>
> 论文 Table III 用的是 m=100,000。需要 100k 时：`--m 100000`（语料选择 19s + 嵌入 ~20–25 min，脚本幂等缓存）。选 50k 是时间盒权衡（嵌入 1165s）。

qrels 过滤：`data/qrels.tsv` 只保留 pid 在语料内的 pair（1080 pairs / 1027 qids；其中 1000 个是 eval qid，27 个非 eval qid 的相关 passage 恰好被硬负例/随机填充抽中）。**下游评测请用 `data/eval_qids.json`（1000 个）作为 query 集**（`data/queries.jsonl` / `query_emb.npy` 只含这 1000 个，顺序一致）。

## 3. 嵌入

- 模型：`sentence-transformers/gtr-t5-base`（T5-base，输出 768d），L2 归一化，float32 存储。
- 硬件：Apple M5（10 核，16 GB，无 CUDA）；**设备 MPS + fp16**：实测 42.9 passages/s（50k 用时 1164.8 s）。⚠️ 该 HF 权重是 fp16，sentence-transformers 6.2 在 **CPU 上会保持 fp16 → 仅 2.4–5.8 passages/s**；脚本对 CPU 强制 fp32（~30 passages/s），对 MPS 用 fp16。
- 精度核验：MPS fp16 vs CPU fp32 前 64 条 embedding 的 cosine：mean=1.000000，min=0.9998；最终 L2 范数偏差 max 4.9e-4（<1e-3 门禁线）。
- 查询与 passage 用同一模型/同一精度编码。

## 4. 产物清单（`data/`）

| 文件 | 形状/大小 | 说明 |
|---|---|---|
| `passages.jsonl` | 50,000 行 / 20.5 MB | `{"pid", "text", "source"}`，真实 MS MARCO passage 文本（平行于 embedding 行） |
| `passage_emb.npy` | (50000, 768) float32 / 154 MB | 已 L2 归一化 |
| `passage_ids.npy` | (50000,) int64 | 与 embedding 行对齐 |
| `corpus_sources.npy` | (50000,) `<U13` | `qrels` / `bm25_hard_neg` / `random_fill` |
| `queries.jsonl` / `query_emb.npy` / `query_ids.npy` | 1000 / (1000,768) / (1000,) | eval query 集 |
| `qrels.tsv` | 1080 行 | `qid \t 0 \t pid \t 1`，仅语料内 pid |
| `eval_qids.json` | 1000 | 有 qrels 覆盖的 eval qid 列表 |
| `meta.json` | — | 计数/模型/耗时/硬件/来源/命令行/args |
| `data/raw/` | ~2.0 GB | 原始下载与缓存（tar、模型、BM25 负例选择缓存、语料选择缓存） |

## 5. 实测数字（真实测量，可复算）

**明文检索 sanity（1000 queries，精确 top-k，`results/data_sanity.json`）**

| 配置 | MRR@10 | Recall@10 | Recall@100 | nDCG@10 |
|---|---|---|---|---|
| 主语料（含 BM25 硬负例，m=50k） | **0.7219** | 0.9700 | 0.9970 | 0.7828 |
| 对照：纯随机填充（去掉 18,544 硬负例） | 0.9486 | 0.9950 | 0.9980 | 0.9603 |

随机排序基线 MRR@10 ≈ 5.9e-5；GTR-T5-base 在**完整 8.8M** MS MARCO dev 上的公开水平 ≈ 0.39（语料变小、随机填充变多 → 数字更高是正确行为）。

**嵌入耗时（`results/embed_timing.json`，Phase-1 overhead 用）**

| 配置 | batch=128 | batch=64 | 单条 |
|---|---|---|---|
| MPS fp16（本机主用） | **222.36 ms** (1.74 ms/q) | 64.43 ms (1.01 ms/q) | 45.53 ms |
| CPU fp32（同机对照） | 565.33 ms (4.42 ms/q) | 165.34 ms (2.58 ms/q) | — |
| 论文（A100 GPU） | 79.52 ms / 128 条 | — | — |

对比结论：MPS fp16 的 batch=128 延迟是论文 A100 的 ~2.8×（无 GPU 环境下的真实差距）；batch=64 反而比论文 128 条还快（64.43 ms），说明 A100 的优势在吞吐而非小 batch 延迟。

## 6. 复跑/校验

```bash
cd "code/CAPRISE复现"
# 全流程（幂等：已存在则跳过下载/语料选择/嵌入；--force 重做嵌入与 sanity）
env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" \
  .venv/bin/python scripts/prepare_data.py --m 50000 --hard-negs-per-query 20 \
  --n-eval-queries 1000 --batch-size 64 --device auto
# 门禁
.venv/bin/python -m gates.run_gates --from M0 --to M0
```

脚本：`scripts/prepare_data.py`（下载→语料选择→嵌入→sanity→timing→meta，全缓存）；门禁：`gates/gate_m0_data.py`。
