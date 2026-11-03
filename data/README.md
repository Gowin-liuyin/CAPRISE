# data/ —— 数据产物目录（不入库）

本目录内容由 `scripts/prepare_data.py` 生成，**不提交到 git**（体积大、可重建）：

| 文件 | 说明 |
|---|---|
| `passages.jsonl` | 语料段落 `{"pid": int, "text": str}`，与嵌入行序一致 |
| `passage_emb.npy` | `(m, 768)` float32，gtr-t5-base 嵌入，L2 归一化 |
| `passage_ids.npy` | `(m,)` passage id |
| `queries.jsonl` | `{"qid": int, "text": str}` |
| `query_emb.npy` | `(n_q, 768)` float32，L2 归一化 |
| `query_ids.npy` | `(n_q,)` query id |
| `qrels.tsv` | `qid\tpid\trel`，仅保留语料内 pid |
| `meta.json` | 规模、模型、耗时、硬件、随机种子 |

重建命令：

```bash
env -u ALL_PROXY HF_HOME="$PWD/.cache/huggingface" \
  .venv/bin/python scripts/prepare_data.py --m 50000
```

数据来源：MS MARCO passage ranking dev（qrels.dev.small.tsv / queries.dev.small.tsv）
+ BeIR/msmarco corpus（流式随机填充）。语料构造保证纳入的每个 dev query
至少有一个 qrels 相关 passage 在库内。
