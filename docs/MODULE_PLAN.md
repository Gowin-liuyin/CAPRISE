# 模块化复现计划与门禁（Module Gates）

> 用户要求：**逐行分析论文 → 逐协议、逐模块复现 → 当前模块门禁不通过，不允许进入下一模块**。
> 论文分析见 `PAPER_ANALYSIS.md`；本文件定义模块顺序、每模块的协议、门禁标准与产物。

## 执行规则

1. 模块按编号顺序执行（M0 → M1 → … → M19）。**上一个模块 gate 文件 `pass=true` 才允许开始下一个**。
2. 每个模块必须产出：
   - 代码（`src/` 或 `experiments/`）
   - 门禁脚本（`gates/gate_<id>_<name>.py`）
   - 结果文件 `results/gate_<id>_<name>.json`（含 `pass`, 每项检查的 expected/actual, 时间戳, 硬件）
   - 若有实验数值：`results/<exp>.json` + `notes/<exp>.md`
3. 门禁失败 → 修代码/修假设 → 重跑门禁，**不允许带病进入下游**。
4. 所有数字真实测得，不得编造；无法满足的验收项要显式标 `"status": "blocked"` 并写原因。
5. 运行器：`python gates/run_gates.py --from M1 --to M9`，遇 fail 即停。

## 模块清单

| 模块 | 协议/内容 | 依赖 | 门禁标准（摘要） | 负责人 |
|---|---|---|---|---|
| **M0** | MS MARCO 数据 + GTR 嵌入 | — | 文件齐全/规模/无 NaN/归一化/覆盖率/qrels sanity | data-retrieval |
| **M1** | PRF 协议：`PRF(K,r)`、randbits→(高斯 n, 均匀 u) | — | 确定性、密钥/消息敏感性、输出长度、n~N(0,1)、u∈(0,1) | Lead |
| **M2** | KeyGen 协议 | M1 | s/K 合法、可复现、非法参数报错、密钥唯一性 | Lead |
| **M3** | 噪声采样协议：d 维球内均匀、模长 < R | M1 | 模长严格 < R 且接近 R；`(‖λ‖/R)^d ~ U(0,1)` KS 检验 p>0.01；方向均匀 | Lead |
| **M4** | EncDB 协议（Alg.1 行 6–14） | M2,M3 | 解密往返误差<1e-9；`‖E'−sE‖<3sβ/8`；batch==single；nonce 唯一 | Lead |
| **M5** | DecDB 协议（Alg.1 行 15–22） | M4 | 确定性、往返、错误 nonce 不得恢复原值 | Lead |
| **M6** | EncQ 协议（Alg.1 行 23–31） | M2,M3 | 往返、`‖η‖<sβ/8`、确定性 | Lead |
| **M7** | Eq.(2) 距离比较保留 | M4,M6 | gap>β 随机三元组 100% 顺序保留（≥5 万组） | Lead |
| **M8** | Eq.(3) 库内不保留 | M4 | 存在反例且反转率显著 >0；ADCPE 对照反转率 =0 | Lead |
| **M9** | ADCPE/SAP 基线协议 | M2,M3 | 往返、`‖λ‖<sβ/4`、β-DCP 成对顺序 100% 保留 | Lead |
| **M10** | 向量分析攻击 ASR（Fig.5） | M4,M9 | ASR@t 与 Kendall τ 输出；CAPRISE < ADCPE（趋势）；数字真实 | privacy |
| **M11** | Vec2Text 重建攻击（Table II） | M4,M0 | 攻击管线可用；基线 BLEU 高、CAPRISE 低；与论文对照 | privacy |
| **M12** | DistanceDP 协议 | — | 半径 Gamma(n,1/ε) 均值≈n/ε；方向球面均匀；‖ẽ−e‖=r | theory |
| **M13** | Theorem 1 / Table III + MC | M12 | 公式与 MC 一致性；给出 Δα=r / 论文 / Δα/r 反推三组数据 | theory |
| **M14** | ppRAG Phase 1 上传（Alg.2 1–9） | M4,M9 | 服务器只有密文+nonce+AES 密文；客户端可解密 round-trip | data-retrieval |
| **M15** | ppRAG Phase 2 检索（Alg.2 10–21） | M6,M12,M13,M14 | k' 由定理给出；解密+重排后 top-k 可复现；只传 e'_q | data-retrieval |
| **M16** | ppRAG Phase 3 prompt | M15 | prompt 组装正确、无明文泄漏 | data-retrieval |
| **M17** | 效率实验（Fig.4 + Phase1/2） | M4,M6,M1 | 吞吐随 d 变化；overhead；CAPRISE vs PHE 对照 | theory |
| **M18** | 检索精度实验（补充） | M15,M0 | Plaintext/CAPRISE/CAPRISE+DP/ADCPE 的 Recall/MRR；β 扫描 | data-retrieval |
| **M19** | 复现报告 | 全部 | 论文 vs 复现对照表、偏差分析、歧义清单 | Lead |

## 门禁脚本约定

```python
# gates/common.py 提供
gate = Gate("M1", "prf", owner="lead")
ok1 = gate.check("prf_deterministic", expected=True, actual=bool(...), detail="...")
...
gate.finish()   # 写 results/gate_M1_prf.json，pass = all checks pass
```

`run_gates.py` 按 M0→M19 顺序执行（跳过尚未实现的模块并标记 `not_implemented`），遇到 fail 立即停止并打印原因。

## 当前状态

（每次更新，保持与 results/gate_*.json 一致）

| 模块 | 状态 |
|---|---|
| M1–M9 | 进行中（Lead） |
| M0 | 进行中（data-retrieval） |
| M10, M11 | 等待 M1–M9 通过（privacy） |
| M12, M13 | 可并行启动（theory，独立模块） |
| M14–M18 | 等待前置模块 |
