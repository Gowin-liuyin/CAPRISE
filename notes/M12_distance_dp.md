# M12 — DistanceDP 协议门禁笔记

> 论文: arXiv:2601.12331 Sec. III-B2;机制来自 RemoteRAG (arXiv:2412.12775) Sec. 3.2.1。
> 实现: `src/caprise/dp.py`（Lead 提供，未改动）;门禁: `gates/gate_m12_distance_dp.py`;
> 结果: `results/gate_M12_distance_dp.json`。

## 1. 协议原文与解读

论文只写“采用 DistanceDP、加校准噪声并扩张检索集”。RemoteRAG 给出的 $(n,\varepsilon)$-DistanceDP：

```
r ~ Gamma(shape=n, scale=1/ε),   E[r] = n/ε          # 径向扰动（质量随机）
d ~ Uniform(S^{n-1})                                   # 方向球面均匀
ẽ_q = e_q + r·d                                        # 不重新归一化
```

- $n$ 是嵌入维度（论文记 $n$，本仓库代码变量为 `dim`，两者同一量）。
- 归一化嵌入（GTR 嵌入 L2 归一化）下 RemoteRAG 给典型 $\bar r \in [0.02,0.1]$，
  等价于 $\varepsilon \in [10n, 50n]$。这与 Table III 使用的固定半径 0.02 / 0.033 一致。
- “扰动角 $\Delta\alpha_k$ 可近似为 $r$”：对单位向量 $q$，$q$ 与 $q+r d$ 的夹角
  $\rho=\arccos\!\big((1+r\,q\cdot d)/\|q+r d\|\big)=2\arcsin(r/2)\approx r$。
  这是 M13 的理论出发点（论文 Table III 的 $\Delta\alpha$ 约定问题见 A10）。

实现有两种模式（`src/caprise/dp.py`）：
1. `distance_dp(e, radius="gamma", eps=ε)`：原始 DP 机制（随机半径）；
2. `distance_dp(e, radius=r)`：固定半径，供 Table III / 消融实验使用。

## 2. 门禁检查与结果（全部 PASS）

| 检查 | 标准 | 实测 |
|---|---|---|
| Gamma 均值 = n/ε | n=768, ε=7680, 5000 次，相对误差 <2% | 0.099989 vs 0.1（rel err 1.1e-4）|
| Gamma 拟合 KS 检验 | p > 0.01 | D=0.00580, p=0.9957 |
| std/mean = 1/√n | 相对误差 <5% | 0.036150 vs 0.036084（0.2%）|
| 方向单位范数 | 最大偏差 ≤1e-12 | 2.2e-16（5000 次）|
| 标准化坐标均值 z 检验 | max\|z_j\|<5（768 个坐标）| 2.925 |
| \|z\|<3 比例 | >0.99（高斯参考 0.9973）| 0.99733 |
| E[z²]=1（坐标方差=1/n）| ±0.02 | 1.00000（精确恒等）|
| ‖ẽ−e‖ = 实现半径 r | ≤1e-12（fixed 200 次 / gamma 200 次）| 6.9e-18 / 2.1e-17 |
| 不重新归一化 | ‖ẽ‖ = ‖e+rd‖ | 一致（1.003925350543）|
| 固定半径模式 | 返回 r 精确、seed 可复现 | 20/20 精确，seed=99 两次逐位相同 |
| gamma 模式 | seed 可复现、换 seed 变化 | 一致/变化 |
| 非法参数 | 缺 eps、r≤0、未知模式、eps≤0 均报错 | 5/5 拒绝 |
| ε↔r̄ 互转 | `epsilon_for_radius(n,r̄)==n/r̄` | 768/0.1=7680 ✓；r̄=0.1↔10n、r̄=0.02↔50n ✓ |
| ε=50n 时均值 | ≈0.02（相对误差<2%）| 0.020020（0.1%）|

运行：`.venv/bin/python -m gates.gate_m12_distance_dp`（或 `python -m gates.run_gates --from M12 --to M12`）。

## 3. 复现约定 / 偏差说明

- 采样用 numpy `Generator.gamma(shape=n, scale=1/ε)`，与 RemoteRAG 定义逐字一致；
  方向用标准高斯归一化（球面均匀的等价构造，M3 已单独验证过同构采样）。
- 扰动后查询**不**归一化：表 III 与 Eq.21 都以径向半径 $r$ 为自变量，若归一化会改变
  $r$ 的含义（且 RemoteRAG 原文也不归一化）。
- 本门禁不重复验证 DP 的 ε-预算形式化证明（论文也未给完整证明，只引用 RemoteRAG），
  只验证机制实现与半径/方向的统计性质；这一点在报告“范围声明”里如实说明。
