# 论文问题清单（FINDINGS F1..Fn）

> 记录复现过程中发现的论文**不一致 / 缺失 / 过强声明 / 可实现性问题**。
> 每条必须有可复跑证据；状态：CONFIRMED / PENDING（待证实）/ REJECTED。
> 与 `docs/CLAIMS_CHECKLIST.md` 的判定（VERIFIED/FALSIFIED/BLOCKED/UNKNOWN）对应。

## F1: Table III 的 top-k' 与正文 `Δα≈r` 约定不一致（A10）

- 类型: 数值不一致（论文内部矛盾）
- 论文位置: Table III、Eq.(21)、§III-B2（"the perturbed angle Δα_k ... can be approximately by r"）
- 现象: 用 `Δα_k = r`（RemoteRAG 约定）代入 Eq.(21)，本复现得到 n=768,r=0.033,k=5 时 k'≈147；论文表值为 258（约 1.75×）。反推每格所需 Δα/r 稳定在 ≈1.19–1.30（均值≈1.2）。
- 影响: 论文 Table III 无法由论文自身公式+约定复现；若按 Δα≈1.2r，则表值可复现到几个百分点内。
- 复现证据: `results/exp5_query_dp.json`、`notes/exp5_query_dp.md`（M13）
- 状态: CONFIRMED（初验，M13 最终确认）
- 处置: 报告中并列给出 [论文原值 | Δα=r 公式值 | 反推 Δα/r | 实测 MC 最小 k']，不修改论文值、不凑数。

## F2: Eq.(3) 是"存在性"命题，高维下实际不可达（A19）

- 类型: 过强声明/解释问题
- 论文位置: §III-B1 Eq.(13)–(20) 证明、Fig.5、§IV-1"vector analysis protection"
- 现象: 证明只要求存在满足 δ+δ'<sβ/2 的噪声实现；但噪声方向在 d 维球面均匀，反转需要 λ1,λ2,λ3 同时与各自差向量对齐，概率随 d 指数衰减（≈e^{-Ω(d)}）。M8 门禁：满足前提（gap>β）的 20,000 组三元组中，真实采样下 CAPRISE 与 ADCPE 反转率均 = 0；而构造的合法噪声实现确实能反转（证明逻辑没错）。
- 影响: 「库内距离顺序被打乱」在 d=768 实际几乎不发生；CAPRISE vs ADCPE 的可测差异落在**距离重建误差**（M8: 0.0046 vs 0.0025，≈1.8×）与 ASR 小幅差异（exp4 初值 CAPRISE ASR@10 低 4–6 个百分点）。
- 复现证据: `results/gate_M8_eq3.json`、`results/exp4_vector_analysis.json`（M10）
- 状态: CONFIRMED
- 处置: 报告中如实给出实测 ASR 差值与距离误差，指出论文 Fig.5 的定性结论在方向上成立、幅度可能被高估。

## F3: Vec2Text 攻击如何适配"加密后"嵌入未说明（A11）

- 类型: 可复现性缺失
- 论文位置: Table II、§V-2
- 现象: 加密嵌入 e'=s·e+λ 范数≈s，vec2text 模型按单位范数 GTR 嵌入训练；论文未说明攻击者是否先归一化、是否已知 s。指标对攻击者假设敏感。
- 处置: exp3 同时报告 raw 攻击与归一化攻击（主表用归一化，对攻击者最有利/最保守），在 notes 中声明。
- 状态: CONFIRMED（规格缺失）

## F4: 论文没有任何检索精度数字（A18）

- 类型: 可复现性缺失
- 论文位置: §V（只声称 "high retrieval accuracy"）
- 处置: 本复现补测 exp2（MS MARCO 子集）：Plaintext / CAPRISE(k'=k) / CAPRISE+oversampling / CAPRISE+DP / ADCPE，β 敏感性。声明为补充实验，不能与论文数值对齐。
- 状态: CONFIRMED

## F5: 效率数字对实现风格高度敏感（A17）

- 类型: 可复现性缺失/跨环境不可比
- 论文位置: §V-1、Fig.4（2339 vec/s @768；128 query 嵌入 79.52ms + 15ms；RemoteRAG PHE 250–300 vec/s；9×）
- 现象: 2339 vec/s ≈ 1.8M elements/s，典型 Python 逐元素实现量级；numpy 向量化实现可达数十万 vectors/s。Phase-1 的 79.52ms 对应 A100 + 具体 batch/token 长度。
- 处置: M17 报告本机（M5 CPU）真实数字、与论文数的对照、以及"实现风格差异"说明；不强行对齐绝对值。
- 状态: CONFIRMED

## F6: Table II 缺少实验细节

- 类型: 可复现性缺失
- 现象: 未给样本量、vec2text 步数/beam、攻击者是否归一化、指标实现口径（BLEU 是否 ×100、ROUGE 具体变体）。
- 处置: exp3 notes 记录本复现全部设置，报告中说明只可比趋势与量级。
- 状态: CONFIRMED

## F7: Algorithm 2 未说明 nonce 如何随密文返回（A9）

- 类型: 规范缺口
- 现象: DecDB 需要 (e'_i, r) 对；Alg.2 行 16 只写返回 {(e'_i, x'_i)}。没有 r，客户端无法重建 λ 解密。
- 处置: 实现中 nonce 与密文同存同传（`EncryptedVectorDB.nonces`）；报告中声明。
- 状态: CONFIRMED

## F8: 高维各向同性嵌入无法满足 gap>β 前提（实验设计约束）

- 类型: 可实现性/实验设计问题
- 现象: d=768 随机单位向量两两距离集中在 √2，std≈0.025，而 β=0.2 ⇒ 随机三元组满足"gap>β"的概率≈0。性质实验（Eq.2/3、Table III 相关）必须用**构造式**前提或真实（聚类）嵌入，否则会误判为"性质不成立"。
- 影响: M7/M8 门禁改为构造式三元组（见 gates/gate_m7_eq2.py 文档字符串）；exp2/exp4 用真实 GTR 嵌入时需报告实际距离 gap 分布（β 是否覆盖真实近邻间距）。
- 状态: CONFIRMED

## F9: Theorem 1 的 k' 是"充分但极宽松上界"

- 类型: 声明过强/实用性问题
- 论文位置: Eq.(21)、Table III、§III-B2
- 现象: Eq.(21) 给出的是"扩大的角壳层内的期望点数"（最坏方位角意义下的上界）；实际检索中，DP 扰动对单点距离的改变约为 r/√n（高维投影效应），实测"包含原 top-k 所需的最小 k'" ≈ k（theory-efficiency M13 初测 k=5→5.3, k=20→21），远小于公式值（论文表 k=5,r=0.033 时 258）。
- 影响: 论文 Table III 的 52× 扩张既非最小需求、也不符合 Δα=r 公式值；实际部署可用远小的 k'。需在报告中区分"最坏情况保证"与"实际需要"。
- 复现证据: `results/exp5_query_dp.json`（minimal k' MC 部分）
- 状态: PENDING（M13 完成后 CONFIRMED）

## F10: 安全性的形式化声明无法在本复现中验证

- 类型: 范围限制（非缺陷）
- 论文位置: §IV（继承 ADCPE 的 IND 类安全性、DistanceDP 的 DP 保证）
- 现象: 复现验证的是机制正确性与经验泄漏（Vec2Text/ASR/距离重建），未重做密码学安全证明。
- 处置: 报告中把 C9/C10 的判定标为 BLOCKED（超出范围），不得声称"安全性已验证"。
- 状态: CONFIRMED（范围声明）
