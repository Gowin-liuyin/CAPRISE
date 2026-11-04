# 论文问题清单（FINDINGS F1..Fn）

> 记录复现过程中发现的论文**不一致 / 缺失 / 过强声明 / 可实现性问题**。
> 每条必须有可复跑证据；状态：CONFIRMED / PENDING（待证实）/ REJECTED。
> 与 `docs/CLAIMS_CHECKLIST.md` 的判定（VERIFIED/FALSIFIED/BLOCKED/UNKNOWN）对应。

## F1: Table III 的 top-k' 与 `Δα≈r` 约定不一致（A10）

- 类型: 数值不一致（论文内部矛盾；且论文正文未定义 r→Δα_k 的换算）
- 论文位置: Table III、Eq.(21)、§V-4（"we perturb the query embedding with r ... as described in Equation 21"）
- 对照来源: `Δα_k ≈ r` 不是 CAPRISE 本文的陈述，而出自论文引用的 RemoteRAG（arXiv:2412.12775）§3.2.2 原文："Since the perturbation is small (r ≪ 1), the perturbed angle Δα_k between e_k and e_k′ can be approximately by r."。CAPRISE 本文既未复述该近似，也未给出 Δα_k 与 r 的任何换算（A10）。
- 现象: 按 RemoteRAG 的 `Δα_k = r` 代入 Eq.(21)，本复现得到 n=768,r=0.033,k=5 时 k'≈147.3；论文表值为 258（论文/公式=1.75×）。反推每格所需 Δα/r 稳定在 ≈1.15–1.30（均值 1.213±0.044，16 格全部 >1.15）。
- **舍入感知细分（2026-11-04，M13）**：论文打印 r=0.033（3 位小数，真值 ±0.0005）与 r=0.02（2 位小数，真值 ±0.005）。把 Δα=r 的预测按打印精度展开后：**r=0.02 列 8/8 格落在预测区间内**（与 Δα=r 兼容），**r=0.033 列 0/8 格**（论文值是公式的 1.75–2.25×，需要真实 r≈0.039–0.043，超出 3 位小数舍入）⇒ **不一致性只出现在 r=0.033 半张表**。
- 系统枚举（M13，2026-11-03 更新）: 对 Table III 的 16 格逐一计算四种约定——
  (a) `Δα=r`：相对误差 −26.3%…−55.6%（mean −41.9%）；
  (b) 弦长 `Δα=2·asin(r/2)`：与 (a) 差 <0.02%，同样系统性低估；
  (c) 欧氏扩张 `α'=2·asin((2sin(α_k/2)+r)/2)`：反向系统性高估 +12.0%…+71.1%（mean +42.0%）；
  (d) 逐格反解：精确复现论文表值（≤5.6e-14），反解 `Δα/r = 1.1537–1.2977`（mean 1.2129±0.0435）。
  三种**可推导**约定逐格取最优仍至少偏离 12.0% ⇒ 不存在由正文/Eq.(21)/RemoteRAG 推出的 Δα 能对上 Table III。
- 参照实现交叉验证（**N 歧义已解决，2026-11-04**）: 直接核对 RemoteRAG arXiv e-print TeX 源——附录 B.4 明写 "Unless otherwise specified ... N=10^5, k=5, k'=160, and T5"，§5.4 未改写 ⇒ **文档默认 N=10^5**。在该 N 下，`Δα=r` 解出 **r\*=0.03392**（四舍五入正是打印的 0.03），打印区间 r∈[0.025,0.035) 上的预测 [69.77,176.20] 包含 160 ⇒ **被引实现与 Δα=r 在打印精度内自洽**（旧口径 N=10^6、170.34/+6.46% 仅作为"打印点简并"的次口径保留）。同一参照点排除 `Δα=1.213r`（199.20=1.25×）与欧氏扩张（274.67=1.72×）。结论保持不变：被引实现用 Δα=r，而论文 Table III 的 r=0.033 列不是。
- 影响: 论文 Table III 无法由论文自身公式 + 被引用来源的 Δα≈r 约定复现；若按未声明的 Δα≈1.2r，则表值可复现到几个百分点内（论文/预测均值 1.042）。`1.2r` 只是表值隐含的拟合约定，不是"正确取值"。
- 复现证据: `results/exp5_query_dp.json`（`delta_alpha_conventions`、`remoterag_reference`）、`notes/exp5_query_dp.md` §4–5、`results/gate_M13_theorem1.json`
- 状态: CONFIRMED（M13 四约定枚举 + 参照实现交叉验证完成）
- 处置: 报告中并列给出 [论文原值 | Δα=r | 弦长 | 欧氏扩张 | 反解 Δα/r | 实测 MC 最小 k']，不修改论文值、不凑数。

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
- 处置: exp3 notes 记录本复现全部设置，报告中说明只可比趋势与量级。**量化（M11 敏感性 + C2 预算研究）**：同一 5 条 passage 上 20/4→50/10 使明文 BLEU 55.7→77.4（+21.7）、
  CAPRISE raw 7.21→7.16（−0.05）、rescaled 29.7→33.0（+3.3）；在 **20 条** passage 上 20/4→50/10 使明文
  BLEU 67.4→77.8（+10.4），50/10 的 95% CI（64.2–89.2）**覆盖论文 83.0** ⇒ 明文基线对反演预算高度敏感，
  加密后水平不敏感（notes/exp3 §4b、§7）。
- 状态: CONFIRMED

## F3b: CAPRISE 的 Vec2Text 防护部分依赖"攻击者不还原尺度"（A11 扩展，量化）

- 类型: 攻击模型未说明/防护强度依赖假设
- 论文位置: Table II、§IV-1（声称噪声足以破坏 Vec2Text）
- 证据: 同一批 CAPRISE 密文（50 条 passage），raw 攻击（e'=3E+λ 直接反演，对应论文口径）BLEU=**13.696**；
  若攻击者估计并还原尺度（e'·‖E‖/‖e'‖，最强攻击）BLEU=**34.584**（F1 0.741），
  仍比明文低 51% 但远高于论文报告的 12.425。论文未说明攻击者如何适配密文（A11）。
- 影响: 论文 Table II 的 12.425 可复现（我们 13.70，95% CI 覆盖论文值；P/R/F1 同一水平），但"CAPRISE 有效防 Vec2Text"
  的强度依赖攻击者假设；报告同时给出两口径并标注。
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

---

## 追加发现（实验完成后）

## F11: 论文 Fig.5 的 ASR 趋势与受控维度实验不一致（A13 扩展）

- 类型: 内部不一致/不可复现
- 论文位置: Fig.5（ASR vs embedding size）
- 证据: privacy-attacks 从论文 LaTeX 源图（`output/pdf/source/caprise/figures/exp_asr.png`）逐柱数字化 12 个尺寸点（64…2048）：CAPRISE 53.5→88.8%，ADCPE 65.8→94.6%，平均差 −2.7~−15.9pt（图上方法名误拼 CONPRISE）。我们的受控维度扫描（合成高斯 d=768/1536/3072）ASR@10 平坦 81.7–84.9%，因为近邻 gap ∝ 1/√d 与噪声 σ ∝ 1/√d 相互抵消；d=768 合成点与论文 768 点高度吻合（83.2/88.2 vs 83.1/87.5）。
- 影响: 论文 12 个"尺寸"点不可能都来自 768 维 gtr-t5-base（A13）；趋势更可能来自不同数据集/邻域结构而非维度本身。
- **2026-11-04 追加（exp6/M20 协议研究）**：对真实 GTR 嵌入枚举三种"768→d 维"协议并逐点对照论文 Fig.5：
  - **H2 截断不归一化**（低维范数 ~√(d/768)，零填充到 d>768）：CAPRISE ASR@10 **51.7→95.7%**（论文 53.5→88.8%），
    逐维 Spearman ρ=**0.951**、平均偏差 **10.4 pp**（ADCPE 7.3 pp）⇒ **趋势可由"范数尺度效应"解释**；
  - H1 投影+归一化 / H3 截断+归一化：动态范围仅 **12.3 / 14.6 pp**（平坦），与论文 35.3 pp 不符 ⇒ 排除"单位球面维度效应"；
  - 残余：高维段仍高 7–10 pp（d=768：92.9/95.6 vs 83.1/87.5）；β=0.2/0.3/0.4 的趋势 ρ=0.974/1.000/0.993、平均偏差 10.1/8.1/8.2 pp；m=5k→100k 影响 <2 pp。论文未给 Fig.5 的 β/ASR 公式/数据协议，残差作为未解项记录，不拟合。
- 证据: results/exp4_vector_analysis.json、results/exp6_asr_protocol.json、notes/exp4_vector_analysis.md、notes/exp6_asr_protocol.md
- 状态: CONFIRMED（趋势已在"截断协议"下复现；绝对水平残差 ≤ ~10 pp 未解释）

## F12: 效率倍数跨实现/跨硬件不可直接比较（A17b/A17c 扩展）

- 类型: 可复现性/可比性
- 论文位置: §V-1（2339 vec/s、PHE 250–300 vec/s、9×）
- 证据: 本机 M5 CPU：CAPRISE 43.5k vec/s（单向量 EncQ，向量化 numpy）；Paillier 实测 1.15 vec/s(1024-bit)/0.17(2048-bit)；与论文引用 RemoteRAG PHE 的比值为 145–174×，而论文 9× 是"同为优化实现"的比值。A17 探针：同机向量化 vs 逐元素实现 36× → 论文 2339 vec/s（428µs/vec ≈ 0.56µs/element）对应逐元素 Python 风格。
- 处置: 报告只比较同环境相对趋势 + 明确声明实现风格差异；不宣称"我们比论文快几十倍"。
- 状态: CONFIRMED

## F13: 加密噪声对自然语料 top-10 的影响极小（实验设计约束）

- 类型: 声明强度/实验设计
- 证据: data-retrieval 的 exp2 语料必须加入 18,544 条官方 BM25 硬负例才能让加密噪声改变 top-10（纯 qrels+随机填充时，明文与加密 top-10 几乎完全一致）。加入硬负例后 CAPRISE k'=k recall@10 才降到 0.9660（m=100k）；且 β=0.05 时 recall 0.9915。
- 影响: 论文"high retrieval accuracy"声明在良性语料上几乎自动成立（加密噪声是 β 量级、远小于硬负例间的典型距离差时），该声明信息量有限；真实风险在对抗性/困难近邻场景。
- 状态: CONFIRMED

## F14: Vec2Text 攻击必须使用 vec2text 训练的嵌入约定（否则基线也失败）

- 类型: 可实现性/攻击模型细节
- 证据: 直接把 sentence-transformers 的 GTR 嵌入（含 Dense+归一化）喂给 vec2text gtr-base corrector：即使 num_steps=20/beam=4，重建也是完全无关文本（说明"无加密也重建失败"，不能作为基线）。改用 vec2text 自身约定（`corrector.inversion_trainer.call_embedding_model`，T5 encoder 均值池化、无 Dense/无 L2，范数 0.93–1.57）后，**50 条** passage 的明文重建 BLEU=69.875（P .924/R .928/F1 .926/ROUGE .860），20 步 beam=4 在共享机上 13–21 s/text。M11 gate 用 npz 数组对嵌入约定与加密链做了字节级审计。
- 影响: 论文 Table II 的基线 BLEU 83 隐含使用 vec2text 约定嵌入；复现 M11 必须对齐该约定，并在报告中声明（这也是 A11 的一部分）。
- 状态: CONFIRMED

## F15: 论文 Table II 的绝对数值只能在相同攻击约定/数据规模下对齐

- 类型: 可复现性
- 证据: 见 F14 + Table II 缺样本量/步数/攻击者归一化细节（F6）。本复现固定 num_steps=20、sequence_beam_width=4、样本 **50 条**（bootstrap 95% CI），报告 BLEU/P/R/F1/ROUGE 并与论文值并列；CAPRISE 行 13.70 [9.72,17.48] 覆盖论文 12.425，明文行 69.9 [62.0,78.2] 低于论文 83.0（0.84×）。C2 预算研究（20 条同 passage）：50/10 明文 77.75，CI 覆盖论文 83.0——差距可由预算解释。
- 状态: CONFIRMED

## F16: 证明 Eq.(12) 的符号笔误（`−` 应为 `+`；结论不受影响）

- 类型: 排版/符号错误（proof typo）
- 论文位置: §III-B1 证明 Eq.(12)（LaTeX 源 ppRAG_FLLM.tex 第 317 行）
- 现象: 论文写
  `‖(s e_q − s e_2) − (η_eq − λ_e2)‖ = ‖C_Q(e_q) − C_DB(e_2)‖`，
  但按 Eq.(5)(6) 有 `e'_q − e'_2 = (s e_q − s e_2) + (η_eq − λ_e2)`；打印式与密文距离**恒不相等**。
  正确的行应为 `≤ ‖(s e_q − s e_2) + (η_eq − λ_e2)‖ = ‖e'_q − e'_2‖`（反向三角不等式方向不变）。
- 验证: `scripts/check_proof_chain.py`（4000 组满足前提的三元组，逐式复核 Eq.(4)–(20)）：
  - `eq12_as_printed_equals_cipher_dist`：4000/4000 不成立（打印式错误）；
  - `eq12_corrected_plus_equals_cipher_dist`：0/4000 违反（改正后成立）；
  - 其余 Eq.(5)–(11) 与 Eq.(13)–(20) 全部 0 违反；Eq.(2) 结论 0/4000 违反。
- 影响: 证明逻辑本身成立（符号修正后），不改变任何结论；但按打印公式无法机械复核，属论文排版错误。
- 证据: `results/proof_chain_check.json`
- 状态: CONFIRMED
