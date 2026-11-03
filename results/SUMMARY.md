# 复现结果汇总（自动生成）

## 模块门禁

| 模块 | 门禁 | 结果 | 检查数 | 失败检查 | 证据 |
|---|---|---|---|---|---|
| P0 | skeleton | PASS | 11 | 0 | results/gate_P0_skeleton.json |
| M0 | data | PASS | 15 | 0 | results/gate_M0_data.json |
| M1 | prf | PASS | 14 | 0 | results/gate_M1_prf.json |
| M2 | keygen | PASS | 10 | 0 | results/gate_M2_keygen.json |
| M3 | noise | PASS | 9 | 0 | results/gate_M3_noise.json |
| M4 | encdb | PASS | 10 | 0 | results/gate_M4_encdb.json |
| M5 | decdb | PASS | 5 | 0 | results/gate_M5_decdb.json |
| M6 | encq | PASS | 7 | 0 | results/gate_M6_encq.json |
| M7 | eq2 | PASS | 3 | 0 | results/gate_M7_eq2.json |
| M8 | eq3 | PASS | 5 | 0 | results/gate_M8_eq3.json |
| M9 | adcpe | PASS | 6 | 0 | results/gate_M9_adcpe.json |
| M10 | vector_analysis | PASS | 29 | 0 | results/gate_M10_vector_analysis.json |
| M11 | vec2text | PASS | 71 | 0 | results/gate_M11_vec2text.json |
| M12 | distance_dp | PASS | 18 | 0 | results/gate_M12_distance_dp.json |
| M13 | theorem1 | PASS | 36 | 0 | results/gate_M13_theorem1.json |
| M14 | phase1 | PASS | 12 | 0 | results/gate_M14_phase1.json |
| M15 | phase2 | PASS | 11 | 0 | results/gate_M15_phase2.json |
| M16 | phase3 | PASS | 11 | 0 | results/gate_M16_phase3.json |
| M17 | efficiency | PASS | 36 | 0 | results/gate_M17_efficiency.json |
| M18 | retrieval | PASS | 19 | 0 | results/gate_M18_retrieval.json |

**门禁 PASS: 20/20（已运行模块）**

## 实验产物

| 实验 | 关键字段 | 备注 |
|---|---|---|
| exp1_efficiency | experiment, module, paper_ref, description, hardware, settings, paper_fig4_digitisation, throughput, paper_style_probe, paper_comparison, phase1_overhead, phase2 | results/exp1_efficiency.json |
| exp2_retrieval | meta, plaintext, configs, beta_sensitivity | results/exp2_retrieval.json |
| exp3_vec2text | experiment, paper_ref, definition, setup, passages, embeddings, encryption, modes, bootstrap, sensitivity, paper_table_ii, comparison | results/exp3_vec2text.json |
| exp4_vector_analysis | experiment, paper_ref, definition, setup, selfcheck, main, paper_dims_synthetic, paper_fig5_reference, m_sweep, real_m_sweep, normalization_control, beta_sweep | results/exp4_vector_analysis.json |
| exp5_query_dp | experiment, module, paper_ref, description, settings, paper_table, formula_table_dalpha_eq_r, formula_table_dalpha_eq_1_2r, paper_vs_formula_dalpha_eq_r, reverse_delta_alpha, delta_alpha_conventions, paper_table_rounding | results/exp5_query_dp.json |
