# QBUS2820 Assignment 1 — Weekly Rent Prediction

| File | Purpose |
|---|---|
| `spec/QBUS2820_S2_2026_Assignment1.pdf` | Assignment specification |
| `WeeklyRent_training.csv` | Training data (5000 rows, includes `WeeklyRent`) |
| `WeeklyRent_test_noLabel.csv` | Test covariates (1000 rows) |
| `SID_Assignment1_implementation.ipynb` | Analysis notebook (currently: setup + EDA) |
| `figures/` | Saved EDA figures |
| `CLAUDE.md`, `tutorial_patterns.md`, `ai_use_log.md` | Working rules, tutorial style guide, AI-use record |
| `tutorials/` | Week 1–7 tutorial notebooks and data |

## 运行方式
在本目录下打开 notebook，**Restart Kernel → Run All**，但**不要运行最后一个 cell**（该 cell 需要只有助教才有的 `WeeklyRent_test.csv`）。

依赖：`pandas numpy scikit-learn matplotlib seaborn statsmodels`（seaborn 的 `lowess=True` 需要 statsmodels）。

## 当前进度
- Phase 0（tutorial 风格整理）：完成，见 `tutorial_patterns.md`
- Phase 1（EDA）：notebook 第 2 节，图保存在 `figures/`
- Phase 2–4（建模、最终模型、报告材料）：待进行。预测 CSV 和 marker-only 最后一个 cell 会在 Phase 3 重新生成

## 还需要你完成的 TODO
- [ ] 把所有 `SID` 换成学号（notebook 文件名、预测 CSV 文件名、notebook 标题）
- [ ] 核对课程 lecture/tutorial，删掉课上没讲过的方法
- [ ] 第 2 节写 EDA 结论（每张图一句 → 对建模的启示）
- [ ] 第 4.3 节根据 EDA 改进 `add_features`，尝试压低 CV MSE
- [ ] 第 4.6 节选择要平均的模型；第 6 节设定最终 `final_model` 并写选择理由
- [ ] 提交前 Restart & Run All（最后一个 cell 除外），确认生成 1000 行的预测 CSV
- [ ] 报告（Word/LaTeX，≤15 页，11–12 号字，数值保留 4 位小数，末尾附 AI 使用声明）
