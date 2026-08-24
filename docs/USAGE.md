# 使用说明（USAGE.md）

> 本文档说明如何复现本作品的所有结果。

---

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 最低 | 推荐 |
|---|---|---|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 硬盘 | 5 GB | 10 GB |
| GPU | 不需要 | — |

### 1.2 软件依赖

- **Python**: 3.10 ~ 3.13
- **操作系统**: Windows / Linux / macOS

### 1.3 安装

```bash
# 切换到项目根目录
cd E:\智能量化投资策略建模挑战赛

# 安装依赖（推荐用 conda 或 venv 创建环境）
pip install -r requirements.txt
```

`requirements.txt`：
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
scipy>=1.10
torch>=2.0  # 可选，无 GPU 也能跑
matplotlib>=3.5  # 可选，用于画图
```

---

## 2. 数据准备

比赛数据应已解压到：
```
train/train.csv
test/test.csv
submission_template/sample_submission.csv
```

如果未解压，请使用 7-Zip 或 unzip 解压 `train.zip` 和 `test.zip`。

---

## 3. 快速复现（推荐）

### 3.1 一键运行 v4 流水线

```bash
# Windows PowerShell
cd E:\智能量化投资策略建模挑战赛
python code\scripts\run_v4.py --n_stocks 4375 --n_per_stock 5 --num_boost 1500
```

**预期时间**：~ 7-10 分钟（CPU 8 核）

**输出**：
- `submission/output/submission.csv` — 最终提交文件
- `output/pipeline_report.json` — 完整报告
- `output/Xtr_df.csv` — OOF 训练数据
- `output/oof_final.npy` — OOF 最终预测
- `output/test_final.npy` — 测试集最终预测
- `output/models/*.pkl` — 校准器 + meta learner
- `logs/v4_full.log` — 运行日志

### 3.2 提交文件

最终提交：`submission/output/submission.csv`

格式：
```csv
code,up_factor
STOCK_0001,0.452464
STOCK_0002,0.452464
...
```

---

## 4. 分步运行

### 4.1 EDA（探索性数据分析）

```bash
python code\scripts\eda.py
```

输出：`output/eda/eda_summary.json`

### 4.2 流水线各版本

```bash
# v2: 基础版（200 维特征）
python code\scripts\run_pipeline.py --n_stocks 2000

# v3: 扩展版（121 维 + 分类 + 回归）
python code\scripts\run_full.py --n_stocks 4375

# v4: 完整版（+ CatBoost + Ranker）
python code\scripts\run_v4.py --n_stocks 4375
```

### 4.3 本地回测

```bash
# backtest.py 已被 run_v4.py 自动调用
# 也可以独立运行
python code\scripts\backtest.py
```

---

## 5. 参数说明

### 5.1 run_v4.py 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_stocks` | 4375 | 用多少只长期可用股票（按历史长度排序） |
| `--n_per_stock` | 5 | 每只股票取多少个 20+20 窗口 |
| `--min_history` | 60 | 最小历史长度（>= 60 天） |
| `--n_splits` | 5 | OOF 折数 |
| `--num_boost` | 1500 | GBDT 迭代次数 |
| `--seed` | 42 | 随机种子 |
| `--out` | submission_v4.csv | 输出文件名 |

### 5.2 推荐配置

- **快速测试**（5 分钟内）：`--n_stocks 1000 --n_per_stock 4 --num_boost 800`
- **完整训练**（10 分钟）：`--n_stocks 4375 --n_per_stock 5 --num_boost 1500`
- **极限训练**（30+ 分钟）：`--n_stocks 4375 --n_per_stock 10 --num_boost 3000`

---

## 6. 调优建议

### 6.1 如果 AUC 偏低

- 增加 `--n_per_stock`（更多窗口 → 更多样本）
- 增加 `--num_boost`（更深的树）
- 检查特征是否有 NaN/Inf

### 6.2 如果 Top-5 命中率不理想

- 增加 Ranker 权重（修改 `train_v2.py`）
- 调整 group_size
- 用 LightGBM Ranker 直接优化 NDCG@5

### 6.3 如果 Brier Score 偏高

- 改用 Platt Scaling 替代 Isotonic
- 增加 stacking 的多样性
- 检查回归模型的目标分布

### 6.4 如果想用 GPU 加速

- 改用 `cuml` 的随机森林（替代 LGB/XGB）
- 改用 PyTorch Lightning 的 MLP
- 暂时不支持——本方案 CPU 即可

---

## 7. 仓库结构

```
.
├── README.md                    # 顶级说明
├── requirements.txt
├── LICENSE
├── problem.docx                 # 比赛原题（只读）
│
├── train/                       # 训练集
├── test/                        # 测试集
├── submission_template/         # 官方模板
│
├── code/                        # 源代码
│   ├── adaptivepath/            # FCPFF 核心包
│   └── scripts/                 # 入口脚本
│
├── output/                      # 运行结果
├── submission/                  # 提交文件
│
├── docs/                        # 文档
│   ├── TECHNICAL.md             # 技术方案详细
│   ├── USAGE.md                 # 本文档
│   └── FIGURES.md               # 关键结果图
│
├── logs/                        # 运行日志
└── tests/                       # 单元测试
```

---

## 8. 故障排查

### 8.1 LightGBM Ranker group 错误

```
[LightGBM] [Fatal] Sum of query counts (X) differs from the length of #data (Y)
```

**原因**：group 数与样本数不匹配。

**解决**：已在 `run_v4.py` 中通过 `safe_groups()` 处理，确保 group 总和 = 样本数。

### 8.2 内存不足

**解决**：
- 减少 `--n_stocks`
- 关闭其他大程序
- 升级到 16 GB 内存

### 8.3 CatBoost 慢

**解决**：
- 减少 `--num_boost`
- 减少 `--n_per_stock`

### 8.4 提交文件格式错误

**检查**：
- 编码必须是 UTF-8
- 必须有表头 `code,up_factor`
- 必须包含 1500 行（不包含表头）
- up_factor ∈ [0, 1]

---

## 9. 性能基准

| 配置 | 时间 | OOF AUC | OOF Brier | 模拟综合分 |
|---|---|---|---|---|
| 1000 股 × 4 窗 × 1000 boost | ~ 1.5 分钟 | 0.607 | 0.238 | 0.285 |
| 2000 股 × 4 窗 × 1500 boost | ~ 2 分钟 | 0.609 | 0.241 | — |
| 4375 股 × 5 窗 × 1500 boost | ~ 7 分钟 | 0.607 | 0.239 | 0.231 |

---

## 10. 引用

如果本方案对您有启发，请引用：

```bibtex
@misc{fcpff2026,
  title={FCPFF: Four-Stage Cascaded Probability Forecasting Framework for A-share Quantitative Investment},
  author={Chongli Team},
  year={2026},
  url={https://github.com/logicore-code/...}
}
```

---

**版本**：v1.0
**最后更新**：2026-08-24
