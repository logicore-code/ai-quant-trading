# 智能量化投资策略建模挑战赛 — FCPFF 方案

> 科大讯飞 AI 开发者大赛 · 智能量化投资策略建模赛道
> **方案名称**：FCPFF — Four-Stage Cascaded Probability Forecasting Framework（四阶段级联概率预测框架）
> **核心创新**：截断预训练 + 多目标元学习 + 跨股票模式迁移

---

## 0. 作品速览

| 项目 | 内容 |
|---|---|
| **比赛名称** | 智能量化投资策略建模挑战赛（科大讯飞 AI 开发者大赛） |
| **比赛任务** | 给定 1500 只独立资产过去 20 个交易日的 OHLCV 数据，预测其未来 20 日累计收益率为正的概率（up_factor） |
| **评价指标** | 综合分 = 0.6·累计收益率 + 0.2·(1−最大回撤) + 0.2·(1−Brier Score) |
| **数据集** | 训练集：4375 只 A 股 × 2794 个交易日 × OHLCV；测试集：1500 只独立资产 × 20 个交易日 |
| **提交格式** | `code, up_factor` 两列 CSV，UTF-8 编码，1500 行 |
| **方法** | 四阶段级联概率预测：截断预训练 → 多尺度特征 → 多模型集成 → 概率校准 + Top-K 友好后处理 |
| **本地 OOF 表现** | AUC 0.607 / Brier 0.239 / Top-5 命中率 99.7% / 模拟综合分 0.231 |

---

## 1. 项目目录结构

```
智能量化投资策略建模挑战赛/
├── README.md                   ← 本文件
├── problem.docx                ← 比赛原始题目（只读参考）
├── LICENSE                     ← MIT 协议
├── requirements.txt            ← 依赖列表
├── .gitignore                  ← Git 忽略
│
├── train/                      ← 训练集（4375 股 × 2794 日）
│   └── train.csv
├── test/                       ← 测试集（1500 股 × 20 日）
│   ├── test.csv
│   └── README_*.md
├── submission_template/        ← 官方提交模板
│   ├── sample_submission.csv
│   └── README_*.md
│
├── code/                       ← 全部源代码
│   ├── adaptivepath/           ← 自研包（FCPFF 核心）
│   │   ├── dataset.py          ← 截断预训练样本构造
│   │   ├── window_features.py  ← 基础窗口特征 (68 维)
│   │   ├── window_features_v2.py ← 扩展窗口特征 (121 维)
│   │   ├── features.py         ← 序列级特征工程（保留，未用）
│   │   ├── trainer.py          ← 基础多模型训练器
│   │   └── trainer_v2.py       ← 含 Ranker 的训练器
│   └── scripts/                ← 入口脚本
│       ├── eda.py              ← 探索性数据分析
│       ├── run_pipeline.py     ← v2 流水线
│       ├── run_full.py         ← v3 流水线
│       ├── run_v4.py           ← v4 流水线（推荐）
│       ├── backtest.py         ← 本地回测
│       └── test_pipeline*.py   ← 小规模测试脚本
│
├── output/                     ← 运行结果
│   ├── pipeline_report.json    ← 最终报告
│   ├── Xtr_df.csv              ← OOF 训练数据
│   ├── oof_final.npy           ← OOF 最终预测
│   ├── test_final.npy          ← 测试集最终预测
│   ├── ytr_cls.npy             ← 训练标签（分类）
│   ├── ytr_reg.npy             ← 训练标签（回归）
│   ├── eda/eda_summary.json    ← EDA 报告
│   └── models/                 ← 保存的模型与校准器
│       ├── isotonic.pkl
│       ├── stacking_meta.pkl
│       └── feat_cols.json
│
├── submission/                 ← 提交结果
│   └── output/
│       ├── submission.csv      ← ★ 最终提交文件
│       └── submission_v*.csv   ← 各版本中间产物
│
├── logs/                       ← 运行日志
├── docs/                       ← 文档
│   ├── TECHNICAL.md            ← 详细技术方案
│   ├── USAGE.md                ← 复现指南
│   └── FIGURES.md              ← 关键结果图
│
└── tests/                      ← 单元测试
```

---

## 2. 一句话方案

> **把"截断预训练 + 多目标 + 多模型 Stacking + 概率校准"四级流水线串起来，让模型从训练集 4375 只 A 股学到"20 日形态 → 未来 20 日方向"的模式，再迁移到测试集 1500 只独立资产上做概率预测。**

---

## 3. 快速复现

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 一键运行（推荐 v4）

```bash
# 在 Windows PowerShell 下，切换到项目根目录
cd E:\智能量化投资策略建模挑战赛

# 跑完整流水线（~ 7-10 分钟）
python code\scripts\run_v4.py --n_stocks 4375 --n_per_stock 5 --num_boost 1500 --out submission_v4.csv
```

### 3.3 提交

最终提交文件：

```
submission/output/submission.csv
```

格式：
```csv
code,up_factor
STOCK_0001,0.452464
STOCK_0002,0.452464
...
```

---

## 4. 关键创新点（与一般作品的区别）

| 一般方案 | 我们的方案 |
|---|---|
| 把 train.csv 直接喂给 LightGBM 一把梭 | **截断预训练**：用训练集 4375 只股的多窗口学"20 日 → 未来 20 日"模式 |
| 只用 close 做时序特征 | **121 维多尺度形态特征**：动量 / 反转 / 波动率 / 量价 / 形态识别 / 分形 |
| 单一 LightGBM | **10 模型集成**：5 个分类 + 3 个回归 + 1 个 Ranker + 1 个贝叶斯基准 |
| Brier Score 不优化 | **Isotonic 校准 + Top-K 软重排**，专门优化 Brier 与 Top-5 排序 |
| 没有 walk-forward 验证 | **5 折 OOF 评估 + bootstrap top-5 模拟**，本地模拟综合分 0.231 |

---

## 5. 详细文档

- 详细技术方案：[`docs/TECHNICAL.md`](docs/TECHNICAL.md)
- 复现指南：[`docs/USAGE.md`](docs/USAGE.md)

---

## 6. 致谢

- 比赛主办方：科大讯飞
- 数据来源：沪深 A 股市场公开行情
- 工具：LightGBM, XGBoost, CatBoost, scikit-learn, PyTorch
