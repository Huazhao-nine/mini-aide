import io
import os
import warnings

import pandas as pd

from config import max_steps, num_drafts, debug_prob, max_debug_depth, timeout
from core.agent import Agent
from core.interpreter import Interpreter

warnings.filterwarnings("ignore")


def get_data_description(df: pd.DataFrame, full_columns: bool = True) -> str:
    cols = list(df.columns)
    target_cols = [c for c in cols if c == "tested_positive_day3"]
    day1_cols = [c for c in cols if c.endswith("_day1")]
    day2_cols = [c for c in cols if c.endswith("_day2")]
    day3_cols = [c for c in cols if c.endswith("_day3")]

    buffer = io.StringIO()
    buffer.write("## 数据概览\n")
    buffer.write(f"- 行数: {df.shape[0]}\n")
    buffer.write(f"- 列数: {df.shape[1]}\n\n")

    buffer.write("## 目标列\n")
    if target_cols:
        buffer.write(f"- 目标列: `{target_cols[0]}`（范围约 0~100）\n\n")
    else:
        buffer.write("- 目标列: `tested_positive_day3`（未找到！）\n\n")

    buffer.write("## 特征分组（按 day 后缀）\n")
    buffer.write(f"- day1 特征数: {len(day1_cols)}\n")
    buffer.write(f"- day2 特征数: {len(day2_cols)}\n")
    buffer.write(f"- day3 特征数（允许使用普通 day3 特征，但不能用目标列）: {len(day3_cols)}\n\n")

    if full_columns:
        buffer.write("## 全量列名列表（可复制粘贴）\n")
        buffer.write("```python\n")
        buffer.write("columns = [\n")
        for c in cols:
            buffer.write(f"    '{c}',\n")
        buffer.write("]\n")
        buffer.write("```\n")

    return buffer.getvalue()


def build_task_prompt(train_path: str, test_path: str, data_desc_str: str) -> str:
    return f"""
# 任务：COVID-19 阳性率回归预测（Kaggle 风格）

你需要针对一个表格回归任务，自动迭代产出更强的 baseline。
可使用树模型、线性模型或 PyTorch MLP（任选合适路线）。

## 评估协议（必须严格遵守）
- 目标列：`tested_positive_day3`
- 回归任务
- 最终指标：**MSE**（越小越好）

### 重要：验证必须可靠（避免本地过拟合 / Kaggle 崩盘）
1) 你必须使用 **5 折交叉验证（OOF）**：
   - `KFold(n_splits=5, shuffle=True, random_state=42)`
   - 最终以 **OOF MSE** 作为报告指标
2) 不允许用单一 holdout 作为主要分数来源
3) 若使用超参搜索（如 Optuna），优化目标必须是 **5 折 OOF MSE**
   - 试验次数 <= 20
   - 不允许反复“刷同一个 holdout”
4) 允许集成，但只能基于 **fold 模型**（对 test 预测做 fold 平均）
   - 不允许“同一验证集 top-k trial 加权”这种刷分策略

### 输出契约（严格）
- 脚本必须在最后一行打印：
  `FINAL_MSE=<number>`
  其中 FINAL_MSE 必须是 **5 折 OOF MSE**
- 若存在测试集，必须保存提交文件到：
  `./working/submission.csv`
  列为：`id,tested_positive_day3`

## 数据路径
- 训练集: `{train_path}`
- 测试集: `{test_path}`

## 数据结构
{data_desc_str}

## 硬约束（防泄漏）
- 允许使用普通 day3 特征列
- 禁止把 `tested_positive_day3` 当作输入特征
- 禁止从任何 `tested_positive_*` 列派生特征
- 构造测试集特征时必须显式按列对齐（reindex）
- 若对 y 做归一化/标准化训练，FINAL_MSE 必须换回原始 0~100 尺度上计算

## 实现要求
- 直接实现训练 + 验证 pipeline，不做冗长 EDA
- 稳定可运行，尽量小步改进
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target_col = "tested_positive_day3"
    print("\n" + "=" * 60)
    print("📋 数据检查")
    print("=" * 60)
    print(f"📊 train shape: {train_df.shape}")
    print(f"📊 test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"训练集中未找到目标列: {target_col}")

    tp_cols = [c for c in train_df.columns if c.startswith("tested_positive_")]
    print(f"⚠️ 训练集中的 tested_positive_* 列数量: {len(tp_cols)}（会在 prompt 中约束：禁止派生）")

    y = train_df[target_col]
    print(f"🎯 目标 min/max/mean: {y.min():.4f} / {y.max():.4f} / {y.mean():.4f}")

    if "id" not in test_df.columns:
        print("⚠️ 测试集没有 'id' 列。脚本需要自行用 index 生成 id（若需要）。")


def main():
    print(f"🚀 [Mini-AIDE] 启动 (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "input", "train.csv")
    test_path = os.path.join(base_dir, "input", "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=True)
    task_prompt = build_task_prompt(train_path, test_path, data_desc_str)

    workdir = os.path.join(base_dir, "working")
    os.makedirs(workdir, exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
    )
    agent.run(max_steps=max_steps)

    print("\n✅ [Mini-AIDE] 结束")


if __name__ == "__main__":
    main()