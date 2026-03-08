"""
Forest Cover Type 任务入口。

任务来自 Kaggle Forest Cover Type Prediction。
为了兼容当前 mini-aide 的统一契约，这里定义：

    FINAL_SCORE = 1 - 5-fold OOF accuracy

数值越小越好。

这个任务在实验矩阵中扮演“结构化表格多分类 benchmark”的角色：
- 既有连续地形特征，也有大块 one-hot wilderness / soil 结构；
- 很适合检验 AIDE 是否会把搜索优先放在树模型和稳健验证上；
- 也适合展示任务入口 prompt 如何把“经验性高分方向”前置给 Agent。
"""

import io
import os
import warnings

import pandas as pd

from utils.config import max_steps, num_drafts
from core.agent import Agent

warnings.filterwarnings("ignore")


def get_data_description(df: pd.DataFrame, full_columns: bool = False) -> str:
    """
    生成 Forest Cover Type 的静态数据概览。

    这里重点把 wilderness 和 soil 这两组结构化 one-hot 特征显式告诉 LLM，帮助它更快
    写出合理的 tabular baseline。
    """
    cols = list(df.columns)
    target_col = "Cover_Type"
    id_col = "Id"
    wilderness_cols = [c for c in cols if c.startswith("Wilderness_Area")]
    soil_cols = [c for c in cols if c.startswith("Soil_Type")]

    buffer = io.StringIO()
    buffer.write("## Data Preview\n")
    buffer.write(f"- rows: {df.shape[0]}\n")
    buffer.write(f"- cols: {df.shape[1]}\n\n")

    buffer.write("## Key Columns\n")
    buffer.write(f"- id column: `{id_col}`\n")
    if target_col in cols:
        dist = df[target_col].value_counts(normalize=True).sort_index()
        dist_str = ", ".join([f"{int(k)}:{v:.3f}" for k, v in dist.items()])
        buffer.write(f"- target column: `{target_col}`\n")
        buffer.write(f"- class distribution: {dist_str}\n\n")
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Structured Feature Groups\n")
    buffer.write(f"- wilderness one-hot columns ({len(wilderness_cols)}): {wilderness_cols}\n")
    buffer.write(f"- soil one-hot columns ({len(soil_cols)}): first={soil_cols[:5]}, last={soil_cols[-5:]}\n\n")
    buffer.write("## Modeling Hints\n")
    buffer.write("- numeric terrain/topography features combine with structured one-hot wilderness and soil groups\n")
    buffer.write("- tree ensembles usually work better than generic neural baselines here\n")
    buffer.write("- class labels are multiclass integers 1..7, not probabilities\n\n")

    if full_columns:
        buffer.write("## Full Column List\n")
        buffer.write("```python\n")
        buffer.write("columns = [\n")
        for c in cols:
            buffer.write(f"    '{c}',\n")
        buffer.write("]\n")
        buffer.write("```\n")
    else:
        buffer.write("## Column Preview\n")
        buffer.write(f"- first columns: {cols[:15]}\n")
        buffer.write(f"- last columns: {cols[-15:]}\n")

    return buffer.getvalue()


def build_task_prompt(train_path: str, test_path: str, data_desc_str: str, submission_path: str) -> str:
    """
    组装 Forest Cover Type 的任务规范。

    它固定了多分类 accuracy、本地 OOF 协议和整数标签 submission 约束。
    """
    return f"""
# Task: Forest Cover Type Prediction

You are solving a Kaggle-style multiclass tabular classification task.
Goal: predict the forest cover type class for each row.

## Evaluation Protocol
- target column: `Cover_Type`
- task type: multiclass classification with labels 1..7
- leaderboard intuition: higher accuracy is better
- unified score definition:
  - the last line must be `FINAL_SCORE=<number>`
  - define `FINAL_SCORE = 1.0 - 5-fold OOF accuracy`
  - lower FINAL_SCORE means a better model

### Validation Requirements
1) You must use **5-fold Stratified CV**:
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - final score must be `1 - OOF accuracy`
2) Do not use a single holdout as the main score
3) If you tune hyperparameters, tune against `1 - OOF accuracy`
4) Fit all preprocessing only on the training fold

### Local-to-Leaderboard Alignment
- OOF accuracy should track Kaggle accuracy if class handling and fold logic are correct
- common leaderboard failures are:
  - mis-handling class labels as zero-based when exporting
  - fitting preprocessors on full train before CV
  - breaking column alignment on one-hot feature blocks
  - overfitting to small local holdouts instead of robust stratified OOF

### Output Contract
- Save `{submission_path}`
- Submission columns must be:
  - `Id`
  - `Cover_Type`
- `Cover_Type` must be an integer class label in 1..7
- Do not write to `./submission.csv`; use the exact path above
- The very last non-empty output line must be:
  `FINAL_SCORE=<number>`
  where the number is exactly `1 - OOF accuracy`

## Data Paths
- train set: `{train_path}`
- test set: `{test_path}`

## Data Structure
{data_desc_str}

## Leakage / Protocol Rules
- never use `Cover_Type` as an input feature
- fit scalers, selectors and learned preprocessing only on the training fold
- keep train/test feature columns aligned explicitly
- do not leak validation labels into feature engineering
- preserve the integer label space 1..7 exactly in submission output

## Modeling Guidance
- preferred family order:
  1. extra trees / random forest style baseline
  2. boosting-style tree model if available
  3. simple validated ensemble only after single-model OOF is strong
- recommended first-wave improvements:
  - leverage terrain distance features directly
  - preserve wilderness and soil one-hot group structure
  - class-balanced evaluation awareness while still optimizing accuracy
  - avoid unnecessary scaling for pure tree models
- prefer robust validation over large EDA blocks
- do not start with generic MLPs unless tree baselines are already exhausted
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """启动前检查标签空间与特征维度，避免多分类任务在最初几轮就跑偏。"""
    target_col = "Cover_Type"
    id_col = "Id"

    print("\n" + "=" * 60)
    print("Forest Cover Type Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Both train/test must contain id column: {id_col}")

    print(f"class labels: {sorted(train_df[target_col].unique().tolist())}")
    print(f"train feature count: {train_df.shape[1] - 2}")


def main() -> None:
    # 这里负责准备 static task context；真正的 trial-and-error 仍在统一 Agent 中完成。
    print(f"[Mini-AIDE ForestCover] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "forest_cover_type")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing Forest Cover Type train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing Forest Cover Type test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=False)
    submission_relpath = "working/submission-forest-cover.csv"
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
        submission_path=f"./{submission_relpath}",
    )

    workdir = os.path.join(base_dir, "working_forest_cover")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "working"), exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
        submission_relpath=submission_relpath,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE ForestCover] done")


if __name__ == "__main__":
    main()
