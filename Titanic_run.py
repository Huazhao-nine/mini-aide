"""
Titanic 任务入口。

这个脚本复用当前 mini-aide 的 Agent / Interpreter 主循环，但把任务定义切换为
Kaggle Titanic 二分类。为了统一成通用 score 契约，这里把最终优化目标定义为：

    FINAL_SCORE = 1 - 5 折 OOF accuracy

数值越小越好，因此不需要改动现有的 evaluator / journal 比较逻辑。
"""

import io
import os
import warnings

import pandas as pd

from config import max_steps, num_drafts
from core.agent import Agent

warnings.filterwarnings("ignore")


def get_data_description(df: pd.DataFrame, full_columns: bool = True) -> str:
    """
    生成 Titanic 数据预览文本。

    保留目标列、ID 列、类别/数值列和缺失率概览，帮助 LLM 快速写出稳定的特征处理。
    """
    cols = list(df.columns)
    target_col = "Survived"
    id_col = "PassengerId"

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    categorical_cols = [c for c in cols if c not in numeric_cols]
    missing_ratio = (df.isna().mean() * 100).sort_values(ascending=False)

    buffer = io.StringIO()
    buffer.write("## Data Preview\n")
    buffer.write(f"- rows: {df.shape[0]}\n")
    buffer.write(f"- cols: {df.shape[1]}\n\n")

    buffer.write("## Key Columns\n")
    buffer.write(f"- id column: `{id_col}`\n")
    if target_col in cols:
        surv_rate = float(df[target_col].mean())
        buffer.write(f"- target column: `{target_col}` (binary 0/1, positive rate={surv_rate:.4f})\n\n")
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Feature Types\n")
    buffer.write(f"- numeric columns ({len(numeric_cols)}): {numeric_cols}\n")
    buffer.write(f"- categorical/text columns ({len(categorical_cols)}): {categorical_cols}\n\n")

    buffer.write("## Missing Rate Top Columns\n")
    for name, ratio in missing_ratio.head(8).items():
        buffer.write(f"- {name}: {ratio:.2f}% missing\n")
    buffer.write("\n")

    if full_columns:
        buffer.write("## Full Column List\n")
        buffer.write("```python\n")
        buffer.write("columns = [\n")
        for c in cols:
            buffer.write(f"    '{c}',\n")
        buffer.write("]\n")
        buffer.write("```\n")

    return buffer.getvalue()


def build_task_prompt(train_path: str, test_path: str, data_desc_str: str) -> str:
    """
    组装 Titanic 任务 prompt。

    这里显式要求二分类协议，但为了兼容当前 evaluator，最终必须打印：
    `FINAL_SCORE=<1 - OOF accuracy>`
    """
    return f"""
# Task: Titanic - Machine Learning from Disaster

You are solving a Kaggle-style binary classification task on tabular passenger data.
Goal: predict whether each passenger survived.

## Evaluation Protocol
- target column: `Survived`
- task type: binary classification
- leaderboard intuition: higher accuracy is better
- unified score definition:
  - the last line must be `FINAL_SCORE=<number>`
  - define `FINAL_SCORE = 1.0 - 5-fold OOF accuracy`
  - therefore lower is better, and a smaller FINAL_SCORE means a better classifier

### Validation Requirements
1) You must use **5-fold Stratified CV**:
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - final reported metric must be **OOF classification error = 1 - OOF accuracy**
2) Do not use a single holdout as the main score
3) If you tune hyperparameters, tune against 5-fold OOF classification error
4) If you ensemble, only ensemble fold models or OOF-validated models

### Output Contract
- The script must save a Kaggle submission file with columns:
  - `PassengerId`
  - `Survived`
- The script must write that file to the canonical submission path requested by the framework
- The very last non-empty output line must be:
  `FINAL_SCORE=<number>`
  where the number is exactly `1 - OOF accuracy`

## Data Paths
- train set: `{train_path}`
- test set: `{test_path}`

## Data Structure
{data_desc_str}

## Leakage / Protocol Rules
- never use `Survived` as an input feature
- fit imputers, encoders, scalers, selectors and any learned preprocessing only on the training fold
- keep train/test feature columns aligned explicitly
- do not use passenger identifiers as predictive targets or leak labels across folds
- do not use the public sample labels or any external survival labels

## Modeling Guidance
- start with a stable baseline first
- suitable model families include logistic regression, random forest, gradient boosting, xgboost/lightgbm if available, or a small MLP
- prefer compact, reliable preprocessing over large EDA blocks
- if categorical handling is needed, use robust sklearn pipelines or clear manual preprocessing
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    启动前的基本数据检查。
    """
    target_col = "Survived"
    id_col = "PassengerId"

    print("\n" + "=" * 60)
    print("Titanic Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Both train/test must contain id column: {id_col}")

    print(f"target positive rate: {train_df[target_col].mean():.4f}")
    print(f"train missing cells: {int(train_df.isna().sum().sum())}")
    print(f"test missing cells : {int(test_df.isna().sum().sum())}")


def main() -> None:
    print(f"[Mini-AIDE Titanic] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "titanic")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing Titanic train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing Titanic test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=True)
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
    )

    workdir = os.path.join(base_dir, "working_titanic")
    os.makedirs(workdir, exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE Titanic] done")


if __name__ == "__main__":
    main()
