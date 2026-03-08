"""
Spaceship Titanic 任务入口。

任务来自 Kaggle Spaceship Titanic。
为了兼容当前 mini-aide 的统一契约，这里定义：

    FINAL_SCORE = 1 - 5-fold OOF accuracy

数值越小越好。

这个任务可看作 Titanic 的升级版表格分类 benchmark：
- 有更复杂的类别/字符串字段和显式组信息；
- 可以很好地测试 AIDE 是否会从 submission 失败逐步调到稳健的特征工程主线；
- 也适合观察 search policy 是否会持续围绕高价值 tabular 分支改进。
"""

import io
import os
import warnings

import pandas as pd

from utils.config import max_steps, num_drafts
from core.agent import Agent

warnings.filterwarnings("ignore")


def get_data_description(df: pd.DataFrame, full_columns: bool = True) -> str:
    """
    生成 Spaceship Titanic 的静态数据概览。

    它把 Cabin、消费列、组信息等高价值结构提前注入 prompt，减少 LLM 在早期轮次的盲目探索。
    """
    cols = list(df.columns)
    target_col = "Transported"
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
        pos_rate = float(df[target_col].astype(int).mean())
        buffer.write(f"- target column: `{target_col}` (binary, positive rate={pos_rate:.4f})\n\n")
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Feature Types\n")
    buffer.write(f"- numeric columns ({len(numeric_cols)}): {numeric_cols}\n")
    buffer.write(f"- categorical/text columns ({len(categorical_cols)}): {categorical_cols}\n\n")

    buffer.write("## Missing Rate Top Columns\n")
    for name, ratio in missing_ratio.head(10).items():
        buffer.write(f"- {name}: {ratio:.2f}% missing\n")
    buffer.write("\n")

    buffer.write("## Useful Feature Blocks\n")
    buffer.write("- `Cabin` can often be split into deck / number / side\n")
    buffer.write("- expenditure columns: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`\n")
    buffer.write("- `Name` and `PassengerId` may encode group/family structure, but use fold-safe processing\n")
    buffer.write("- zero-spend and cryosleep interactions are often useful\n\n")

    if full_columns:
        buffer.write("## Full Column List\n")
        buffer.write("```python\n")
        buffer.write("columns = [\n")
        for c in cols:
            buffer.write(f"    '{c}',\n")
        buffer.write("]\n")
        buffer.write("```\n")

    return buffer.getvalue()


def build_task_prompt(train_path: str, test_path: str, data_desc_str: str, submission_path: str) -> str:
    """
    组装 Spaceship Titanic 的任务规范。

    这里重点固定 accuracy 导向的本地评测、组特征的防泄漏要求和提交格式要求。
    """
    return f"""
# Task: Spaceship Titanic

You are solving a Kaggle-style binary classification task on passenger records.
Goal: predict whether each passenger was transported.

## Evaluation Protocol
- target column: `Transported`
- task type: binary classification
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
- local stratified OOF accuracy should track Kaggle accuracy if feature engineering is fold-safe
- common leaderboard failures are:
  - leaking group statistics across folds
  - splitting `Cabin` or `Name` inconsistently between train and test
  - fitting encoders on full train before CV
  - converting boolean targets / predictions inconsistently at export time

### Output Contract
- Save `{submission_path}`
- Submission columns must be:
  - `PassengerId`
  - `Transported`
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
- never use `Transported` as an input feature
- fit imputers, encoders, scalers and selectors only on the training fold
- keep train/test columns aligned explicitly
- do not leak family/group labels across folds
- do not use sample submission labels or external target data
- if you derive group features from `PassengerId`, `Name`, or shared tickets/cabins, keep the logic purely unsupervised

## Modeling Guidance
- preferred family order:
  1. robust tabular linear / tree baseline
  2. boosting-style model after feature handling is stable
  3. light ensemble only after single-model OOF is strong
- recommended first-wave features:
  - `CabinDeck`, `CabinNum`, `CabinSide`
  - total spend and zero-spend flags
  - cryosleep spending interaction
  - group size from `PassengerId`
  - surname / group consistency features if implemented safely
- use compact and reliable preprocessing
- do not start with deep tabular networks or large ensembles
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """启动前核对目标列、ID 列和缺失规模，确保静态上下文可靠。"""
    target_col = "Transported"
    id_col = "PassengerId"

    print("\n" + "=" * 60)
    print("Spaceship Titanic Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Both train/test must contain id column: {id_col}")

    print(f"target positive rate: {train_df[target_col].astype(int).mean():.4f}")
    print(f"train missing cells: {int(train_df.isna().sum().sum())}")
    print(f"test missing cells : {int(test_df.isna().sum().sum())}")


def main() -> None:
    # `main()` 负责把任务定义标准化；真正的改进搜索由 Agent 在统一框架内执行。
    print(f"[Mini-AIDE SpaceshipTitanic] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "spaceship_titanic")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing Spaceship Titanic train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing Spaceship Titanic test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=True)
    submission_relpath = "working/submission-spaceship-titanic.csv"
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
        submission_path=f"./{submission_relpath}",
    )

    workdir = os.path.join(base_dir, "working_spaceship_titanic")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "working"), exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
        submission_relpath=submission_relpath,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE SpaceshipTitanic] done")


if __name__ == "__main__":
    main()
