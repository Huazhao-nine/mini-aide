"""
House Prices 任务入口。

任务来自 Kaggle House Prices - Advanced Regression Techniques。
为了兼容当前 mini-aide 的统一契约，这里定义：

    FINAL_SCORE = 5-fold OOF RMSLE

数值越小越好，与 Kaggle leaderboard 的方向一致。

这个任务在论文复现实验中扮演“复杂表格回归 benchmark”的角色：
- 它比课程回归任务有更多类别列、缺失值和目标变换需求；
- 很适合检验 summarization operator 是否能记住有效特征工程经验；
- 也能体现 search policy 是否会把搜索集中在真正影响 leaderboard 的方向上。
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
    生成 House Prices 的静态数据预览。

    这里把高缺失列、数值/类别列和关键特征块提前写进 prompt，作用是把人工做题时常见的
    先验知识“前置”给 coding operator。
    """
    cols = list(df.columns)
    target_col = "SalePrice"
    id_col = "Id"

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
        y = df[target_col]
        buffer.write(
            f"- target column: `{target_col}` "
            f"(min={float(y.min()):.2f}, max={float(y.max()):.2f}, mean={float(y.mean()):.2f})\n\n"
        )
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Feature Types\n")
    buffer.write(f"- numeric columns ({len(numeric_cols)}): {numeric_cols[:20]}\n")
    buffer.write(f"- categorical columns ({len(categorical_cols)}): {categorical_cols[:20]}\n\n")

    buffer.write("## Missing Rate Top Columns\n")
    for name, ratio in missing_ratio.head(10).items():
        buffer.write(f"- {name}: {ratio:.2f}% missing\n")
    buffer.write("\n")

    buffer.write("## Useful Feature Blocks\n")
    buffer.write("- quality / condition: `OverallQual`, `OverallCond`\n")
    buffer.write("- size / area: `GrLivArea`, `LotArea`, basement and garage area columns\n")
    buffer.write("- age / year: `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`\n")
    buffer.write("- location: `Neighborhood`, `MSZoning`\n")
    buffer.write("- many categorical columns are sparse and require careful missing handling\n\n")

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
    组装 House Prices 任务规范。

    它把 leaderboard 指标 RMSLE、log1p 目标处理、submission 契约和防泄漏规则固定成
    统一上下文，供整棵 solution tree 复用。
    """
    return f"""
# Task: House Prices - Advanced Regression Techniques

You are solving a Kaggle-style tabular regression task.
Goal: predict `SalePrice` for each house.

## Evaluation Protocol
- target column: `SalePrice`
- task type: regression
- Kaggle leaderboard intuition: lower RMSLE is better
- unified score definition:
  - the last line must be `FINAL_SCORE=<number>`
  - define `FINAL_SCORE = 5-fold OOF RMSLE`
  - lower FINAL_SCORE means a better model

### Validation Requirements
1) You must use **5-fold CV**:
   - `KFold(n_splits=5, shuffle=True, random_state=42)`
   - final score must be OOF RMSLE
2) Do not use a single holdout as the main score
3) If you tune hyperparameters, tune against 5-fold OOF RMSLE
4) Fit all preprocessing only on the training fold

### Local-to-Leaderboard Alignment
- RMSLE is close to Kaggle public score if target handling and non-negative prediction logic are correct
- common leaderboard failures are:
  - computing RMSE on raw target instead of RMSLE
  - applying preprocessing on full train before CV
  - forgetting to clip negative predictions before RMSLE / submission
  - fitting target encodings or frequency encodings with validation leakage

### RMSLE Details
- RMSLE should be computed on the original target scale with non-negative predictions
- clip negative predictions before applying `log1p`
- if you train on `log1p(SalePrice)`, report OOF RMSLE on the original target scale

### Output Contract
- Save `{submission_path}`
- Submission columns must be:
  - `Id`
  - `SalePrice`
- Do not write to `./submission.csv`; use the exact path above
- The very last non-empty output line must be:
  `FINAL_SCORE=<number>`
  where the number is exactly the 5-fold OOF RMSLE

## Data Paths
- train set: `{train_path}`
- test set: `{test_path}`

## Data Structure
{data_desc_str}

## Leakage / Protocol Rules
- never use `SalePrice` as an input feature
- fit imputers, encoders, scalers and selectors only on the training fold
- keep train/test feature columns aligned explicitly
- do not use any target statistics computed with validation-fold labels
- if using target transformation, ensure OOF and test predictions are transformed back correctly

## Modeling Guidance
- preferred family order:
  1. `log1p(SalePrice)` + strong linear / elastic-net baseline
  2. tree boosting / lightgbm-style tabular model if available
  3. simple blend of validated single models only after both are strong
- recommended first-wave improvements:
  - robust missing-value handling by column semantics
  - sparse one-hot for categoricals
  - skewed numeric feature transforms
  - neighborhood / quality / size interactions kept simple and fold-safe
- prioritize reliable preprocessing over long EDA
- do not start with stacking, pseudo-labeling, or highly custom encoders
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    启动前的静态输入检查。

    这一步用于保证任务定义中的目标列、ID 列和缺失情况与 prompt 一致。
    """
    target_col = "SalePrice"
    id_col = "Id"

    print("\n" + "=" * 60)
    print("House Prices Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Both train/test must contain id column: {id_col}")

    print(f"target mean: {train_df[target_col].mean():.4f}")
    print(f"target median: {train_df[target_col].median():.4f}")
    print(f"train missing cells: {int(train_df.isna().sum().sum())}")
    print(f"test missing cells : {int(test_df.isna().sum().sum())}")


def main() -> None:
    # 这里不直接做搜索，而是把 House Prices 任务整理成论文里的 static task context。
    print(f"[Mini-AIDE HousePrices] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "house_prices")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing House Prices train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing House Prices test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=True)
    submission_relpath = "working/submission-house-prices.csv"
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
        submission_path=f"./{submission_relpath}",
    )

    workdir = os.path.join(base_dir, "working_house_prices")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "working"), exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
        submission_relpath=submission_relpath,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE HousePrices] done")


if __name__ == "__main__":
    main()
