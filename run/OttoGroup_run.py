"""
Otto Group Product Classification 任务入口。

任务来自 Kaggle Otto Group Product Classification Challenge。
为了兼容当前 mini-aide 的统一契约，这里定义：

    FINAL_SCORE = 5-fold OOF multiclass log loss

数值越小越好。

这个任务是一个很适合复试讲解的“多分类概率建模 benchmark”：
- 它不像 Titanic 那样只看准确率，而是直接优化多分类 logloss；
- 因此能展示 AIDE 不只是会写分类器，还会处理 class order、概率质量和 submission 约束；
- 也能体现 solution tree 中的 improve 节点如何围绕“概率输出质量”持续优化。
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
    生成 Otto Group 的数据预览。

    这里强调的是“高维数值特征 + 9 类概率输出”的任务形态，为后续 prompt 提供强先验。
    """
    cols = list(df.columns)
    target_col = "target"
    id_col = "id"
    feature_cols = [c for c in cols if c not in {id_col, target_col}]

    buffer = io.StringIO()
    buffer.write("## Data Preview\n")
    buffer.write(f"- rows: {df.shape[0]}\n")
    buffer.write(f"- cols: {df.shape[1]}\n")
    buffer.write(f"- feature count: {len(feature_cols)}\n\n")

    buffer.write("## Key Columns\n")
    buffer.write(f"- id column: `{id_col}`\n")
    if target_col in cols:
        class_dist = df[target_col].value_counts(normalize=True).sort_index()
        dist_str = ", ".join([f"{k}:{v:.3f}" for k, v in class_dist.items()])
        buffer.write(f"- target column: `{target_col}`\n")
        buffer.write(f"- class distribution: {dist_str}\n\n")
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Features\n")
    buffer.write("- all input features are numeric count-like columns named `feat_1` ... `feat_93`\n")
    buffer.write(f"- first features: {feature_cols[:10]}\n")
    buffer.write(f"- last features: {feature_cols[-10:]}\n\n")
    buffer.write("## Modeling Hints\n")
    buffer.write("- probability quality matters more than raw class prediction accuracy\n")
    buffer.write("- preserve class order exactly as `Class_1` ... `Class_9`\n")
    buffer.write("- dense numeric features often work well with boosting and calibrated linear baselines\n\n")

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
    组装 Otto Group 任务规范。

    它把多分类 logloss、submission 概率列顺序和数值稳定性要求固化到 prompt 中。
    """
    return f"""
# Task: Otto Group Product Classification Challenge

You are solving a Kaggle-style multiclass tabular classification task.
Goal: predict the class probabilities for each product.

## Evaluation Protocol
- target column: `target`
- task type: multiclass classification with 9 classes
- leaderboard intuition: lower multiclass log loss is better
- unified score definition:
  - the last line must be `FINAL_SCORE=<number>`
  - define `FINAL_SCORE = 5-fold OOF multiclass log loss`
  - lower FINAL_SCORE means a better model

### Validation Requirements
1) You must use **5-fold Stratified CV**:
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - final score must be OOF multiclass log loss
2) Do not use a single holdout as the main score
3) If you tune hyperparameters, tune against 5-fold OOF multiclass log loss
4) Fit all preprocessing only on the training fold

### Local-to-Leaderboard Alignment
- OOF multiclass log loss is a strong proxy for Kaggle log loss if probabilities are numerically stable
- common leaderboard failures are:
  - wrong class-column order in submission
  - probabilities not summing to 1
  - overconfident predictions without calibration
  - clipping or thresholding probabilities incorrectly

### Output Contract
- Save `{submission_path}`
- Submission columns must be:
  - `id`
  - `Class_1`
  - `Class_2`
  - `Class_3`
  - `Class_4`
  - `Class_5`
  - `Class_6`
  - `Class_7`
  - `Class_8`
  - `Class_9`
- The class probability columns must sum to 1 for each row
- Do not write to `./submission.csv`; use the exact path above
- The very last non-empty output line must be:
  `FINAL_SCORE=<number>`
  where the number is exactly the 5-fold OOF multiclass log loss

## Data Paths
- train set: `{train_path}`
- test set: `{test_path}`

## Data Structure
{data_desc_str}

## Leakage / Protocol Rules
- never use `target` as an input feature
- fit scalers, selectors and any learned preprocessing only on the training fold
- keep class order stable between OOF evaluation and submission output
- do not leak validation labels into thresholding or calibration
- any calibration must be fitted in a fold-safe way

## Modeling Guidance
- preferred family order:
  1. multinomial logistic / linear probability baseline
  2. boosting-style tabular model
  3. validated soft-voting blend only if class order and probability normalization remain exact
- prioritize probability quality, calibration, and numerical stability over raw accuracy
- do not start with hard-voting, argmax-only thinking, or non-probabilistic models without proper probability output
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """启动前检查类别标签空间和测试列结构，避免 submission 维度出错。"""
    target_col = "target"
    id_col = "id"

    print("\n" + "=" * 60)
    print("Otto Group Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Both train/test must contain id column: {id_col}")

    print(f"class labels: {sorted(train_df[target_col].unique().tolist())}")
    print(f"feature count: {train_df.shape[1] - 2}")


def main() -> None:
    # 本入口的职责仍然是任务描述标准化，而不是直接替代论文中的搜索主循环。
    print(f"[Mini-AIDE OttoGroup] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "otto_group")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing Otto Group train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing Otto Group test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=False)
    submission_relpath = "working/submission-otto-group.csv"
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
        submission_path=f"./{submission_relpath}",
    )

    workdir = os.path.join(base_dir, "working_otto_group")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "working"), exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
        submission_relpath=submission_relpath,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE OttoGroup] done")


if __name__ == "__main__":
    main()
