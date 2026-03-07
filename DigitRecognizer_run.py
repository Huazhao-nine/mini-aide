"""
Digit Recognizer 任务入口。

任务来自 Kaggle Digit Recognizer（MNIST CSV 版本）。为了统一成通用 score 契约，
这里使用如下目标：

    FINAL_SCORE = 1 - 5-fold OOF accuracy

这样仍然是越小越好，同时保留分类任务的本质。
"""

import io
import os
import warnings

import pandas as pd

from config import max_steps, num_drafts
from core.agent import Agent

warnings.filterwarnings("ignore")


def get_data_description(df: pd.DataFrame, full_columns: bool = False) -> str:
    """
    生成 Digit Recognizer 数据预览。
    """
    cols = list(df.columns)
    target_col = "label"
    pixel_cols = [c for c in cols if c != target_col]

    buffer = io.StringIO()
    buffer.write("## Data Preview\n")
    buffer.write(f"- rows: {df.shape[0]}\n")
    buffer.write(f"- cols: {df.shape[1]}\n")
    buffer.write(f"- pixel feature count: {len(pixel_cols)}\n")
    buffer.write("- image shape: 28x28 grayscale\n\n")

    buffer.write("## Target\n")
    if target_col in cols:
        label_dist = df[target_col].value_counts(normalize=True).sort_index()
        dist_str = ", ".join([f"{int(k)}:{v:.3f}" for k, v in label_dist.items()])
        buffer.write(f"- target column: `{target_col}`\n")
        buffer.write(f"- class distribution: {dist_str}\n\n")
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Pixel Range\n")
    pixel_min = float(df[pixel_cols].min().min())
    pixel_max = float(df[pixel_cols].max().max())
    buffer.write(f"- raw pixel min/max: {pixel_min:.1f} / {pixel_max:.1f}\n")
    buffer.write("- pixels are flattened in row-major order\n\n")

    if full_columns:
        buffer.write("## Full Column List\n")
        buffer.write("```python\n")
        buffer.write("columns = [\n")
        for c in cols:
            buffer.write(f"    '{c}',\n")
        buffer.write("]\n")
        buffer.write("```\n")
    else:
        preview_cols = cols[:12]
        buffer.write("## Column Preview\n")
        buffer.write(f"- first columns: {preview_cols}\n")
        buffer.write(f"- last columns: {cols[-5:]}\n")

    return buffer.getvalue()


def build_task_prompt(train_path: str, test_path: str, data_desc_str: str) -> str:
    """
    组装 Digit Recognizer prompt。
    """
    return f"""
# Task: Digit Recognizer

You are solving the Kaggle Digit Recognizer task, a 10-class image classification problem
using flattened 28x28 grayscale MNIST digits.

This is a neural network homework style task.

## Evaluation Protocol
- target column: `label`
- task type: multiclass classification (classes 0-9)
- leaderboard intuition: higher accuracy is better
- unified score definition:
  - the last line must be `FINAL_SCORE=<number>`
  - define `FINAL_SCORE = 1.0 - 5-fold OOF accuracy`
  - lower FINAL_SCORE means a better classifier

### Validation Requirements
1) You must use **5-fold Stratified CV**:
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - final score must be **1 - OOF accuracy**
2) Do not use a single holdout as the main score
3) If you tune hyperparameters, tune against 5-fold OOF classification error
4) Keep preprocessing and normalization fit logic fold-safe

### Output Contract
- Save `./submission.csv`
- Submission columns must be:
  - `ImageId`
  - `Label`
- `ImageId` starts from 1 and increments by 1 over the test rows
- The very last non-empty output line must be:
  `FINAL_SCORE=<number>`
  where the number is exactly `1 - OOF accuracy`

## Modeling Guidance
- DRAFT baseline must be a **small PyTorch CNN**
- prioritize simple, reliable CNN changes first:
  - conv width
  - batch norm
  - dropout
  - optimizer / lr
  - augmentation only if it is lightweight and fold-safe
- avoid huge models or long training loops
- restore best checkpoint before generating OOF/test predictions

## Data Paths
- train set: `{train_path}`
- test set: `{test_path}`

## Data Structure
{data_desc_str}

## Leakage / Protocol Rules
- never use the label as an input feature
- reshape pixels to 28x28 consistently for both train and test
- if you normalize inputs, apply the same transformation to test data
- keep OOF predictions strictly out-of-fold
- do not use external MNIST labels or pretrained models trained on the Kaggle test labels
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    启动前的数据检查。
    """
    target_col = "label"

    print("\n" + "=" * 60)
    print("Digit Recognizer Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if test_df.shape[1] != train_df.shape[1] - 1:
        raise ValueError("Test split must contain exactly the feature columns without the label column")

    print(f"label classes: {sorted(train_df[target_col].unique().tolist())}")
    print(f"pixel range train: {train_df.drop(columns=[target_col]).min().min()} .. {train_df.drop(columns=[target_col]).max().max()}")
    print(f"pixel range test : {test_df.min().min()} .. {test_df.max().max()}")


def main() -> None:
    print(f"[Mini-AIDE DigitRecognizer] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "digit_recognizer")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing Digit Recognizer train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing Digit Recognizer test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=False)
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
    )

    workdir = os.path.join(base_dir, "working_digit_recognizer")
    os.makedirs(workdir, exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE DigitRecognizer] done")


if __name__ == "__main__":
    main()
