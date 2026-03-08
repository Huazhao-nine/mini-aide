"""
Porto Seguro 任务入口。

任务来自 Kaggle Porto Seguro's Safe Driver Prediction。
为了兼容当前 mini-aide 的统一契约，这里定义：

    FINAL_SCORE = 1 - 5-fold OOF normalized gini

数值越小越好。

这个任务是“排行榜导向二分类”的代表 benchmark：
- 本地代理指标不是 accuracy，而是更贴近 Kaggle 排名的 normalized gini；
- 这能检验 AIDE 是否会围绕排序质量而不是阈值分类去优化；
- 也能体现 prompt 里任务特定约束对 coding operator 的引导作用。
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
    生成 Porto Seguro 的结构化数据概览。

    它把 `ps_ind / ps_reg / ps_car` 等列族和缺失值哨兵等先验压缩成文本，以便后续 prompt
    更像“有经验选手的任务说明”。
    """
    cols = list(df.columns)
    target_col = "target"
    id_col = "id"
    feature_cols = [c for c in cols if c not in {id_col, target_col}]
    family_counts = {}
    for col in feature_cols:
        key = "_".join(col.split("_")[:2])
        family_counts[key] = family_counts.get(key, 0) + 1

    buffer = io.StringIO()
    buffer.write("## Data Preview\n")
    buffer.write(f"- rows: {df.shape[0]}\n")
    buffer.write(f"- cols: {df.shape[1]}\n")
    buffer.write(f"- feature count: {len(feature_cols)}\n\n")

    buffer.write("## Key Columns\n")
    buffer.write(f"- id column: `{id_col}`\n")
    if target_col in cols:
        pos_rate = float(df[target_col].mean())
        buffer.write(f"- target column: `{target_col}` (binary, positive rate={pos_rate:.4f})\n\n")
    else:
        buffer.write(f"- target column: `{target_col}` (not present in this split)\n\n")

    buffer.write("## Feature Families\n")
    for name, count in sorted(family_counts.items()):
        buffer.write(f"- {name}: {count}\n")
    buffer.write("\n")
    buffer.write("## Modeling Hints\n")
    buffer.write("- many Porto Seguro variants use `-1` as a missing-value sentinel; detect and handle this carefully\n")
    buffer.write("- ranking quality matters more than thresholded classification accuracy\n")
    buffer.write("- probability outputs must remain smooth and well ordered\n\n")

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
    组装 Porto Seguro 的任务规范。

    这里尤其强调 normalized gini、概率输出和 fold-safe 统计特征，是因为这些点最容易导致
    本地分数与 leaderboard 脱节。
    """
    return f"""
# Task: Porto Seguro's Safe Driver Prediction

You are solving a Kaggle-style binary classification task on insurance driver records.
Goal: predict the probability of claim risk.

## Evaluation Protocol
- target column: `target`
- task type: binary classification
- Kaggle leaderboard intuition: higher normalized gini is better
- unified score definition:
  - the last line must be `FINAL_SCORE=<number>`
  - define `FINAL_SCORE = 1.0 - 5-fold OOF normalized gini`
  - lower FINAL_SCORE means a better model

### Validation Requirements
1) You must use **5-fold Stratified CV**:
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - final score must be `1 - OOF normalized gini`
2) Do not use a single holdout as the main score
3) If you tune hyperparameters, tune against `1 - OOF normalized gini`
4) Fit all preprocessing only on the training fold

### Local-to-Leaderboard Alignment
- normalized gini on OOF probabilities should track Kaggle ranking quality if CV is stable
- common leaderboard failures are:
  - evaluating thresholded labels instead of probabilities
  - mishandling `-1` sentinel values
  - leaking frequency/target statistics across folds
  - writing malformed probability submissions

### Normalized Gini Details
- use probability predictions for the positive class
- implement normalized gini carefully and deterministically
- OOF score must be based only on out-of-fold predictions

### Output Contract
- Save `{submission_path}`
- Submission columns must be:
  - `id`
  - `target`
- `target` must be the predicted positive-class probability
- Do not write to `./submission.csv`; use the exact path above
- The very last non-empty output line must be:
  `FINAL_SCORE=<number>`
  where the number is exactly `1 - OOF normalized gini`

## Data Paths
- train set: `{train_path}`
- test set: `{test_path}`

## Data Structure
{data_desc_str}

## Leakage / Protocol Rules
- never use `target` as an input feature
- fit imputers, scalers and selectors only on the training fold
- keep OOF predictions strictly out-of-fold
- do not threshold probabilities before scoring
- do not use sample submission or external labels
- if using frequency/count/categorical statistics, compute them strictly inside each training fold

## Modeling Guidance
- preferred family order:
  1. regularized linear / logistic probability baseline
  2. boosting-style model with careful missing-value treatment
  3. light blend only after both ranking quality and probability format are validated
- recommended first-wave improvements:
  - detect binary/categorical/continuous groups
  - handle `-1` sentinel values cleanly
  - simple interactions inside `ps_ind`, `ps_reg`, `ps_car`
  - probability calibration only if done fold-safely
- keep ranking quality in mind; do not optimize thresholded accuracy
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """启动前检查目标分布和列结构，确保后续 prompt 建立在正确事实之上。"""
    target_col = "target"
    id_col = "id"

    print("\n" + "=" * 60)
    print("Porto Seguro Data Check")
    print("=" * 60)
    print(f"train shape: {train_df.shape}")
    print(f"test  shape: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Training split is missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Both train/test must contain id column: {id_col}")

    print(f"target positive rate: {train_df[target_col].mean():.4f}")
    print(f"feature count: {train_df.shape[1] - 2}")


def main() -> None:
    # 这个 `main()` 的作用是准备 ranking-oriented task specification，并把搜索交给 Agent。
    print(f"[Mini-AIDE PortoSeguro] start (max_steps={max_steps}, num_drafts={num_drafts})")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "input", "porto_seguro")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing Porto Seguro train set: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing Porto Seguro test set: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=False)
    submission_relpath = "working/submission-porto-seguro.csv"
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
        submission_path=f"./{submission_relpath}",
    )

    workdir = os.path.join(base_dir, "working_porto_seguro")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "working"), exist_ok=True)

    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
        submission_relpath=submission_relpath,
    )
    agent.run(max_steps=max_steps)

    print("\n[Mini-AIDE PortoSeguro] done")


if __name__ == "__main__":
    main()
