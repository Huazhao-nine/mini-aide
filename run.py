"""
项目入口。

这个文件对应论文 3.2 节里提到的两个“静态上下文”来源：
1. data preview：把数据规模、列名、目标列等信息整理成文本，持续放入 prompt；
2. task specification：把评估协议、输出契约、防泄漏约束等要求固定下来，
   让后续的 Agent 在每一轮 drafting / debugging / improving 时都遵守同一套规则。

换句话说，`run.py` 不负责搜索本身，而是负责把“问题定义”整理好，再把它交给
`core.agent.Agent` 去执行论文中的 AIDE 主循环。
"""

import io
import os
import warnings

import pandas as pd

from config import max_steps, num_drafts, debug_prob, max_debug_depth, timeout
from core.agent import Agent
from core.interpreter import Interpreter

warnings.filterwarnings("ignore")


def get_data_description(df: pd.DataFrame, full_columns: bool = True) -> str:
    """
    生成数据概览文本。

    论文在 3.2 节明确提到，AIDE 不做完整 EDA，而是在 prompt 里注入一个轻量级的
    data preview，帮助 LLM 理解数据规模、字段布局和目标列位置。这里就是该项目
    对应的实现：把行列数、按 day 划分的特征组、以及完整列名序列化成字符串。
    """
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
        # 列名完整展开后，LLM 可以更可靠地写出显式特征选择、reindex 对齐和防泄漏逻辑。
        buffer.write("## 全量列名列表（可复制粘贴）\n")
        buffer.write("```python\n")
        buffer.write("columns = [\n")
        for c in cols:
            buffer.write(f"    '{c}',\n")
        buffer.write("]\n")
        buffer.write("```\n")

    return buffer.getvalue()


def build_task_prompt(
    train_path: str,
    test_path: str,
    data_desc_str: str,
    allow_tested_positive_history: bool = True,
    task_track: str = "nn_homework",
) -> str:
    """
    拼装任务 prompt。

    这里把项目约束写成“常驻系统规范”：
    - 评估协议：5 折 OOF、以统一的 FINAL_SCORE 为最终指标；
    - 输出契约：最后一行必须打印 FINAL_SCORE，并输出 submission.csv；
    - 防泄漏规则：明确哪些列可用、哪些绝对不能用；
    - 建模轨道：对本课程作业优先要求 PyTorch MLP baseline。

    从论文视角看，这些内容会在后续所有 coding operator 调用中复用，属于“静态任务上下文”。
    """
    if allow_tested_positive_history:
        leak_rules = """
- 允许使用 `tested_positive_day1/day2` 作为历史观测特征（这是过去信息，不是泄漏）
- 允许使用普通 day3 非目标特征
- 禁止把 `tested_positive_day3` 当作输入特征
- 禁止任何从目标列 `tested_positive_day3` 反推/变换得到的特征
- 构造测试集特征时必须显式按列对齐（reindex）
- 若对 y 做归一化/标准化训练，FINAL_SCORE 必须换回原始 0~100 尺度上的 OOF MSE
""".strip()
    else:
        leak_rules = """
- 允许使用普通 day3 特征列
- 禁止把 `tested_positive_day3` 当作输入特征
- 禁止使用任何 `tested_positive_*` 历史列（day1/day2）
- 禁止从任何 `tested_positive_*` 列派生特征
- 构造测试集特征时必须显式按列对齐（reindex）
- 若对 y 做归一化/标准化训练，FINAL_SCORE 必须换回原始 0~100 尺度上的 OOF MSE
""".strip()

    if task_track == "nn_homework":
        model_track_rules = """
## 建模主线（高优先级：神经网络作业）
- DRAFT 首版必须使用 **PyTorch MLP** 作为 baseline（不要用树模型/线性模型作为首版）
- 在前 6 轮优先做神经网络改进：层数/宽度、激活、dropout、norm、optimizer、lr scheduler、batch size、early stopping
- 若某轮要尝试非神经网络模型，仅可作为对照或 debug，不可长期偏离 NN 主线
- 若收到与本规则冲突的“家族提示”，以本规则为准（本任务优先级更高）
""".strip()
    else:
        model_track_rules = """
## 建模主线
- 可在树模型、线性模型、神经网络中选择合适路线
- 以可复现、稳定改进为原则
- 可以使用GPU加速
""".strip()

    return f"""
# 任务：COVID-19 阳性率回归预测（Kaggle 风格）

你需要针对一个表格回归任务，自动迭代产出更强的 baseline。
可使用树模型、线性模型或 PyTorch 深度学习（任选合适路线）。

## 评估协议（必须严格遵守）
- 目标列：`tested_positive_day3`
- 回归任务
- 统一 score 定义：**FINAL_SCORE = 5 折 OOF MSE**（越小越好）

### 重要：验证必须可靠（避免本地过拟合 / Kaggle 崩盘）
1) 你必须使用 **5 折交叉验证（OOF）**：
   - `KFold(n_splits=5, shuffle=True, random_state=42)`
   - 最终以 **OOF MSE** 作为报告指标，也就是 `FINAL_SCORE`
2) 不允许用单一 holdout 作为主要分数来源
3) 若使用超参搜索（如 Optuna），优化目标必须是 **5 折 OOF MSE**
   - 试验次数 <= 20
   - 不允许反复“刷同一个 holdout”
4) 允许集成，但只能基于 **fold 模型**（对 test 预测做 fold 平均）
   - 不允许“同一验证集 top-k trial 加权”这种刷分策略

### 输出契约（严格）
- 脚本必须在最后一行打印：
  `FINAL_SCORE=<number>`
  其中 FINAL_SCORE 必须是 **5 折 OOF MSE**
- 若存在测试集，必须保存提交文件到：
  `./submission.csv`（可额外复制到 `./working/submission.csv`）
  列为：`id,tested_positive_day3`
- 若使用 early stopping / best checkpoint：
  - 必须使用 `copy.deepcopy(model.state_dict())` 保存 best 权重（禁止 `state_dict().copy()`）
  - OOF 预测与测试集提交预测都必须来自“恢复后的 best 权重”

## 数据路径
- 训练集: `{train_path}`
- 测试集: `{test_path}`

## 数据结构
{data_desc_str}

{model_track_rules}

## 硬约束（防泄漏）
{leak_rules}

## 实现要求
- 直接实现训练 + 验证 pipeline，不做冗长 EDA
- 稳定可运行，尽量小步改进
""".strip()


def sanity_check_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    在真正启动搜索前做一次人类可读的数据自检。

    这一步不参与论文中的树搜索算法，但工程上很重要：如果目标列缺失、测试集 `id`
    不存在、目标尺度异常，那么后续所有 LLM 生成的代码都会在错误前提上迭代。
    """
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
    # 这里打印的是搜索超参数；真正的搜索逻辑在 `Agent.run()` 中。
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

    # 先做人类可见的数据确认，再把同样的信息压缩成 prompt 里的 data preview。
    sanity_check_data(train_df, test_df)

    data_desc_str = get_data_description(train_df, full_columns=True)
    allow_tp_history = os.getenv("ALLOW_TESTED_POSITIVE_HISTORY", "1").strip() == "1"
    task_track = os.getenv("TASK_TRACK", "nn_homework").strip().lower()
    print(f"🧭 任务轨道: {task_track}")
    task_prompt = build_task_prompt(
        train_path=train_path,
        test_path=test_path,
        data_desc_str=data_desc_str,
        allow_tested_positive_history=allow_tp_history,
        task_track=task_track,
    )

    workdir = os.path.join(base_dir, "working")
    os.makedirs(workdir, exist_ok=True)

    # 这里正式进入论文 Algorithm 1 的入口：
    # Agent 持有任务描述、工作目录、solution tree（Journal）和 evaluator（Interpreter），
    # 后续每一步都会围绕这几个对象展开。
    agent = Agent(
        task_prompt=task_prompt,
        workdir=workdir,
    )
    agent.run(max_steps=max_steps)

    print("\n✅ [Mini-AIDE] 结束")


if __name__ == "__main__":
    main()
