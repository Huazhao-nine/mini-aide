import sys
import os
import pandas as pd
import io
import warnings
from config import timeout, max_steps

# 将当前目录添加到 pythonpath
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent import Agent

# ==========================================
# 📚 领域知识 (Domain Knowledge)
# ==========================================
# 汲取 AIDE 原版的优点：赋予 Agent 数据的语义理解能力
COVID_DATA_DESC = """
**数据集语义说明 (Data Dictionary)**:
这是一个关于 COVID-19 的调查数据，用于预测未来的阳性病例比例。

1. **地理信息 (States)**:
   - `AL`, `AK`, `AZ`... : 美国各州的 One-Hot 编码。

2. **核心症状 (Symptoms) - 强预测信号**:
   - `cli`: COVID-like illness (类新冠症状百分比)。
   - `ili`: Influenza-like illness (类流感症状百分比)。
   - `hh_cmnty_cli`: 同住家庭中有人出现类新冠症状的百分比。
   - `nohh_cmnty_cli`: 社区中(非同住)有人出现类新冠症状的百分比。

3. **行为习惯 (Behaviors)**:
   - `wearing_mask_7d`: 过去7天佩戴口罩的比例 (通常与感染率负相关)。
   - `public_transit`: 乘坐公共交通的比例。
   - `work_outside_home_1d`: 外出工作的比例。
   - `shop_indoors`, `restaurant_indoors`: 室内购物/就餐比例。

4. **心理与经济 (Mental & Economic)**:
   - `anxious_7d`, `depressed_7d`: 焦虑/抑郁比例。
   - `worried_finances_7d`: 担心经济状况的比例。

5. **历史检测 (Testing History)**:
   - `tested_positive_14d`: 过去14天阳性率（这是历史趋势特征，非常重要）。
   - `tested_negative_14d`: 过去14天阴性率。

**预测目标**: 预测第3天的检测阳性率 (`tested_positive`)。
"""

def get_data_summary(df: pd.DataFrame, name: str) -> str:
    """生成数据的详细统计摘要"""
    buffer = io.StringIO()
    buffer.write(f"--- Data Summary for {name} ---\n")
    buffer.write(f"Shape: {df.shape}\n")
    
    buffer.write("\n[Columns Info]\n")
    # 获取列名和类型
    df.info(buf=buffer, verbose=True, show_counts=True)
    
    buffer.write("\n[First 3 Rows]\n")
    try:
        buffer.write(df.head(3).to_markdown(index=False, numalign="left", stralign="left"))
    except ImportError:
        buffer.write(df.head(3).to_string(index=False))
    
    buffer.write("\n\n[Categorical Features Sample]\n")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = df[col].unique()
        if len(unique_vals) < 20:
            buffer.write(f"- {col}: {list(unique_vals)}\n")
        else:
            buffer.write(f"- {col}: {list(unique_vals[:5])} ... (Total {len(unique_vals)})\n")
            
    return buffer.getvalue()

def main():
    print(f"🚀 [系统] Mini-AIDE 启动中... (Max Steps: {max_steps})")
    
    # 1. 定义文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "input", "train.csv")
    test_path = os.path.join(base_dir, "input", "test.csv")

    # 兼容绝对路径
    if not os.path.exists(train_path):
        train_path = "/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv"
        test_path = "/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ 错误：未找到数据文件！请检查 input 文件夹。")
        return

    # 2. 读取数据并生成摘要
    print("👀 [系统] 读取数据并生成摘要...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 自动推断 Target 列 (Train 有 but Test 没有的列)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    target_candidates = list(train_cols - test_cols)
    target_col = target_candidates[0] if target_candidates else 'tested_positive'
    
    print(f"✅ 自动检测到的目标列: {target_col}")
    
    train_summary = get_data_summary(train_df, "Train Set")
    # Test 摘要可以简化，节省 Token
   #  test_summary = get_data_summary(test_df, "Test Set")

    # ==========================================
    # 🎯 任务 Prompt
    # ==========================================
    task_prompt = f"""
    **角色**: 你是一位 Kaggle Grandmaster，擅长处理时间序列回归问题。

    **任务目标**: 预测 COVID-19 新增病例百分比。
    **文件路径**:
    - 训练集: `{train_path}`
    - 测试集: `{test_path}`

    **1. 数据认知 (Data Understanding)**:
    - **预测目标 (Target)**: `{target_col}` (请在代码中动态确认，不要硬编码)
    - **语义字典 (Domain Knowledge)**:
      {COVID_DATA_DESC}
    - **数据统计**:
      {train_summary}
    - **测试集结构**:
      与数据统计保持一致，缺失列为预测目标项：{target_col}

    **2. 核心约束 (Critical Constraints - 违反将导致严重的过拟合)**:
    - ❌ **禁止 Shuffle (NO SHUFFLE)**: 这是一个时间序列数据集。在 `train_test_split` 或 `DataLoader` 中，**必须设置 `shuffle=False`**。验证集必须是训练数据的最后 20%。
    - ❌ **禁止删除 Outliers**: 目标变量中的高值代表疫情爆发，是极具价值的信号。严禁删除目标列的异常值。
    - ❌ **禁止 Target Clipping**: 在后处理阶段，**不要**对预测结果设置上限 (Clipping)，除非是处理负值。

    **3. 建模策略 (Modeling Strategy)**:
    - **特征工程**: 
      - 必须使用 `StandardScaler` 对特征进行归一化。
      - 建议使用 `SelectKBest(k=15)` 选择最重要的特征，减少噪音。
      - 尝试构建交互特征 (例如 `cli` * `tested_positive_14d`，代表症状与历史趋势的结合)。
    - **模型架构**: 
      - 推荐: **PyTorch DNN** (Deep Neural Network)。
      - 结构: 简单有效为主 (例如 3 层: Input -> 64 -> 32 -> 1)。
      - 正则化: 必须使用 `BatchNorm1d` 和 `Dropout` (0.1~0.2) 防止过拟合。
      - 损失函数: 推荐尝试 **`nn.L1Loss()` (MAE)** 进行训练（比 MSE 更抗噪），但最终评估指标仍看 RMSE。

    **4. 输出要求**:
    - 程序必须是单文件 Python 脚本。
    - 必须在最后一行打印验证分数，格式严格为: `Score= (1.0 / (1.0 + RMSE)) = 0.8`。
    - 生成 `submission.csv` (包含 `id` 和 预测值列)。

    请编写完整的 Python 代码。
    """

    # ==========================================
    # ⚙️ Agent 配置
    # ==========================================
    # 确保 config.py 中 num_drafts >= 3
    agent = Agent(max_steps=max_steps, timeout=timeout)

    try:
        print("\n" + "="*50)
        print("🤖 Agent 正在接管，准备进行 20 Steps 的深度优化...")
        print("="*50)
        
        final_code = agent.solve(task_prompt)
        
        if final_code:
            print("\n✅ [系统] 任务完成！")
            print("📄 提交文件: workspace/submission.csv")
        else:
            print("\n❌ [系统] 未能生成有效方案。")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ [系统] 用户终止。")

if __name__ == "__main__":
    main()