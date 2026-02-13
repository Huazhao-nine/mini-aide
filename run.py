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
# 📚 领域知识 (Domain Knowledge - 修正版)
# ==========================================
# 基于 PPT 描述，准确反映特征含义
COVID_DATA_DESC = """
**数据集语义说明 (Data Dictionary)**:
任务目标：基于过去3天的调查数据，预测第3天的检测阳性率 (`tested_positive`)。

1. **地理信息 (Geography)**:
   - `AL`, `AZ`, `CA`... (共35个): 美国各州的 One-Hot 编码。

2. **类新冠症状 (COVID-like Illness)**:
   - `cli`: COVID-like illness % (类新冠症状百分比)。
   - `ili`: Influenza-like illness % (类流感症状百分比)。
   - `hh_cmnty_cli`: 同住家庭中有人出现类新冠症状的百分比。
   - `nohh_cmnty_cli`: 社区中(非同住)有人出现类新冠症状的百分比。
   - `wnohh_cmnty_cli`: 加权后的社区症状指标。

3. **行为指标 (Behavior Indicators)**:
   - `wearing_mask_7d`: 过去7天佩戴口罩的比例。
   - `shop_indoors`: 室内购物比例。
   - `restaurant_indoors`: 室内用餐比例。
   - `public_transit`: 公共交通使用比例。
   - `wlarge_event_indoors`: 参加大型室内活动的比例。

4. **信念指标 (Belief Indicators)**:
   - `wbelief_mask_effective`: 相信口罩有效的比例。
   - `wbelief_distancing_effective`: 相信社交距离有效的比例。

5. **心理与环境 (Mental & Environmental)**:
   - `wworried_catch_covid`: 担心感染新冠的比例。
   - `wworried_finance`: 担心经济状况的比例。
   - `wother_masked_public`: 公共场合他人戴口罩的比例。
   - `wother_distanced_public`: 公共场合他人保持距离的比例。
   - `wcovid_vaccinated_friends`: 接种疫苗的朋友比例。

6. **预测目标 (Target)**:
   - `tested_positive_day3`: 检测阳性病例百分比 (第3天的数据)，训练集有这一项，但是测试集没有，需要你预测。
"""

def get_data_summary(df: pd.DataFrame, name: str) -> str:
    """生成数据的详细统计摘要（针对 COVID 数据集优化）"""
    buffer = io.StringIO()
    buffer.write(f"--- Data Summary for {name} ---\n")
    buffer.write(f"Shape: {df.shape}\n")
    
    # 1. 列信息 (显示所有列，让 Agent 知道 State 的 One-Hot 编码存在)
    buffer.write("\n[Columns Info]\n")
    # 如果列数过多 (>60)，只显示简略信息；否则显示完整列表
    if len(df.columns) > 60:
        buffer.write(f"Total Columns: {len(df.columns)}\n")
        buffer.write("Dtypes:\n")
        buffer.write(df.dtypes.value_counts().to_string())
        buffer.write("\n\n(Too many columns to list all details. See Sample Data below.)\n")
    else:
        df.info(buf=buffer, verbose=True, show_counts=True)
    
    # 2. 数据样例 (查看实际数值格式)
    buffer.write("\n[First 3 Rows]\n")
    try:
        buffer.write(df.head(3).to_markdown(index=False, numalign="left", stralign="left"))
    except ImportError:
        buffer.write(df.head(3).to_string(index=False))
    
    # 3. 关键特征统计 (核心修改)
    buffer.write("\n\n[Key Features Statistics]\n")
    
    # 覆盖 PPT 中提到的各类特征代表，以及可能的 Target 名称
    potential_targets = ['tested_positive_day3']
    
    feature_groups = [
        # 症状 (Illness)
        'cli', 'ili', 'hh_cmnty_cli', 
        # 行为 (Behavior)
        'wearing_mask_7d', 'shop_indoors', 'public_transit',
        # 信念 (Belief)
        'wbelief_mask_effective', 'wbelief_distancing_effective',
        # 心理 (Mental)
        'wworried_catch_covid', 'wworried_finance',
        # 环境 (Environmental)
        'wcovid_vaccinated_friends'
    ]
    
    # 合并列表并去重
    key_cols = potential_targets + feature_groups
    
    # 过滤出实际存在的列
    existing_cols = [c for c in key_cols if c in df.columns]
    
    # 如果找不到关键列（比如列名完全不同），则回退到统计所有数值列
    if not existing_cols:
        existing_cols = df.select_dtypes(include=['number']).columns[:15].tolist()

    if existing_cols:
        # 显示统计信息 (Mean, Std, Min, Max) 对 Agent 判断是否需要 Scaling 至关重要
        buffer.write(df[existing_cols].describe().T.to_markdown())
            
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
    # test_df = pd.read_csv(test_path) # Test 不需要读取，省内存
    
    # 自动推断 Target 列 (Train 有 but Test 没有的列)
    # 根据你的PPT，Target就是 tested_positive
    target_col = 'tested_positive_day3'
    
    train_summary = get_data_summary(train_df, "Train Set")

    # ==========================================
    # 🎯 任务 Prompt (优化版)
    # ==========================================
    task_prompt = f"""
    **角色**: 你是一位 Kaggle Grandmaster，擅长处理时间序列回归问题。

    **任务目标**: 预测 COVID-19 新增病例百分比。
    **文件路径**:
    - 训练集: `{train_path}`
    - 测试集: `{test_path}`

    **1. 数据认知**:
    - **Target**: `{target_col}` (0-100 的百分比数值)
    - **语义字典**: {COVID_DATA_DESC}
    - **数据统计**: {train_summary}
    - **重要**: 测试集有tested_positive_day1，tested_positive_day2，没有 `{target_col}` 列。
    - 本任务允许使用 Autoregression (自回归)。即：可以使用 Day 1 和 Day 2 的所有列（包含 Target 列）作为特征来预测 Day 3。这是合法的。

    **2. 核心约束 (Critical Constraints)**:
    - ⚠️ **验证集划分 (Validation Scheme)**: 
      - 必须使用 `sklearn.model_selection.TimeSeriesSplit` (n_splits=5)。
      - **最终分数计算**: 必须计算 5 折验证的**平均 MSE** (Average MSE across folds)，以此作为模型的最终评估指标。
      - 严禁只使用最后一折 (Last Fold Only)，这会导致评估不稳定。
    - ⚠️ **数据泄露**: 
      - 在计算最终打印的 `Score` 时，**严禁**使用“训练集覆盖验证集”的模型。
      - **正确做法**: 只能使用在 `X_train_split` (前80%) 上训练的模型来预测 `X_val` (后20%)。
      - **错误做法 (严禁)**: 在全量数据 `X_train_full` 上训练，然后预测其中的一部分。这会导致分数为 1.0 (过拟合)。
    - ⚠️ **Target处理**: Target (`tested_positive_day3`) 是百分比。
      - 如果使用 **神经网络**：**必须**将 Target 除以 100 或进行 StandardScaler，并在预测后还原。
      - 如果使用 **树模型 (LGBM/XGB)**：通常不需要处理 Target。

    **3. 建模策略 (Modeling Strategy)**:
    - **Draft 阶段强烈建议**: 优先使用 **LightGBM**, **XGBoost** 或 **CatBoost**。
      - 原因：它们在表格数据上通常比未调优的 DNN 表现更好且更稳定（Score > 0.5）。
    - **如果不幸使用了 DNN**:
      - Loss: 必须使用 `nn.MSELoss()`。
      - DataLoader: 训练集必须 `shuffle=True`，验证集 `shuffle=False`。
      - 必须做 Target Scaling。
    - **特征工程 (这是上分的关键!)**: 
      - 必须构造 **Lag Features (滞后特征)**: 例如 `day1` 的数据是 `day2` 的滞后，`day2` 是 `day3` 的滞后。
      - 必须构造 **Rolling Features (滚动特征)**: 例如过去 3 天的均值、标准差、最大值。
      - 尝试 **Target Encoding**: 使用 `State` 的平均阳性率作为特征。
      - 尝试 **Diff/Trend**: (Day2 - Day1) 表示趋势。
    - **模型选择**: 继续优先使用 LightGBM/XGBoost/CatBoost。

    **4. 输出要求**:
    - 程序必须是单文件 Python 脚本。
    - **评分标准**: 请基于 `5-Fold TimeSeriesSplit` 的平均 RMSE 计算最终分数。
    - 必须在最后一行打印验证分数，格式严格为: `Score= (1.0 / (1.0 + MSE)) = 0.8` (示例)。
    - 生成 `submission.csv`。
    - **关于 Score**: 
      - 程序必须打印的 `Score` 应该是 **Out-of-Sample (样本外)** 验证分数的真实反映。
      - 建议直接打印 K-Fold CV 的平均分数，或者在 Split 后的 hold-out set 上的分数。
      - **绝对不要**打印在训练集上的拟合分数！
    - **关于 Submission**:
      - 生成 `submission.csv` 时，**允许且建议**使用全量数据 (Train + Val) 重新训练模型，以利用更多数据预测 Test 集。
      - 但这个全量模型**不能**用来计算上面打印的 Score。

    请编写完整的 Python 代码。
    """

    # ==========================================
    # ⚙️ Agent 配置
    # ==========================================
    # 确保 config.py 中 num_drafts >= 3
    agent = Agent(max_steps=max_steps,data_preview=train_summary)

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