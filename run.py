import sys
import os
import pandas as pd  # 需要用到 pandas 读取表头
from config import timeout, max_steps
# 将当前目录添加到 pythonpath
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent import Agent

def main():
    print("🚀 [系统] Mini-AIDE (COVID-19 增强版) 启动中...")
    
    # 1. 定义文件路径
    # -------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用相对路径更通用，或者你可以保留你的绝对路径
    train_path = os.path.join(base_dir, "input", "train.csv")
    test_path = os.path.join(base_dir, "input", "test.csv")

    # 兼容你的绝对路径写法（如果文件不在当前项目下）
    if not os.path.exists(train_path):
        train_path = "/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv"
        test_path = "/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ 错误：未找到数据文件！")
        print(f"检查路径: {train_path}")
        return

    # 2. 【关键优化】动态读取列名
    # -------------------------------------------------------------
    print("👀 [系统] 正在读取数据表头...")
    try:
        # 只读取表头，不读取数据，速度极快
        train_cols = list(pd.read_csv(train_path, nrows=0).columns)
        test_cols = list(pd.read_csv(test_path, nrows=0).columns)
        
        # 找出目标列（在训练集但不在测试集的列）
        target_candidates = set(train_cols) - set(test_cols)
        print(f"✅ 检测到训练集列数: {len(train_cols)}")
        print(f"✅ 检测到测试集列数: {len(test_cols)}")
        print(f"🎯 推断目标列为: {target_candidates}")
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    # ==========================================
    # 🎯 任务 Prompt (注入真实列名)
    # ==========================================
    task_prompt = f"""
    **任务目标**: 完成 Kaggle 竞赛 "ML2025 Spring HW2: COVID-19 Cases Prediction"。
    这是一个回归任务 (Regression Task)。

    **数据文件路径**:
    - 训练集 (Train): `{train_path}`
    - 测试集 (Test): `{test_path}`

    **【关键信息】数据列结构 (由系统自动检测)**:
    - **训练集列名 (包含目标列)**: 
      {train_cols}
    
    - **测试集列名 (无目标列)**: 
      {test_cols}

    - **预测目标 (Target)**: `tested_positive_day3` (请再次检查上方训练集列名，确保目标列名完全一致)

    **任务要求**:
    1. **数据加载与清洗**:
       - 加载上述路径的 csv 文件。
       - **注意**: 测试集 (`test.csv`) **完全没有** 目标列。在处理测试集时，绝对不要尝试 `drop` 或访问该列，否则会报错 "KeyError"。
       - 训练集和测试集包含 35 个州的独热编码 (AL, AK, AZ...)，这些是特征。
    
    2. **特征工程**:
       - **数据标准化 (必须)**: 神经网络对数值范围敏感，必须使用 `sklearn.preprocessing.StandardScaler` 对所有数值型特征进行标准化 (Fit on Train, Transform on Val/Test)。
       - **特征选择**: 建议使用 `SelectKBest(k=10~20)` 或基于相关性矩阵筛选特征，去除噪音。
       - 尝试构造简单的交互特征（例如 `cli` * `wearing_mask`），但这通过特征选择来决定是否保留。

    3. **建模**:
       - 将训练集切分为 Training Set 和 Validation Set。
       - **模型架构**: 推荐使用 PyTorch 深度神经网络 (DNN)。
         - 结构建议: 2-3 层 Hidden Layers (例如: Input -> 64 -> 32 -> 1)。不要太深，防止过拟合。
         - **关键组件**: 必须使用 `ReLU` 激活函数，并在层之间加入 `BatchNorm1d` 和 `Dropout` (rate=0.1~0.2) 以提高泛化能力。
       - 优化器: Adam 或 AdamW (lr=1e-3 左右)。
       - 损失函数: MSELoss。

    4. **自我评估 (必须)**:
       - 在本地验证集上计算 **RMSE**。
       - **优化方向**: Agent 倾向于最大化分数。请打印 `Score: <1.0 / (1.0 + RMSE)>`，这样 RMSE 越低，Score 越高。
       - **分数标准**: 请打印如 `Score = (1.0 / (1.0 + RMSE)) = 0.5` 这样的格式，并确保这是基于验证集计算的。

    5. **生成提交文件**:
       - 使用模型对 `test.csv` 进行预测。
       - 生成 `submission.csv`，必须包含 `id` 和 `tested_positive` 两列。
       - 不要包含 index。

    **代码约束**:
    - 必须是单文件 Python 脚本。
    - 处理好 pandas 的 SettingWithCopyWarning。

    **关键修正要求 (必须严格遵守)**:
    1. **禁止打乱数据 (No Shuffle)**:
       - 这是一个时间序列任务。在划分训练集/验证集时，**绝对禁止**使用 `shuffle=True`。
       - 必须使用时间切分：取最后 20% 的数据作为验证集 (e.g., `X_train = data[:-val_size]`, `X_val = data[-val_size:]`)。
       - 对应的 `train_test_split` 参数必须是 `shuffle=False`。

    2. **保留异常值 (Keep Outliers)**:
       - **不要**删除目标列中的异常值。疫情爆发时的峰值是非常重要的数据信号，删除会导致模型在测试集上严重低估，导致分数极差。
       - 可以对特征进行缩放 (Scaler)，但不要丢弃行。

    3. **特征工程警告**:
       - 必须处理好 Input Shape，确保输入 DNN 的维度正确。
    """
    # ==========================================
    # ⚙️ Agent 配置
    # ==========================================
    # 步数设为 5，时间给足 3 分钟
    agent = Agent(max_steps=max_steps, timeout=timeout)

    try:
        print("\n" + "="*50)
        print("🤖 Agent 正在接管，目标：预测 tested_positive ...")
        print("="*50)
        
        final_code = agent.solve(task_prompt)
        
        if final_code:
            print("\n✅ [系统] 任务完成！")
            print("📄 提交文件已生成: workspace/submission.csv")
        else:
            print("\n❌ [系统] 任务失败，未生成有效代码。")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ [系统] 用户手动终止程序。")

if __name__ == "__main__":
    main()