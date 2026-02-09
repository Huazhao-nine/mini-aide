import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

# 设置随机种子以保证可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("加载数据...")
    train = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
    test = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')
    
    print(f"训练集形状: {train.shape}, 测试集形状: {test.shape}")
    print(f"目标列: 'tested_positive_day3'")
    
    # 检查目标列是否存在
    if 'tested_positive_day3' not in train.columns:
        raise KeyError("训练集中找不到目标列 'tested_positive_day3'")
    
    # 分离特征和目标
    X = train.drop(columns=['id', 'tested_positive_day3'])
    y = train['tested_positive_day3'].values
    
    # 测试集特征（不要尝试访问目标列）
    X_test = test.drop(columns=['id'])
    
    print(f"特征维度: {X.shape}")
    return X, y, X_test, test['id']

def create_features(X, X_test):
    """创建特征工程"""
    print("创建特征...")
    
    # 复制数据以避免SettingWithCopyWarning
    X_processed = X.copy()
    X_test_processed = X_test.copy()
    
    # 确保数据类型正确
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')
    
    # 处理缺失值（使用均值填充）
    X_processed = X_processed.fillna(X_processed.mean())
    X_test_processed = X_test_processed.fillna(X_test_processed.mean())
    
    # 创建简单的交互特征
    # 选择一些可能有意义的特征进行交互
    interaction_features = []
    
    # 症状相关特征
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in X_processed.columns and f'wearing_mask_7d_day{day}' in X_processed.columns:
            col_name = f'cli_mask_interaction_day{day}'
            X_processed[col_name] = X_processed[f'cli_day{day}'] * X_processed[f'wearing_mask_7d_day{day}']
            X_test_processed[col_name] = X_test_processed[f'cli_day{day}'] * X_test_processed[f'wearing_mask_7d_day{day}']
            interaction_features.append(col_name)
    
    # 信念相关特征
    if 'wbelief_masking_effective_day1' in X_processed.columns and 'wbelief_distancing_effective_day1' in X_processed.columns:
        col_name = 'belief_interaction_day1'
        X_processed[col_name] = X_processed['wbelief_masking_effective_day1'] * X_processed['wbelief_distancing_effective_day1']
        X_test_processed[col_name] = X_test_processed['wbelief_masking_effective_day1'] * X_test_processed['wbelief_distancing_effective_day1']
        interaction_features.append(col_name)
    
    print(f"创建了 {len(interaction_features)} 个交互特征")
    
    return X_processed, X_test_processed

def feature_selection_and_scaling(X, y, X_test):
    """特征选择和标准化"""
    print("特征选择和标准化...")
    
    # 1. 特征选择
    selector = SelectKBest(score_func=f_regression, k=20)  # 选择20个最佳特征
    X_selected = selector.fit_transform(X, y)
    X_test_selected = selector.transform(X_test)
    
    # 获取选择的特征列名
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    print(f"选择了 {len(selected_features)} 个特征")
    print("选择的特征:", list(selected_features))
    
    # 2. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    print(f"标准化后特征形状: {X_scaled.shape}")
    
    return X_scaled, X_test_scaled

def time_series_split(X, y, val_ratio=0.2):
    """时间序列数据划分（不打乱）"""
    print("执行时间序列数据划分...")
    
    # 确保数据按时间顺序排列
    n_samples = len(X)
    val_size = int(n_samples * val_ratio)
    
    X_train = X[:-val_size]
    X_val = X[-val_size:]
    y_train = y[:-val_size]
    y_val = y[-val_size:]
    
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    return X_train, X_val, y_train, y_val

class COVIDPredictor(nn.Module):
    """COVID-19病例预测神经网络"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        super(COVIDPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """训练神经网络模型"""
    print("开始训练模型...")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列不打乱
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = COVIDPredictor(input_dim, hidden_dims=[64, 32], dropout_rate=0.2)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 训练循环
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor)
            val_rmse = torch.sqrt(val_loss).item()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val RMSE: {val_rmse:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, X_val, y_val):
    """评估模型并计算分数"""
    print("评估模型...")
    
    # 转换为PyTorch张量
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor)
        mse = nn.MSELoss()(predictions, y_val_tensor)
        rmse = torch.sqrt(mse).item()
    
    # 计算分数
    score = 1.0 / (1.0 + rmse)
    print(f"验证集RMSE: {rmse:.6f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")
    
    return rmse, score, predictions.numpy()

def main():
    """主函数"""
    print("=" * 60)
    print("ML2025 Spring HW2: COVID-19 Cases Prediction")
    print("=" * 60)
    
    # 1. 加载数据
    X, y, X_test_raw, test_ids = load_and_preprocess_data()
    
    # 2. 创建特征
    X_processed, X_test_processed = create_features(X, X_test_raw)
    
    # 3. 特征选择和标准化
    X_scaled, X_test_scaled = feature_selection_and_scaling(X_processed, y, X_test_processed)
    
    # 4. 时间序列数据划分
    X_train, X_val, y_train, y_val = time_series_split(X_scaled, y, val_ratio=0.2)
    
    # 5. 训练模型
    model = train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # 6. 评估模型
    rmse, score, val_predictions = evaluate_model(model, X_val, y_val)
    
    # 7. 打印最终分数（必须的格式）
    print("\n" + "=" * 40)
    print(f"Score: {score:.6f}")
    print("=" * 40)
    
    # 8. 在测试集上进行预测
    print("\n生成测试集预测...")
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).numpy()
    
    # 9. 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'tested_positive': test_predictions
    })
    
    # 确保预测值非负
    submission['tested_positive'] = submission['tested_positive'].clip(lower=0)
    
    # 保存提交文件
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存: {submission_path}")
    print(f"提交文件形状: {submission.shape}")
    
    # 显示一些预测示例
    print("\n前5个预测示例:")
    print(submission.head())
    
    return score

if __name__ == "__main__":
    main()